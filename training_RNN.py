from contextlib import contextmanager
from functools import partial
import types
import gc
from pathlib import Path
import json
import random
import math
import torch
import torch.nn.functional as F
import config as runtime_config

from tokenizer import vocab_size
from miniGPT_model_RNN import MiniGPTRecall

from config import (
    DATA_DIR,
    POOL_SIZE,
    EVAL_INTERVAL_TOKENS,
    CHECKPOINT_INTERVAL_TOKENS,
    VALIDATION_BATCHES,
    VALIDATION_SET_SIZE,
    TRAIN_SET_SIZE,
    MEMORY_WINDOWS_PER_SEQUENCE,
    MEMORY_TEST,
    RECALL_LOSS_WEIGHT,
    WARMUP_TOKENS,
    MAX_TOKENS,
    MIN_LEARN_RATE,
    STOP_AFTER_POOL_CT,
    FACT_OFFSET_SCHEDULE,
    USE_FACT_OFFSET_CURRICULUM,
    ALL_FACTS_SINGLE_QA,
    learn_rate,
    batch_size,
    block_size,
    num_epochs,
)

from data_loader import (
    find_data_files,
    split_set,
    make_ram_pool,
    tokenize_set,
    load_memory_set,
    build_context_window,
    get_batch,
    fact_offset_range,
    build_seeded_training_windows,
    build_seeded_memory_sequences,
    materialize_window_slice,
    get_memory_batch,
    carry_count,
    append_to_carry,
    scheduled_memory_samples,
)

from checkpoint import (
    save_checkpoint,
    load_checkpoint,
    checkpoint_exists,
    remove_stale_temp_checkpoint,
)


# ---------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path("./checkpoints")
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

# Keep the recurrent-memory experiment isolated from the standard model
# checkpoints so one training mode can never overwrite/resume the other.
MEMORY_CHECKPOINT_DIR = Path("./checkpoints_memory")
MEMORY_LATEST_CHECKPOINT = MEMORY_CHECKPOINT_DIR / "latest.pt"
MEMORY_BEST_CHECKPOINT = MEMORY_CHECKPOINT_DIR / "best.pt"
MEMORY_SET_PATH = Path("./memory_train_set.jsonl")
MEMORY_SET_VALID_PATH = Path("./memory_validation_set.jsonl")
ACTIVE_FACT_OFFSET_SCHEDULE = getattr(
    runtime_config, "FACT_OFFSET_SCHEDULE", FACT_OFFSET_SCHEDULE)
if not isinstance(USE_FACT_OFFSET_CURRICULUM, bool):
    raise ValueError("USE_FACT_OFFSET_CURRICULUM must be True or False.")
if USE_FACT_OFFSET_CURRICULUM:
    fact_offset_range(0, ACTIVE_FACT_OFFSET_SCHEDULE)  # Validate only when used.




def setup_training():
    model = MiniGPTRecall(vocab_size=vocab_size).to(device)
    learning = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    writer = partial(save_checkpoint, model=model, optimizer=learning, scaler=scaler)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, learning, scaler, writer


def training_files(memory_mode=False):
    files = find_data_files(DATA_DIR)
    train_files, valid_files = split_set(files, train_ratio=0.9, seed=42)
    if VALIDATION_SET_SIZE != 0:
        valid_files = valid_files[:VALIDATION_SET_SIZE]
    if memory_mode:
        # Preserve the supplied file's small-training-set experiment.
        if TRAIN_SET_SIZE != 0:
            train_files = train_files[:VALIDATION_SET_SIZE]
        print(f"Dataset: {len(train_files)} train files / {len(valid_files)} validation files")
    return train_files, valid_files


def compact_memory_sequences(sequences):
    """Keep CPU token storage small; get_memory_batch restores torch.long.

    Existing offset helpers clone/copy into these tensors and preserve token IDs.
    Metadata, sample order, and checkpoint cursors are unchanged.
    """
    if vocab_size > torch.iinfo(torch.int32).max:
        return sequences  # Keep int64 for an unusually large token vocabulary.
    for sample in sequences:
        for key in ('x', 'y'):
            tensor = sample[key]
            if tensor.device.type == 'cpu' and tensor.dtype == torch.int64:
                sample[key] = tensor.to(dtype=torch.int32)
    return sequences


def memory_forward(model, xb, yb, starts, ends):
    """One uninterrupted recurrent graph; final-window writing is unnecessary."""
    memory = None
    logits = None
    language_sum = 0.0
    for index in range(MEMORY_WINDOWS_PER_SEQUENCE):
        final = index == MEMORY_WINDOWS_PER_SEQUENCE - 1
        logits, loss, memory = model(
            xb[:, index], yb[:, index], memory=memory, update_memory=not final)
        language_sum = language_sum + loss
    assert logits is not None
    language_loss = language_sum / MEMORY_WINDOWS_PER_SEQUENCE
    recall_loss = compute_recall_loss(logits, yb[:, -1], starts, ends)
    return logits, memory, language_loss, recall_loss


def cosine_decay(tokens_this_run, warm_up, max_tokens, learn_rate, min_learn_rate):
    if tokens_this_run < warm_up:
        return learn_rate * tokens_this_run / warm_up
    if tokens_this_run >= max_tokens:
        return min_learn_rate
    progress = (tokens_this_run - warm_up) / (max_tokens - warm_up)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learn_rate + coeff * (learn_rate - min_learn_rate)


@torch.no_grad()
def evaluate(model, documents, windows, batch_size, block_size, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    batches = 0
    full_window_count = len(windows) // batch_size * batch_size
    for batch_start in range(0, full_window_count, batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        xb, yb = get_batch(documents, windows, batch_start, batch_size, block_size, device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
            _, loss, _ = model(xb, yb, update_memory=False)
        total_loss += loss.item()
        batches += 1
    model.train()
    if batches == 0:
        raise RuntimeError('Validation set does not contain a complete batch.')
    return total_loss / batches


def compute_recall_loss(logits, targets, answer_start, answer_end):
    """Cross-entropy over only the artificial answer targets in W4."""
    B, T = targets.shape
    if logits.dim() == 3:
        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(B * T, vocab_size)
    elif logits.dim() == 2 and logits.shape[0] == B * T:
        flat_logits = logits
    else:
        raise ValueError(f'Unexpected logits shape for recall loss: {tuple(logits.shape)}')
    positions = torch.arange(T, device=targets.device).unsqueeze(0)
    answer_mask = (positions >= answer_start.unsqueeze(1)) & (positions < answer_end.unsqueeze(1))
    flat_mask = answer_mask.reshape(B * T)
    if not flat_mask.any():
        raise RuntimeError('Recall-loss mask contains no answer tokens.')
    flat_targets = targets.reshape(B * T)
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


@torch.no_grad()
def answer_accuracy_counts(logits, targets, starts, ends):
    """First-answer and teacher-forced exact-answer counts."""
    B, T = targets.shape
    correct = logits.reshape(B, T, -1).argmax(-1).eq(targets)
    positions = torch.arange(T, device=targets.device)[None, :]
    mask = (positions >= starts[:, None]) & (positions < ends[:, None])
    rows = torch.arange(B, device=targets.device)
    return (correct[rows, starts].sum().item(), (correct | ~mask).all(-1).sum().item(), B)


@contextmanager
def disabled_memory_reader(model):
    originals = []

    def zero_forward(module, x, memory):
        return torch.zeros_like(x)
    try:
        for block in model.blocks:
            module = block.memory_attention
            originals.append((module, module.forward))
            module.forward = types.MethodType(zero_forward, module)
        yield
    finally:
        for module, forward in originals:
            module.forward = forward


@torch.no_grad()
def recall_detail(logits, targets, starts, ends):
    """Token counts, first-token accuracy, exact answer accuracy, and CE sum."""
    B, T = targets.shape
    logits = logits.reshape(B, T, -1).float()
    mask = (torch.arange(T, device=targets.device)[None, :] >= starts[:, None]) & (torch.arange(T, device=targets.device)[None, :] < ends[:, None])
    predictions = logits.argmax(-1)
    correct = predictions.eq(targets)
    selected = logits[mask]
    labels = targets[mask]
    lp = F.log_softmax(selected, -1).gather(1, labels[:, None]).squeeze(1)
    rows = torch.arange(B, device=targets.device)
    return dict(
        tokens=labels.numel(),
        first_correct=correct[rows, starts].sum().item(),
        exact_correct=(correct | ~mask).all(-1).sum().item(),
        samples=B,
        ce_sum=-lp.sum().item(),
    )


def aggregate_recall(rows):
    tokens = sum((r['tokens'] for r in rows))
    samples = sum((r['samples'] for r in rows))
    return dict(
        first_accuracy=sum((r['first_correct'] for r in rows)) / samples,
        exact_accuracy=sum((r['exact_correct'] for r in rows)) / samples,
        ce=sum((r['ce_sum'] for r in rows)) / tokens,
        tokens=tokens,
        samples=samples,
    )


@torch.no_grad()
def evaluate_memory(model, sequences, batch_size, device, max_batches=None):
    """Paired validation: same W4 inputs, normal versus disabled memory reader."""
    was_training = model.training
    model.eval()
    total_combined_loss = total_language_loss = total_recall_loss = 0.0
    total_weighted_recall_loss = 0.0
    batches = 0
    details = {key: [] for key in ('Normal', 'NoRead')}
    position_groups = {}
    full_sequence_count = len(sequences) // batch_size * batch_size
    try:
        for batch_start in range(0, full_sequence_count, batch_size):
            if max_batches is not None and batches >= max_batches:
                break
            xb, yb, answer_start, answer_end = get_memory_batch(sequences, batch_start, batch_size, device)
            memory = None
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                final_logits, memory, language_loss, recall_loss = memory_forward(
                    model, xb, yb, answer_start, answer_end)
                weighted = RECALL_LOSS_WEIGHT * recall_loss
                combined = language_loss + weighted
                with disabled_memory_reader(model):
                    off_logits, _, _ = model(xb[:, -1], memory=memory, update_memory=False)
            for label, output in (('Normal', final_logits), ('NoRead', off_logits)):
                details[label].append(recall_detail(output, yb[:, -1], answer_start, answer_end))
            normal_3d = final_logits.reshape(batch_size, block_size, -1)
            for bi, sample in enumerate(sequences[batch_start:batch_start + batch_size]):
                fact = next(
                    (q for q in sample.get('placements', []) if q.get('is_target') and q.get('kind') == 'fact'),
                    None,
                )
                if fact is None:
                    bucket = 'unknown'
                else:
                    pos = fact['local_start']
                    bucket = 'early' if pos < block_size / 3 else 'middle' if pos < 2 * block_size / 3 else 'late'
                row = recall_detail(
                    normal_3d[bi:bi + 1],
                    yb[bi:bi + 1, -1],
                    answer_start[bi:bi + 1],
                    answer_end[bi:bi + 1],
                )
                position_groups.setdefault(bucket, []).append(row)
            total_language_loss += float(language_loss)
            total_recall_loss += recall_loss.item()
            total_weighted_recall_loss += weighted.item()
            total_combined_loss += combined.item()
            batches += 1
    finally:
        model.train(was_training)
    if not batches:
        raise RuntimeError('Memory validation set does not contain a complete batch.')
    conditions = {key: aggregate_recall(rows) for key, rows in details.items()}
    return dict(
        total_loss=total_combined_loss / batches,
        language_loss=total_language_loss / batches,
        recall_loss=total_recall_loss / batches,
        weighted_recall_loss=total_weighted_recall_loss / batches,
        conditions=conditions,
        position_groups={k: aggregate_recall(v) for k, v in position_groups.items()},
    )


def print_memory_diagnostics(metrics):
    for label, row in metrics['conditions'].items():
        print(
            f"  {label:<7} | Answer CE {row['ce']:.4f} | First {100 * row['first_accuracy']:.2f}% | Exact (teacher-forced) {100 * row['exact_accuracy']:.2f}% | n={row['samples']}",
        )
    normal, off = (metrics['conditions']['Normal'], metrics['conditions']['NoRead'])
    print(
        f"  Memory benefit | First gap {100 * (normal['first_accuracy'] - off['first_accuracy']):+.2f} pp | CE improvement {off['ce'] - normal['ce']:+.4f}",
    )
    for label, row in sorted(metrics['position_groups'].items()):
        print(
            f"  Fact position {label:<7} | First {100 * row['first_accuracy']:.2f}% | Exact {100 * row['exact_accuracy']:.2f}% | Answer CE {row['ce']:.4f} | n={row['samples']}",
        )


def memory_training():
    """Dedicated recurrent W0 -> W4 training path selected by MEMORY_TEST."""
    train_files, valid_files = training_files(memory_mode=True)
    train_memories = load_memory_set(path=MEMORY_SET_PATH)
    valid_memories = load_memory_set(path=MEMORY_SET_VALID_PATH)
    model, learning, scaler, write_checkpoint = setup_training()
    print(f'Training artificial memory records: {len(train_memories):,}')
    print(f'Validation artificial memory records: {len(valid_memories):,}')
    if USE_FACT_OFFSET_CURRICULUM:
        print(f'Training fact offset schedule: {ACTIVE_FACT_OFFSET_SCHEDULE}')
        print('Validation: running on same offset schedule as training.')
    else:
        print('Fact offset curriculum disabled: using randomized base placements.')
        print('Validation: fixed seeded randomized placements; no offset relocation.')
    print(f'Associations per sequence: {runtime_config.MEMORY_MIN_ASSOCIATIONS}..'
          f'{runtime_config.MEMORY_MAX_ASSOCIATIONS}')
    if USE_FACT_OFFSET_CURRICULUM and ALL_FACTS_SINGLE_QA:
        print("all W0 facts / one final QA enabled")
    print('QA split isolation: training uses memory_set.jsonl | validation uses memory_validation_set.jsonl')
    print('Building fixed memory validation set...')
    valid_documents = tokenize_set(valid_files, block_size)
    if not valid_documents:
        raise ValueError('Validation data set is empty. Add more validation files or check the dataset path.')
    valid_sequences = build_seeded_memory_sequences(valid_documents, valid_memories, block_size, pool_seed=42)
    if not valid_sequences:
        raise ValueError('No W0 -> W4 validation sequences could be built. The selected validation documents may be too short.')
    built_validation_count = len(valid_sequences)
    # evaluate_memory reads only this prefix. Keeping the same shuffled prefix
    # preserves evaluated examples and the seeded offset helper's RNG order.
    if VALIDATION_BATCHES is not None:
        if VALIDATION_BATCHES <= 0:
            raise ValueError('VALIDATION_BATCHES must be positive or None.')
        validation_limit = VALIDATION_BATCHES * batch_size
        del valid_sequences[validation_limit:]
    compact_memory_sequences(valid_sequences)
    print(f'Memory validation: {len(valid_documents):,} documents | '
          f'{built_validation_count:,} built / {len(valid_sequences):,} retained W0->W4 sequences')
    # Memory sequences own their X/Y storage; original document tensors are no
    # longer needed in memory mode (standard training still needs documents).
    del valid_documents
    gc.collect()
    start_epoch = 0
    resume_pool_index = 0
    resume_sequence_start = 0
    resume_pool_seed = None
    resume_file_order = None
    tokens_seen = 0
    tokens_this_run = 0
    best_val_loss = float('inf')
    remove_stale_temp_checkpoint(MEMORY_LATEST_CHECKPOINT)
    if checkpoint_exists(MEMORY_LATEST_CHECKPOINT):
        state = load_checkpoint(
            MEMORY_LATEST_CHECKPOINT,
            model=model,
            optimizer=learning,
            scaler=scaler,
            device=device,
        )
        start_epoch = state['epoch']
        resume_pool_index = state['pool_index']
        resume_sequence_start = state['batch_start']
        resume_pool_seed = state['pool_seed']
        tokens_seen = state['tokens_seen']
        best_val_loss = state['best_val_loss']
        resume_file_order = [Path(path) for path in state['file_order']]
        print(
            f'Resuming memory checkpoint | epoch={start_epoch + 1} | pool={resume_pool_index} | sequence={resume_sequence_start:,} | tokens={tokens_seen:,}',
        )
    next_eval = (tokens_seen // EVAL_INTERVAL_TOKENS + 1) * EVAL_INTERVAL_TOKENS
    next_checkpoint = (tokens_seen // CHECKPOINT_INTERVAL_TOKENS + 1) * CHECKPOINT_INTERVAL_TOKENS
    global_steps = tokens_seen // (batch_size * MEMORY_WINDOWS_PER_SEQUENCE * block_size)
    running_language_loss = 0.0
    running_recall_loss = 0.0
    running_first_correct = 0
    running_exact_correct = 0
    running_answer_samples = 0
    running_train_batches = 0
    previous_validation = None
    previous_offset_range = None
    for epoch in range(start_epoch, num_epochs):
        print(f'\nMemory Epoch {epoch + 1}/{num_epochs}')
        if epoch == start_epoch and resume_file_order is not None:
            epoch_files = resume_file_order
            first_pool_index = resume_pool_index
        else:
            epoch_files = train_files.copy()
            random.shuffle(epoch_files)
            first_pool_index = 0
        print('Building new RAM pool for memory training...')
        pools_completed_this_run = 0
        for pool_index, pool_paths in enumerate(make_ram_pool(epoch_files, POOL_SIZE)):
            if pool_index < first_pool_index:
                continue
            documents = tokenize_set(pool_paths, block_size)
            if not documents:
                continue
            resuming_this_pool = epoch == start_epoch and resume_file_order is not None and (pool_index == resume_pool_index) and (resume_pool_seed is not None)
            if resuming_this_pool:
                pool_seed = resume_pool_seed
                sequence_cursor = resume_sequence_start
            else:
                pool_seed = random.getrandbits(64)
                sequence_cursor = 0
            sequences = build_seeded_memory_sequences(documents, train_memories, block_size, pool_seed)
            document_count = len(documents)
            del documents
            compact_memory_sequences(sequences)
            if sequence_cursor > len(sequences):
                raise RuntimeError('Checkpoint memory-sequence position is beyond the reconstructed pool. The dataset or memory-window builder may have changed.')
            full_sequence_count = len(sequences) // batch_size * batch_size
            # A resumed pool is reconstructed identically, but already-trained
            # samples need not remain alive. Keep list length/cursors unchanged.
            for consumed in range(sequence_cursor):
                sequences[consumed] = None
            gc.collect()
            print(
                f'Memory Pool {pool_index} | {document_count:,} documents | {len(sequences):,} W0->W4 sequences | starting sequence {sequence_cursor:,}',
            )
            while sequence_cursor + batch_size <= full_sequence_count:
                active_range = (fact_offset_range(tokens_seen, ACTIVE_FACT_OFFSET_SCHEDULE)
                                if USE_FACT_OFFSET_CURRICULUM else None)
                if active_range != previous_offset_range:
                    print(f'Fact offset stage at {tokens_seen:,} tokens | W0 starts {active_range}')
                    previous_offset_range = active_range
                offset_sequences = scheduled_memory_samples(
                    sequences[sequence_cursor:sequence_cursor + batch_size], tokens_seen,
                    ACTIVE_FACT_OFFSET_SCHEDULE, seed=f'{pool_seed}:{sequence_cursor}:{tokens_seen}',
                    use_offset_curriculum=USE_FACT_OFFSET_CURRICULUM)
                xb, yb, answer_start, answer_end = get_memory_batch(offset_sequences, 0, batch_size, device)
                # Only this batch's offset copies are needed now. Release the
                # original samples after stacking; do not detach recurrent states.
                for consumed in range(sequence_cursor, sequence_cursor + batch_size):
                    sequences[consumed] = None
                del offset_sequences
                sequence_cursor += batch_size
                learning.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                    final_logits, memory, language_loss, recall_loss = memory_forward(
                        model, xb, yb, answer_start, answer_end)
                    weighted_recall_loss = RECALL_LOSS_WEIGHT * recall_loss
                    total_loss = language_loss + weighted_recall_loss
                batch_first_correct, batch_exact_correct, batch_samples = answer_accuracy_counts(
                    final_logits,
                    yb[:, -1],
                    answer_start,
                    answer_end,
                )
                lr = cosine_decay(tokens_this_run, WARMUP_TOKENS, MAX_TOKENS, learn_rate, MIN_LEARN_RATE)
                for param_group in learning.param_groups:
                    param_group['lr'] = lr
                scaler.scale(total_loss).backward()
                scaler.step(learning)
                scaler.update()
                global_steps += 1
                tokens_seen += xb.numel()
                tokens_this_run += xb.numel()
                running_language_loss += language_loss.item()
                running_recall_loss += recall_loss.item()
                running_first_correct += batch_first_correct
                running_exact_correct += batch_exact_correct
                running_answer_samples += batch_samples
                running_train_batches += 1
                if tokens_seen >= next_eval:
                    scheduled_validation = scheduled_memory_samples(
                        valid_sequences, tokens_seen, ACTIVE_FACT_OFFSET_SCHEDULE, seed=42,
                        use_offset_curriculum=USE_FACT_OFFSET_CURRICULUM)
                    val_metrics = evaluate_memory(
                        model,
                        scheduled_validation,
                        batch_size,
                        device,
                        max_batches=VALIDATION_BATCHES,
                    )
                    del scheduled_validation
                    avg_language_loss = running_language_loss / running_train_batches
                    avg_recall_loss = running_recall_loss / running_train_batches
                    train_first = running_first_correct / running_answer_samples
                    train_exact = running_exact_correct / running_answer_samples
                    print(f'Memory Step: {global_steps} | Tokens: {tokens_seen:,} | Tokens this run: {tokens_this_run:,} | LR: {lr:.3e}')
                    print(
                        f'  Train interval (dropout on) | LM {avg_language_loss:.4f} | Answer CE {avg_recall_loss:.4f} | First {100 * train_first:.2f}% | Exact (teacher-forced) {100 * train_exact:.2f}%',
                    )
                    print(
                        f"  Valid (dropout off) | LM {val_metrics['language_loss']:.4f} | Combined loss {val_metrics['total_loss']:.4f} | Recall weight {RECALL_LOSS_WEIGHT:g}",
                    )
                    print_memory_diagnostics(val_metrics)
                    current_first = val_metrics['conditions']['Normal']['first_accuracy']
                    if previous_validation is not None:
                        print(
                            f'  Since previous eval | First delta {100 * (current_first - previous_validation):+.2f} pp',
                        )
                    previous_validation = current_first
                    MEMORY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                    with (MEMORY_CHECKPOINT_DIR / 'memory_diagnostics.jsonl').open('a', encoding='utf-8') as log:
                        log.write(
                            json.dumps(dict(tokens=tokens_seen, step=global_steps, lr=lr, recall_weight=RECALL_LOSS_WEIGHT, training_fact_offset_range=active_range, validation_fact_offset_range=active_range, use_fact_offset_curriculum=USE_FACT_OFFSET_CURRICULUM, fact_offset_schedule=ACTIVE_FACT_OFFSET_SCHEDULE if USE_FACT_OFFSET_CURRICULUM else None, validation=val_metrics, train_interval=dict(lm=avg_language_loss, recall_ce=avg_recall_loss, first_accuracy=train_first, exact_accuracy=train_exact, samples=running_answer_samples))) + '\n',
                        )
                    running_language_loss = 0.0
                    running_recall_loss = 0.0
                    running_first_correct = 0
                    running_exact_correct = 0
                    running_answer_samples = 0
                    running_train_batches = 0
                    val_loss = val_metrics['total_loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        write_checkpoint(
                            path=MEMORY_BEST_CHECKPOINT,
                            epoch=epoch,
                            pool_index=pool_index,
                            batch_start=sequence_cursor,
                            tokens_seen=tokens_seen,
                            best_val_loss=best_val_loss,
                            file_order=[str(path) for path in epoch_files],
                            pool_seed=pool_seed,
                            carry_x=None,
                            carry_y=None,
                        )
                        print(
                            f'New best COMBINED-loss memory checkpoint (not recall-only): val_loss={best_val_loss:.4f}',
                        )
                    while tokens_seen >= next_eval:
                        next_eval += EVAL_INTERVAL_TOKENS
                if tokens_seen >= next_checkpoint:
                    write_checkpoint(
                        path=MEMORY_LATEST_CHECKPOINT,
                        epoch=epoch,
                        pool_index=pool_index,
                        batch_start=sequence_cursor,
                        tokens_seen=tokens_seen,
                        best_val_loss=best_val_loss,
                        file_order=[str(path) for path in epoch_files],
                        pool_seed=pool_seed,
                        carry_x=None,
                        carry_y=None,
                    )
                    print(f'Saved latest memory checkpoint at {tokens_seen:,} tokens')
                    while tokens_seen >= next_checkpoint:
                        next_checkpoint += CHECKPOINT_INTERVAL_TOKENS
            write_checkpoint(
                path=MEMORY_LATEST_CHECKPOINT,
                epoch=epoch,
                pool_index=pool_index + 1,
                batch_start=0,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
                file_order=[str(path) for path in epoch_files],
                pool_seed=None,
                carry_x=None,
                carry_y=None,
            )
            skipped = len(sequences) - full_sequence_count
            print(f'Finished memory pool {pool_index} | skipped partial-batch sequences={skipped}')
            del sequences
            gc.collect()
            pools_completed_this_run += 1
            if STOP_AFTER_POOL_CT != 0:
                if pools_completed_this_run >= STOP_AFTER_POOL_CT:
                    print(f'RAM pool count for stop reached: {STOP_AFTER_POOL_CT}, exiting training')
                    return
            resume_sequence_start = 0
            resume_pool_seed = None
        resume_file_order = None
        resume_pool_index = 0
        resume_sequence_start = 0
        resume_pool_seed = None
    print(f'\nMemory training complete. Total target tokens processed: {tokens_seen:,}')


def standard_training():
    train_files, valid_files = training_files(memory_mode=False)
    model, learning, scaler, write_checkpoint = setup_training()
    print('Building fixed validation set...')
    valid_documents = tokenize_set(valid_files, block_size)
    if valid_documents is None or len(valid_documents) == 0:
        raise ValueError('Validation data set is empty. Add more validation files or check the dataset path.')
    valid_windows = build_context_window(valid_documents, block_size, random_offset=False, shuffle=False)
    if valid_windows is None:
        raise ValueError('Validation context windows could not be built.')
    print(f'Validation: {len(valid_documents):,} documents | {len(valid_windows):,} windows')
    start_epoch = 0
    resume_pool_index = 0
    resume_batch_start = 0
    resume_pool_seed = None
    resume_file_order = None
    tokens_seen = 0
    best_val_loss = float('inf')
    carry_x = None
    carry_y = None
    remove_stale_temp_checkpoint(LATEST_CHECKPOINT)
    if checkpoint_exists(LATEST_CHECKPOINT):
        state = load_checkpoint(LATEST_CHECKPOINT, model=model, optimizer=learning, scaler=scaler, device=device)
        start_epoch = state['epoch']
        resume_pool_index = state['pool_index']
        resume_batch_start = state['batch_start']
        resume_pool_seed = state['pool_seed']
        tokens_seen = state['tokens_seen']
        best_val_loss = state['best_val_loss']
        carry_x = state['carry_x']
        carry_y = state['carry_y']
        resume_file_order = [Path(path) for path in state['file_order']]
        print(
            f'Resuming checkpoint | epoch={start_epoch + 1} | pool={resume_pool_index} | window={resume_batch_start:,} | carry={carry_count(carry_x)} | tokens={tokens_seen:,}',
        )
    next_eval = (tokens_seen // EVAL_INTERVAL_TOKENS + 1) * EVAL_INTERVAL_TOKENS
    next_checkpoint = (tokens_seen // CHECKPOINT_INTERVAL_TOKENS + 1) * CHECKPOINT_INTERVAL_TOKENS
    running_train_loss = 0.0
    running_train_batches = 0
    global_steps = tokens_seen // (batch_size * block_size)
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        if epoch == start_epoch and resume_file_order is not None:
            epoch_files = resume_file_order
            first_pool_index = resume_pool_index
        else:
            epoch_files = train_files.copy()
            random.shuffle(epoch_files)
            first_pool_index = 0
        print('Building new RAM pool...')
        for pool_index, pool_paths in enumerate(make_ram_pool(epoch_files, POOL_SIZE)):
            if pool_index < first_pool_index:
                continue
            documents = tokenize_set(pool_paths, block_size)
            if not documents:
                write_checkpoint(
                    path=LATEST_CHECKPOINT,
                    epoch=epoch,
                    pool_index=pool_index + 1,
                    batch_start=0,
                    tokens_seen=tokens_seen,
                    best_val_loss=best_val_loss,
                    file_order=[str(path) for path in epoch_files],
                    pool_seed=None,
                    carry_x=carry_x,
                    carry_y=carry_y,
                )
                continue
            resuming_this_pool = epoch == start_epoch and resume_file_order is not None and (pool_index == resume_pool_index) and (resume_pool_seed is not None)
            if resuming_this_pool:
                pool_seed = resume_pool_seed
                window_cursor = resume_batch_start
            else:
                pool_seed = random.getrandbits(64)
                window_cursor = 0
            windows = build_seeded_training_windows(documents, block_size, pool_seed)
            if window_cursor > len(windows):
                raise RuntimeError('Checkpoint window position is beyond the reconstructed pool. The dataset or context-window logic may have changed.')
            print(
                f'Pool {pool_index} | {len(documents):,} documents | {len(windows):,} windows | starting window {window_cursor:,} | carry {carry_count(carry_x)}',
            )
            while window_cursor < len(windows):
                current_carry = carry_count(carry_x)
                if current_carry > 0:
                    needed = batch_size - current_carry
                    available = len(windows) - window_cursor
                    take = min(needed, available)
                    new_x, new_y = materialize_window_slice(documents, windows, window_cursor, take, block_size)
                    carry_x, carry_y = append_to_carry(carry_x, carry_y, new_x, new_y)
                    window_cursor += take
                    if carry_count(carry_x) < batch_size:
                        break
                    batch_x_cpu = carry_x
                    batch_y_cpu = carry_y
                    carry_x = None
                    carry_y = None
                elif len(windows) - window_cursor >= batch_size:
                    batch_x_cpu, batch_y_cpu = materialize_window_slice(
                        documents,
                        windows,
                        window_cursor,
                        batch_size,
                        block_size,
                    )
                    window_cursor += batch_size
                else:
                    remaining = len(windows) - window_cursor
                    new_x, new_y = materialize_window_slice(
                        documents,
                        windows,
                        window_cursor,
                        remaining,
                        block_size,
                    )
                    carry_x, carry_y = append_to_carry(carry_x, carry_y, new_x, new_y)
                    window_cursor = len(windows)
                    break
                if batch_x_cpu is None or batch_y_cpu is None or batch_x_cpu.shape[0] != batch_size:
                    raise RuntimeError('Internal batching error: optimizer batch is not full.')
                xb = batch_x_cpu.long().to(device, non_blocking=True)
                yb = batch_y_cpu.long().to(device, non_blocking=True)
                learning.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                    _, loss, _ = model(xb, yb, update_memory=False)
                lr = cosine_decay(tokens_seen, WARMUP_TOKENS, MAX_TOKENS, learn_rate, MIN_LEARN_RATE)
                for param_group in learning.param_groups:
                    param_group['lr'] = lr
                scaler.scale(loss).backward()
                scaler.step(learning)
                scaler.update()
                global_steps += 1
                tokens_seen += xb.numel()
                running_train_loss += loss.item()
                running_train_batches += 1
                if tokens_seen >= next_eval:
                    val_loss = evaluate(
                        model,
                        valid_documents,
                        valid_windows,
                        batch_size,
                        block_size,
                        device,
                        max_batches=VALIDATION_BATCHES,
                    )
                    avg_train_loss = running_train_loss / running_train_batches
                    print(
                        f'Training Step: {global_steps} | Tokens: {tokens_seen:,} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}',
                    )
                    running_train_loss = 0.0
                    running_train_batches = 0
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        write_checkpoint(
                            path=BEST_CHECKPOINT,
                            epoch=epoch,
                            pool_index=pool_index,
                            batch_start=window_cursor,
                            tokens_seen=tokens_seen,
                            best_val_loss=best_val_loss,
                            file_order=[str(path) for path in epoch_files],
                            pool_seed=pool_seed,
                            carry_x=carry_x,
                            carry_y=carry_y,
                        )
                        print(f'New best checkpoint: val_loss={best_val_loss:.4f}')
                    while tokens_seen >= next_eval:
                        next_eval += EVAL_INTERVAL_TOKENS
                if tokens_seen >= next_checkpoint:
                    write_checkpoint(
                        path=LATEST_CHECKPOINT,
                        epoch=epoch,
                        pool_index=pool_index,
                        batch_start=window_cursor,
                        tokens_seen=tokens_seen,
                        best_val_loss=best_val_loss,
                        file_order=[str(path) for path in epoch_files],
                        pool_seed=pool_seed,
                        carry_x=carry_x,
                        carry_y=carry_y,
                    )
                    print(f'Saved latest checkpoint at {tokens_seen:,} tokens')
                    while tokens_seen >= next_checkpoint:
                        next_checkpoint += CHECKPOINT_INTERVAL_TOKENS
            if carry_count(carry_x) >= batch_size:
                raise RuntimeError('Carry buffer should never contain a complete batch at a pool boundary.')
            write_checkpoint(
                path=LATEST_CHECKPOINT,
                epoch=epoch,
                pool_index=pool_index + 1,
                batch_start=0,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
                file_order=[str(path) for path in epoch_files],
                pool_seed=None,
                carry_x=carry_x,
                carry_y=carry_y,
            )
            print(f'Finished pool {pool_index} | carry={carry_count(carry_x)} windows')
            del documents
            del windows
            resume_batch_start = 0
            resume_pool_seed = None
        resume_file_order = None
        resume_pool_index = 0
        resume_batch_start = 0
        resume_pool_seed = None
    remaining_contexts = carry_count(carry_x)
    print(f'\nTraining complete. Total target tokens processed: {tokens_seen:,}')
    if remaining_contexts:
        print(
            f'{remaining_contexts} context windows remain in the carry buffer. They were not trained because a final partial optimizer batch would violate the fixed batch-size requirement.',
        )


def main():
    """Select the standard or recurrent-memory training path."""
    if MEMORY_TEST:
        print('MEMORY_TEST=True -> using recurrent-memory training path')
        memory_training()
    else:
        print('MEMORY_TEST=False -> using standard training path')
        standard_training()


if __name__ == '__main__':
    main()