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
    TOKEN_STOP_COUNT,
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

CHECKPOINT_DIR = Path(getattr(runtime_config, "STANDARD_CHECKPOINT_DIR", "./checkpoints"))
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

# Keep the recurrent-memory experiment isolated from the standard model
# checkpoints so one training mode can never overwrite/resume the other.
MEMORY_CHECKPOINT_DIR = Path(getattr(runtime_config, "GUTENBERG_CHECKPOINT_DIR", "./checkpoints_gutenberg_qa"))
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

def migrate_multihead_weights(state, target_state):
    """Initialize only a wholly absent pooling branch; reject partial layouts."""
    state = dict(state)
    prefix = "memory_compressor.multihead_"
    expected = {k for k in target_state if k.startswith(prefix)}
    present = {k for k in state if k.startswith(prefix)}
    if present and present != expected:
        raise ValueError(f"Incompatible multi-head pooling checkpoint: missing {sorted(expected-present)}, unexpected {sorted(present-expected)}")
    if expected and not present:
        for key in expected:
            state[key] = target_state[key].detach().cpu().clone()
            if key.startswith("memory_compressor.multihead_out."):
                state[key].zero_()
        print("Initialized residual multi-head pooling with zero output; fresh optimizer stage.")
    return state


def verify_pooling_resume(path, model):
    """Architecture changes require weights-only migration, not optimizer resume."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", {})
    target = model.state_dict()
    layout_keys = {'memory_compressor.slot_embeddings'} | {
        key for key in target if key.startswith('memory.grus.')}
    present_layout = {'memory_compressor.slot_embeddings'} | {
        key for key in state if key.startswith('memory.grus.')}
    if (present_layout != layout_keys
            or any(key not in state or state[key].shape != target[key].shape for key in layout_keys)):
        raise ValueError('Memory slot count/width or GRU layout changed. Use a new '
                         'checkpoint directory and GUTENBERG_INIT_CHECKPOINT; '
                         'optimizer resume requires the same memory layout.')
    keys = {k for k in target if k.startswith("memory_compressor.multihead_")}
    present = {k for k in state if k.startswith("memory_compressor.multihead_")}
    if keys != present or any(state[k].shape != target[k].shape for k in keys & present):
        raise ValueError("Pooling architecture changed. Use a new checkpoint directory and "
                         "GUTENBERG_INIT_CHECKPOINT for weights-only initialization; "
                         "the old optimizer cannot be resumed with the new parameters.")
    sidecar = Path(path).with_name(Path(path).name + '.writer.json')
    if sidecar.exists():
        recorded = json.loads(sidecar.read_text(encoding='utf-8'))
        if not isinstance(recorded.get('writer_config'), dict):
            raise ValueError('Checkpoint writer metadata is missing its writer_config.')
        verify_stage_writer(dict(pooling_config=recorded['writer_config']), writer_config())


def migrate_routing_weights(state, target_state):
    """Initialize a new refinement branch; preserve trained branches."""
    state = dict(state)

    prefixes = (
        "memory_compressor.refine_gru.",
        "memory_compressor.refine_delta.",
    )

    expected = {
        key for key in target_state
        if key.startswith(prefixes)
    }
    present = {
        key for key in state
        if key.startswith(prefixes)
    }

    if not expected:
        if present:
            raise ValueError(
                "Checkpoint has routing refinement weights, "
                "but the model does not."
            )
        return state

    if not present:
        # Fresh model supplies random internal weights and a
        # zero-initialized final delta projection.
        for key in expected:
            state[key] = target_state[key].detach().cpu().clone()

        print(
            "Initialized iterative routing with zero query corrections."
        )

    elif present != expected:
        raise ValueError(
            "Incomplete or incompatible routing refinement checkpoint. "
            f"Missing: {sorted(expected - present)}; "
            f"unexpected: {sorted(present - expected)}"
        )

    return state

def migrate_encoder_weights(state, target_state):
    """Preserve trained blocks and append identity-initialized encoder blocks.

    Supports legacy memory_encoder.block.* and stacked blocks.N.* layouts.
    Does not fill missing parameters inside an existing block: strict loading
    must still catch incomplete checkpoints and unrelated architecture changes.
    """
    state = dict(state)
    legacy = "memory_encoder.block."
    prefix = "memory_encoder.blocks."

    def block_indices(mapping):
        indices = set()
        for key in mapping:
            if key.startswith(prefix):
                index, separator, _ = key[len(prefix):].partition(".")
                if not separator or not index.isdigit():
                    raise ValueError(f"Invalid encoder block key: {key}")
                indices.add(int(index))
        return indices

    target_indices = block_indices(target_state)
    legacy_keys = [key for key in state if key.startswith(legacy)]
    source_indices = block_indices(state)
    if legacy_keys and source_indices:
        raise ValueError("Checkpoint mixes legacy and stacked encoder layouts.")
    if not target_indices:
        # Existing single-block models still load without encoder migration.
        if source_indices:
            raise ValueError("Cannot load stacked encoder into a legacy encoder.")
        return state
    if legacy_keys:
        for key in legacy_keys:
            state[prefix + "0." + key[len(legacy):]] = state.pop(key)
        source_indices = {0}
        print("Mapped legacy memory encoder to encoder block 0.")
    if not source_indices:
        raise ValueError("Checkpoint has no memory encoder block weights.")
    for label, indices in (("checkpoint", source_indices), ("model", target_indices)):
        if indices != set(range(max(indices) + 1)):
            raise ValueError(f"Noncontiguous encoder blocks in {label}: {indices}")
    if not source_indices.issubset(target_indices):
        raise ValueError("Refusing to discard trained encoder blocks.")

    added = sorted(target_indices - source_indices)
    zero_suffixes = (
        "attention.proj.weight", "attention.proj.bias",
        "ffwd.net.2.weight", "ffwd.net.2.bias",
    )
    for index in added:
        block_prefix = f"{prefix}{index}."
        for suffix in zero_suffixes:
            if block_prefix + suffix not in target_state:
                raise ValueError(
                    f"Cannot initialize identity block: missing {block_prefix + suffix}"
                )
        for key, value in target_state.items():
            if key.startswith(block_prefix):
                value = value.detach().cpu().clone()
                if key[len(block_prefix):] in zero_suffixes:
                    value.zero_()
                state[key] = value
    if added:
        print(
            f"Expanded encoder: {len(source_indices)} -> {len(target_indices)} blocks. "
            f"Preserved trained blocks; added identity-initialized blocks {added}."
        )
    return state


def migrate_slot_gru_weights(state, target_state, old_count, new_count):
    """Expand independent GRUs by copying trained cells, never tying weights.

    Only entirely new slots may be initialized. Incomplete existing cells and
    width changes remain errors. A legacy shared cell can seed all slots.
    """
    if old_count < 1 or new_count < old_count:
        raise ValueError('GRU migration requires a positive, nondecreasing slot count.')
    state = dict(state)
    names = ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')
    target_keys = {f'memory.grus.{slot}.{name}'
                   for slot in range(new_count) for name in names}
    actual_target = {key for key in target_state if key.startswith('memory.grus.')}
    if actual_target != target_keys:
        raise ValueError('Expected one independent GRUCell per target memory slot.')
    shared_keys = {key for key in state if key.startswith('memory.gru.')}
    slot_keys = {key for key in state if key.startswith('memory.grus.')}
    if shared_keys:
        expected_shared = {f'memory.gru.{name}' for name in names}
        if shared_keys != expected_shared or slot_keys:
            raise ValueError('Checkpoint has incomplete or mixed shared/per-slot GRU weights.')
        for slot in range(new_count):
            for name in names:
                key = f'memory.grus.{slot}.{name}'
                value = state[f'memory.gru.{name}']
                if value.shape != target_state[key].shape:
                    raise ValueError(f'Incompatible shared GRU dimensions for {key}.')
                state[key] = value.clone()
        for key in shared_keys:
            del state[key]
        print('Initialized each independent slot GRU from the trained shared GRU.')
        return state
    expected_source = {f'memory.grus.{slot}.{name}'
                       for slot in range(old_count) for name in names}
    if slot_keys != expected_source:
        raise ValueError(
            f'Incomplete/inconsistent source slot GRUs. Missing: '
            f'{sorted(expected_source - slot_keys)}; unexpected: {sorted(slot_keys - expected_source)}')
    for key in expected_source:
        if state[key].shape != target_state[key].shape:
            raise ValueError(f'GRU dimensions changed for {key}; only slot-count expansion is supported.')
    for slot in range(old_count, new_count):
        source = slot % old_count
        for name in names:
            key = f'memory.grus.{slot}.{name}'
            value = state[f'memory.grus.{source}.{name}']
            if value.shape != target_state[key].shape:
                raise ValueError(f'Incompatible added GRU dimensions for {key}.')
            state[key] = value.clone()
    if new_count > old_count:
        print(f'Expanded independent slot GRUs: {old_count} -> {new_count}. '
              'Preserved existing cells; copied trained cells round-robin into new slots.')
    return state


def resize_memory_tensor(source, target, axes, groups=None):
    """Copy matching coordinates, preserving packed GRU gates/pooling heads.

    Uncopied coordinates retain target initialization. This initializes a new
    architecture; it does not claim function preservation after changing width.
    """
    from itertools import product
    groups = groups or {}
    if source.ndim != target.ndim:
        raise ValueError('Cannot migrate tensors with different ranks.')
    if any(source.shape[i] != target.shape[i] and i not in axes
           for i in range(source.ndim)):
        raise ValueError(f'Unrelated dimension changed: {tuple(source.shape)} -> {tuple(target.shape)}.')
    counts = [groups.get(i, 1) for i in range(source.ndim)]
    if any(source.shape[i] % count or target.shape[i] % count
           for i, count in enumerate(counts)):
        raise ValueError('Packed dimension does not divide evenly into gates/heads.')
    result = target.detach().cpu().clone()
    source = source.detach().cpu().to(dtype=result.dtype)
    for coordinate in product(*(range(count) for count in counts)):
        src_slices, dst_slices = [], []
        for axis, group in enumerate(coordinate):
            src_width = source.shape[axis] // counts[axis]
            dst_width = target.shape[axis] // counts[axis]
            width = min(src_width, dst_width)
            src_slices.append(slice(group*src_width, group*src_width+width))
            dst_slices.append(slice(group*dst_width, group*dst_width+width))
        result[tuple(dst_slices)].copy_(source[tuple(src_slices)])
    return result


def migrate_memory_width(state, target, old_count, old_width, new_count, new_width,
                         pooling_heads, allow_shrink=False):
    """Migrate the known memory interfaces; reject unrelated shape changes.

    Slot reduction, when explicitly enabled, retains the first target-count
    slot identities/cells. It does not merge trained slots.
    """
    if (new_count < old_count or new_width < old_width) and not allow_shrink:
        raise ValueError('Memory layout shrinks. Prefer initializing 16x128 from '
                         'your 16x64 baseline. To explicitly truncate slots/channels, '
                         'set GUTENBERG_ALLOW_MEMORY_SHRINK=True.')
    if type(pooling_heads) is not int or pooling_heads < 1:
        raise ValueError('Expected a positive pooling head count.')
    state = dict(state)
    names = ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')
    shared = {k for k in state if k.startswith('memory.gru.')}
    independent = {k for k in state if k.startswith('memory.grus.')}
    expected = {f'memory.grus.{i}.{name}' for i in range(old_count) for name in names}
    if shared:
        if shared != {f'memory.gru.{name}' for name in names} or independent:
            raise ValueError('Incomplete or mixed shared/per-slot GRU checkpoint.')
        for i in range(old_count):
            for name in names:
                state[f'memory.grus.{i}.{name}'] = state[f'memory.gru.{name}'].clone()
        for key in shared:
            del state[key]
    elif independent != expected:
        raise ValueError(f'Incomplete/inconsistent source GRUs. Missing: {sorted(expected-independent)}; '
                         f'unexpected: {sorted(independent-expected)}')
    target_grus = {f'memory.grus.{i}.{name}' for i in range(new_count) for name in names}
    if {k for k in target if k.startswith('memory.grus.')} != target_grus:
        raise ValueError('Target must contain one independent GRU per slot.')
    for key in expected:
        name = key.rsplit('.', 1)[1]
        shape = ((3*old_width, old_width) if name.startswith('weight_') else (3*old_width,))
        if tuple(state[key].shape) != shape:
            raise ValueError(f'Invalid source GRU shape for {key}: expected {shape}.')
    migrated = {key: value for key, value in state.items() if not key.startswith('memory.grus.')}
    for slot in range(new_count):
        donor = slot % old_count
        for name in names:
            key = f'memory.grus.{slot}.{name}'
            axes = {0, 1} if name.startswith('weight_') else {0}
            migrated[key] = resize_memory_tensor(
                state[f'memory.grus.{donor}.{name}'], target[key], axes, {0: 3})
    # name -> (allowed resize axes, packed groups, required source channel sizes)
    layouts = {
        'slot_embeddings': ({0, 1}, {}, {0: old_count, 1: old_width}),
        'slot_norm.weight': ({0}, {}, {0: old_width}),
        'slot_norm.bias': ({0}, {}, {0: old_width}),
        'query.weight': ({0, 1}, {}, {1: old_width}),
        'key.weight': ({0}, {}, {}),
        'value.weight': ({0}, {}, {0: old_width}),
        'multihead_query.weight': ({0, 1}, {0: pooling_heads}, {1: old_width}),
        'multihead_key.weight': ({0}, {0: pooling_heads}, {}),
        'multihead_value.weight': ({0}, {0: pooling_heads}, {0: old_width}),
        'multihead_out.weight': ({0, 1}, {1: pooling_heads}, {0: old_width, 1: old_width}),
        'refine_gru.weight_ih': ({0, 1}, {0: 3}, {0: 3*old_width, 1: old_width}),
        'refine_gru.weight_hh': ({0, 1}, {0: 3}, {0: 3*old_width, 1: old_width}),
        'refine_gru.bias_ih': ({0}, {0: 3}, {0: 3*old_width}),
        'refine_gru.bias_hh': ({0}, {0: 3}, {0: 3*old_width}),
        'refine_delta.0.weight': ({0}, {}, {0: old_width}),
        'refine_delta.0.bias': ({0}, {}, {0: old_width}),
        'refine_delta.1.weight': ({0, 1}, {}, {0: 2*old_width, 1: old_width}),
        'refine_delta.1.bias': ({0}, {}, {0: 2*old_width}),
        'refine_delta.3.weight': ({0, 1}, {}, {0: old_width, 1: 2*old_width}),
        'refine_delta.3.bias': ({0}, {}, {0: old_width}),
    }
    resized = []
    for key, value in list(migrated.items()):
        if key not in target:
            continue  # Encoder migration and strict load handle these keys.
        spec = None
        if key.startswith('memory_compressor.'):
            spec = layouts.get(key[len('memory_compressor.'):])
        elif key.startswith('blocks.'):
            if key.endswith(('.memory_ln.weight', '.memory_ln.bias')):
                spec = ({0}, {}, {0: old_width})
            elif key.endswith(('.memory_attention.key.weight', '.memory_attention.value.weight')):
                spec = ({1}, {}, {1: old_width})
        if spec is None:
            if value.shape != target[key].shape:
                raise ValueError(f'Unrelated shape mismatch for {key}: '
                                 f'{tuple(value.shape)} -> {tuple(target[key].shape)}.')
            continue
        axes, groups, expected_dimensions = spec
        if any(value.shape[axis] != length for axis, length in expected_dimensions.items()):
            raise ValueError(f'Invalid source memory dimensions for {key}.')
        if value.shape != target[key].shape:
            migrated[key] = resize_memory_tensor(value, target[key], axes, groups)
            resized.append(key)
    print(f'Migrated memory layout {old_count}x{old_width} -> {new_count}x{new_width}; '
          f'resized {len(resized)} interfaces plus persistent GRUs. '
          'Added coordinates retain fresh initialization; retraining is required.')
    if new_count < old_count or new_width < old_width:
        print('Explicit shrink enabled: discarded trailing slots/channels; no merging performed.')
    return migrated


def load_initial_weights(model, path):
    """Load stage weights with slot, GRU, and encoder-depth migration.

    Use a new checkpoint directory and GUTENBERG_INIT_CHECKPOINT for
    architecture changes; ordinary optimizer resume requires the same layout.
    """
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint containing model_state_dict.")

    state = dict(checkpoint["model_state_dict"])
    key = "memory_compressor.slot_embeddings"

    if key not in state:
        raise ValueError(f"Checkpoint is missing {key}")

    old_slots = state[key]

    # Fresh model already initialized every slot with random values.
    new_slots = model.state_dict()[key].detach().cpu().clone()

    if old_slots.ndim != 2 or new_slots.ndim != 2:
        raise ValueError("Slot embeddings must be two-dimensional.")

    old_count, old_width = old_slots.shape
    new_count, new_width = new_slots.shape
    if min(old_count, old_width, new_count, new_width) < 1:
        raise ValueError('Memory slot count and width must be positive.')

    allow_shrink = getattr(runtime_config, 'GUTENBERG_ALLOW_MEMORY_SHRINK', False)
    if type(allow_shrink) is not bool:
        raise ValueError('GUTENBERG_ALLOW_MEMORY_SHRINK must be True or False.')
    if old_width != new_width or new_count < old_count:
        target = model.state_dict()
        sidecar = Path(path).with_name(Path(path).name + '.writer.json')
        saved_writer = (json.loads(sidecar.read_text(encoding='utf-8')).get('writer_config', {})
                        if sidecar.exists() else {})
        target_heads = model.memory_compressor.pooling_heads
        source_heads = saved_writer.get('pooling_heads',
            getattr(runtime_config, 'GUTENBERG_INIT_POOLING_HEADS', target_heads))
        if source_heads != target_heads:
            raise ValueError('Width migration requires unchanged pooling head count.')
        if not sidecar.exists():
            print(f'No source writer metadata: assuming {source_heads} pooling heads. '
                  'Set GUTENBERG_INIT_POOLING_HEADS if needed.')
        state = migrate_memory_width(state, target, old_count, old_width,
                                     new_count, new_width, target_heads, allow_shrink)
        state = migrate_encoder_weights(state, target)
        state = migrate_routing_weights(state, target)
        state = migrate_multihead_weights(state, target)
        model.load_state_dict(state, strict=True)
        print(f'Initialized widened/resized memory weights from {path}; '
              'fresh optimizer/scaler/counters required.')
        return

    if new_count < old_count:
        raise ValueError(
            f"Refusing to discard slots: {old_count} -> {new_count}."
        )

    if new_count > old_count:
        # Preserve trained identities for existing slots.
        new_slots[:old_count].copy_(
            old_slots.to(dtype=new_slots.dtype)
        )

        # Additional rows retain their fresh random initialization.
        state[key] = new_slots

        print(
            f"Expanded memory slots: {old_count} -> {new_count}, "
            f"width={new_width}. Preserved {old_count} trained "
            f"slot identities; initialized {new_count - old_count} new ones."
        )

    state = migrate_slot_gru_weights(state, model.state_dict(), old_count, new_count)

    state = migrate_encoder_weights(state, model.state_dict())
    state = migrate_routing_weights(state, model.state_dict())
    state = migrate_multihead_weights(state, model.state_dict())

    # Any unrelated missing keys or shape mismatches still raise an error.
    model.load_state_dict(state, strict=True)

    print(
        f"Initialized model weights from {path}; "
        "starting a new training stage."
    )


def configure_decoder_only_training(model, enabled=False):
    """Toggle decoder-only optimization and bypass memory reads/writes.

    Call before creating the optimizer. The bypass also applies to validation.
    Original methods and requires_grad flags are restored when disabled.
    Embeddings (tied to the LM head), self-attention, FFNs and decoder norms train.
    """
    if not isinstance(enabled, bool):
        raise ValueError('GUTENBERG_DECODER_ONLY must be True or False.')
    if enabled and getattr(model, '_memory_only_enabled', False):
        raise ValueError('Disable memory-only mode before enabling decoder-only mode.')
    previous = getattr(model, '_decoder_only_state', None)
    if previous is not None:
        model.forward = previous['forward']
        for module, forward in previous['readers']:
            module.forward = forward
        for parameter, requires_grad in previous['parameters']:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
        del model._decoder_only_state
    model._decoder_only_enabled = enabled
    if not enabled:
        return

    frozen_modules = [model.memory_encoder, model.memory_compressor, model.memory]
    for block in model.blocks:
        frozen_modules.extend([block.memory_attention, block.memory_ln, block.ln2])
    frozen_ids = {id(p) for module in frozen_modules for p in module.parameters()}
    original_forward = model.forward
    state = dict(forward=original_forward, readers=[],
                 parameters=[(p, p.requires_grad) for p in model.parameters()])
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) not in frozen_ids)
        parameter.grad = None

    def zero_reader(module, x, memory):
        return torch.zeros_like(x)

    for block in model.blocks:
        reader = block.memory_attention
        state['readers'].append((reader, reader.forward))
        reader.forward = types.MethodType(zero_reader, reader)

    def decoder_forward(module, idx, targets=None, memory=None, update_memory=True):
        # Avoid encoder/compressor/GRU work, even when the caller requests a write.
        return original_forward(idx, targets, memory=memory, update_memory=False)

    model.forward = types.MethodType(decoder_forward, model)
    model._decoder_only_state = state
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Decoder-only mode: {trainable:,} trainable parameters; memory reads/writes disabled.')


def configure_memory_only_training(model, enabled=False):
    """Train the memory writer/reader while preserving decoder parameters.

    Call before optimization. Autograd stays enabled through the frozen decoder
    so answer-loss gradients can reach memory. Reads and writes remain active.
    Disable this mode before enabling decoder-only mode on the same instance.
    """
    if not isinstance(enabled, bool):
        raise ValueError('GUTENBERG_MEMORY_ONLY must be True or False.')
    if enabled and getattr(model, '_decoder_only_enabled', False):
        raise ValueError('Disable decoder-only mode before enabling memory-only mode.')
    previous = getattr(model, '_memory_only_parameters', None)
    if previous is not None:
        for parameter, requires_grad in previous:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
        del model._memory_only_parameters
    model._memory_only_enabled = enabled
    if not enabled:
        return

    train_modules = [model.memory_encoder, model.memory_compressor, model.memory]
    for block in model.blocks:
        train_modules.extend([block.memory_attention, block.memory_ln, block.ln2])
    train_ids = {id(p) for module in train_modules for p in module.parameters()}
    model._memory_only_parameters = [(p, p.requires_grad) for p in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in train_ids)
        parameter.grad = None
    verify_memory_only_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Memory-only mode: {trainable:,} trainable / {frozen:,} frozen parameters; '
          'memory reads/writes enabled.')
    print('Frozen: decoder self-attention, FFNs, backbone norms, and tied embedding/output weights.')
    print('Training: memory encoder, compressor, GRU, cross-attention, memory_ln and reader ln2.')


def verify_memory_only_parameters(model):
    """Fail if the frozen backbone or tied output accidentally becomes trainable."""
    if model.lm_head.weight is not model.token_embedding_table.weight:
        raise RuntimeError('Expected tied token embedding and output weights.')
    modules = [model.memory_encoder, model.memory_compressor, model.memory]
    for block in model.blocks:
        modules.extend([block.memory_attention, block.memory_ln, block.ln2])
    allowed = {id(p) for module in modules for p in module.parameters()}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != (id(parameter) in allowed):
            raise RuntimeError(f'Incorrect memory-only trainability: {name}')
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f'Frozen parameter still has a gradient: {name}')
    if model.token_embedding_table.weight.requires_grad:
        raise RuntimeError('Tied embedding/output weights must remain frozen.')


def writer_config():
    """Non-parameter settings needed to reproduce the memory-write schedule."""
    chunk_size = getattr(runtime_config, 'memory_write_chunk_size', block_size)
    if (type(chunk_size) is not int or chunk_size <= 0
            or block_size % chunk_size != 0):
        raise ValueError('memory_write_chunk_size must be a positive integer '
                         'that divides block_size.')
    if (type(MEMORY_WINDOWS_PER_SEQUENCE) is not int
            or not 2 <= MEMORY_WINDOWS_PER_SEQUENCE <= 5):
        raise ValueError('MEMORY_WINDOWS_PER_SEQUENCE must be 2..5 '
                         '(history windows plus one question window).')
    return dict(
        block_size=block_size,
        memory_slots=getattr(runtime_config, 'memory_slots', None),
        memory_slot_dims=getattr(runtime_config, 'memory_slot_dims', None),
        sequence_windows=MEMORY_WINDOWS_PER_SEQUENCE,
        routing_passes=getattr(runtime_config, 'memory_routing_passes', 1),
        pooling_heads=getattr(runtime_config, 'memory_pooling_heads', 4),
        write_chunk_size=chunk_size,
        encoder_scope='full_window',
    )


def verify_stage_writer(previous, current):
    """Old stage markers describe one write per full window."""
    saved = dict(previous.get('pooling_config', {}))
    saved.setdefault('block_size', block_size)
    saved.setdefault('write_chunk_size', saved['block_size'])
    saved.setdefault('encoder_scope', 'full_window')
    saved.setdefault('sequence_windows', 5)
    # Older sidecars lack these fields; checkpoint tensor shapes are also
    # checked by verify_pooling_resume before optimizer restoration.
    for key in ('memory_slots', 'memory_slot_dims'):
        saved.setdefault(key, current[key])
    # Preserve compatibility with markers predating pooling metadata.
    for key in ('routing_passes', 'pooling_heads'):
        saved.setdefault(key, current[key])
    if saved != current:
        raise ValueError(
            f'Memory writer settings changed: saved={saved}, requested={current}. '
            'Use a NEW GUTENBERG_CHECKPOINT_DIR and GUTENBERG_INIT_CHECKPOINT '
            'for weights-only initialization with fresh optimizer/counters.')


def verify_model_writer(model):
    expected = writer_config()
    actual = getattr(model, 'memory_write_chunk_size', block_size)
    if actual != expected['write_chunk_size']:
        raise ValueError(f'Model uses write chunk size {actual}, but config requests '
                         f'{expected["write_chunk_size"]}. Load the updated model source.')
    if actual != block_size and not callable(getattr(model, 'write_memory_features', None)):
        raise ValueError('Chunked writing requires the updated model write_memory_features method.')


def save_with_writer_metadata(*, model, optimizer, scaler, **kwargs):
    """Keep checkpoint.py's format intact; record runtime writer config beside it."""
    result = save_checkpoint(model=model, optimizer=optimizer, scaler=scaler, **kwargs)
    path = Path(kwargs['path'])
    metadata = dict(
        metadata_schema_version=1,
        recall_includes_newline=True,
        writer_config=writer_config(),
        tokens_seen=kwargs.get('tokens_seen'),
        decoder_only=getattr(runtime_config, 'GUTENBERG_DECODER_ONLY', False),
        memory_only=getattr(runtime_config, 'GUTENBERG_MEMORY_ONLY', False),
        evidence_windows=getattr(runtime_config, 'GUTENBERG_EVIDENCE_WINDOWS', None),
        support_window=getattr(runtime_config, 'GUTENBERG_SUPPORT_WINDOW', None),
        support_label=getattr(runtime_config, 'GUTENBERG_SUPPORT_LABEL', True),
    )
    sidecar = path.with_name(path.name + '.writer.json')
    temporary = sidecar.with_name(sidecar.name + '.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    temporary.replace(sidecar)
    return result


def prepare_training_stage(decoder_only, memory_only):
    """Keep fresh baseline initialization separate from same-stage resume.

    A sidecar records the training mode because requires_grad is not stored in
    a model state_dict. Existing unmarked memory-only resumes require a new
    directory: their previous training mode cannot be verified.
    """
    if not isinstance(decoder_only, bool) or not isinstance(memory_only, bool):
        raise ValueError('Training mode flags must be True or False.')
    if decoder_only and memory_only:
        raise ValueError('Enable only one of GUTENBERG_DECODER_ONLY / GUTENBERG_MEMORY_ONLY.')
    mode = 'memory_only' if memory_only else 'decoder_only' if decoder_only else 'joint'
    marker = MEMORY_CHECKPOINT_DIR / 'training_stage.json'
    resuming = checkpoint_exists(MEMORY_LATEST_CHECKPOINT)
    initial = None if resuming else getattr(runtime_config, 'GUTENBERG_INIT_CHECKPOINT', None)
    pooling_config = writer_config()
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if resuming:
            verify_stage_writer(previous, pooling_config)
        if previous.get('mode') != mode:
            raise ValueError('Training mode changed: choose a new GUTENBERG_CHECKPOINT_DIR '
                             'and set GUTENBERG_INIT_CHECKPOINT to the desired baseline.')
    elif resuming and (memory_only or pooling_config['write_chunk_size'] != block_size
                       or pooling_config['sequence_windows'] != 5):
        raise ValueError('Existing checkpoint has no verifiable training-stage marker. To protect '
                         'your baseline, use a NEW GUTENBERG_CHECKPOINT_DIR and set '
                         'GUTENBERG_INIT_CHECKPOINT to the stronger baseline checkpoint.')
    if memory_only and not resuming and not initial:
        raise ValueError('This frozen-decoder stage requires GUTENBERG_INIT_CHECKPOINT '
                         'pointing to your stronger baseline checkpoint.')
    if initial and not Path(initial).is_file():
        raise FileNotFoundError(f'Initial checkpoint not found: {initial}')
    MEMORY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not marker.exists() or not resuming:
        marker.write_text(json.dumps(dict(mode=mode, initial_checkpoint=str(initial) if initial else None, pooling_config=pooling_config),
                                     indent=2) + '\n', encoding='utf-8')
    if resuming:
        print(f'Resuming {mode} stage from {MEMORY_LATEST_CHECKPOINT}; '
              'GUTENBERG_INIT_CHECKPOINT is not reapplied.')
    else:
        print(f'New {mode} stage: fresh optimizer/scaler and training counters.')
    return initial


def build_grouped_optimizer(model):

    # Same modules as memory-only mode
    memory_modules = [model.memory_encoder, model.memory_compressor, model.memory]

    for block in model.blocks:
        memory_modules.extend([block.memory_attention, block.memory_ln, block.ln2])

    # Identify parameters by identity
    memory_ids = {
        id(parameter)
        for module in memory_modules
        for parameter in module.parameters()
        
    }

    memory_params = []
    decoder_params = []

    for parameter in model.parameters():
        if id(parameter) in memory_ids:
            memory_params.append(parameter)
        else:
            decoder_params.append(parameter)
    
    groups = [
        {
            "params": memory_params,
            "name": "memory",
            "lr_scale": runtime_config.MEMORY_LR_SCALE,
        },
        {
            "params": decoder_params,
            "name": "decoder",
            "lr_scale": runtime_config.DECODER_LR_SCALE,
        },
    ]

    for group in groups:
        group["lr"] = learn_rate * group["lr_scale"]

    return torch.optim.AdamW(groups, lr=learn_rate)

def setup_training(initial_checkpoint=None, decoder_only=False, memory_only=False):
    if not isinstance(decoder_only, bool) or not isinstance(memory_only, bool):
        raise ValueError('Training mode flags must be True or False.')
    if decoder_only and memory_only:
        raise ValueError('GUTENBERG_DECODER_ONLY and GUTENBERG_MEMORY_ONLY cannot both be True.')
    model = MiniGPTRecall(vocab_size=vocab_size).to(device)
    verify_model_writer(model)
    if initial_checkpoint:
        load_initial_weights(model, initial_checkpoint)
    configure_decoder_only_training(model, decoder_only)
    configure_memory_only_training(model, memory_only)
    # Keep the original optimizer group layout for checkpoint compatibility.
    # Frozen parameters have grad=None, so AdamW skips their updates/weight decay.
    learning = build_grouped_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    writer = partial(save_with_writer_metadata, model=model, optimizer=learning, scaler=scaler)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, learning, scaler, writer


def training_files(memory_mode=False):
    if memory_mode:
        from gutenberg_qa_loader import dataset_files
        train_files, valid_files = dataset_files(getattr(runtime_config, 'GUTENBERG_QA_DIR', './gutenberg_qa'))
        if TRAIN_SET_SIZE:
            train_files = train_files[:TRAIN_SET_SIZE]
        if VALIDATION_SET_SIZE:
            valid_files = valid_files[:VALIDATION_SET_SIZE]
        return train_files, valid_files
    files = find_data_files(DATA_DIR)
    train_files, valid_files = split_set(files, train_ratio=0.9, seed=42)
    if VALIDATION_SET_SIZE != 0:
        valid_files = valid_files[:VALIDATION_SET_SIZE]
    if TRAIN_SET_SIZE:
        train_files = train_files[:TRAIN_SET_SIZE]
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


def get_gutenberg_memory_batch(sequences, batch_start, batch_size, device):
    """Stack variable-stage examples without data_loader's fixed-layout assumptions."""
    samples = sequences[batch_start:batch_start + batch_size]
    if len(samples) != batch_size:
        raise ValueError('Incomplete Gutenberg memory batch.')
    expected = (MEMORY_WINDOWS_PER_SEQUENCE, block_size)
    if any(tuple(s['x'].shape) != expected or tuple(s['y'].shape) != expected
           for s in samples):
        raise ValueError(f'Builder/trainer sequence shape mismatch; expected {expected}.')
    xb = torch.stack([s['x'] for s in samples]).to(device=device, dtype=torch.long)
    yb = torch.stack([s['y'] for s in samples]).to(device=device, dtype=torch.long)
    starts = torch.tensor([s['answer_start'] for s in samples], device=device, dtype=torch.long)
    ends = torch.tensor([s['answer_end'] for s in samples], device=device, dtype=torch.long)
    return xb, yb, starts, ends


def memory_forward(model, xb, yb, starts, ends):
    """Model handles chunk writes internally; never detach between windows/chunks.

    The full W4 includes gold answers, so writing it would leak answer content.
    """
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
    """Supervise answer tokens AND their existing GPT-2 newline terminator.

    answer_end stays exclusive of the word for answer-only accuracy/export.
    Targets are shifted Y coordinates, so targets[row, answer_end] is the
    newline already appended by gutenberg_qa_loader (token 198).
    """
    B, T = targets.shape
    if (answer_start.shape != (B,) or answer_end.shape != (B,)
            or ((answer_start < 0) | (answer_end <= answer_start)
                | (answer_end >= T)).any()):
        raise ValueError('Recall span must contain an answer and leave room for its newline target.')
    rows = torch.arange(B, device=targets.device)
    if (targets[rows, answer_end] != 198).any():
        raise ValueError('Expected GPT-2 newline token 198 immediately after each recall answer; '
                         'check loader answer_end alignment.')
    if logits.dim() == 3:
        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(B * T, vocab_size)
    elif logits.dim() == 2 and logits.shape[0] == B * T:
        flat_logits = logits
    else:
        raise ValueError(f'Unexpected logits shape for recall loss: {tuple(logits.shape)}')
    positions = torch.arange(T, device=targets.device).unsqueeze(0)
    answer_mask = (positions >= answer_start.unsqueeze(1)) & (positions <= answer_end.unsqueeze(1))
    flat_mask = answer_mask.reshape(B * T)
    if not flat_mask.any():
        raise RuntimeError('Recall-loss mask contains no answer tokens.')
    flat_targets = targets.reshape(B * T)
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


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


def evidence_window_label(sample):
    """Effective evidence window: support copy when present, original otherwise."""
    window = sample.get('source_window')
    if window is None:
        fact = next((q for q in sample.get('placements', [])
                     if q.get('is_target') and q.get('kind') == 'fact'), None)
        window = fact.get('window') if fact else None
    return (f'W{window}' if type(window) is int
            and 0 <= window < MEMORY_WINDOWS_PER_SEQUENCE else 'unknown')


@torch.no_grad()
def recall_rows(logits, targets, starts, ends):
    """Compute answer-only statistics once, transferring one small batch to CPU."""
    B, T = targets.shape
    logits = logits.reshape(B, T, -1)
    mask = ((torch.arange(T, device=targets.device)[None, :] >= starts[:, None]) &
            (torch.arange(T, device=targets.device)[None, :] < ends[:, None]))
    row_ids, positions = mask.nonzero(as_tuple=True)
    # Avoid full-window FP32 conversion, argmax and per-sample softmax calls.
    selected = logits[row_ids, positions].float()
    labels = targets[row_ids, positions]
    correct = selected.argmax(-1).eq(labels)
    losses = F.cross_entropy(selected, labels, reduction='none')
    counts = torch.zeros(B, device=targets.device).scatter_add_(
        0, row_ids, torch.ones_like(losses))
    correct_counts = torch.zeros_like(counts).scatter_add_(0, row_ids, correct.float())
    first = torch.zeros_like(counts).scatter_add_(
        0, row_ids, (correct & positions.eq(starts[row_ids])).float())
    ce = torch.zeros_like(counts).scatter_add_(0, row_ids, losses)
    stats = torch.stack((counts, first, correct_counts.eq(counts).float(), ce), dim=1).cpu().tolist()
    if any(row[0] == 0 for row in stats):
        raise RuntimeError('Validation sample contains no answer tokens.')
    return [dict(tokens=int(n), first_correct=int(f), exact_correct=int(e),
                 ce_sum=c, samples=1) for n, f, e, c in stats]


def aggregate_recall(rows):
    tokens = sum(r['tokens'] for r in rows)
    samples = sum(r['samples'] for r in rows)
    return dict(
        first_accuracy=sum(r['first_correct'] for r in rows) / samples if samples else None,
        exact_accuracy=sum(r['exact_correct'] for r in rows) / samples if samples else None,
        ce=sum(r['ce_sum'] for r in rows) / tokens if tokens else None,
        tokens=tokens, samples=samples,
    )


@torch.no_grad()
def evaluate_memory(model, sequences, batch_size, device, max_batches=None):
    """Normal evaluation; optional NoRead costs one additional W4 forward per batch."""
    was_training = model.training
    model.eval()
    total_language_loss = total_recall_loss = 0.0
    batches = 0
    evaluate_no_read = getattr(runtime_config, 'GUTENBERG_EVAL_NOREAD', False)
    details = {'Normal': []}
    if evaluate_no_read:
        details['NoRead'] = []
    window_groups = {f'W{i}': [] for i in range(MEMORY_WINDOWS_PER_SEQUENCE - 1)}
    full_sequence_count = len(sequences) // batch_size * batch_size
    try:
        for batch_start in range(0, full_sequence_count, batch_size):
            if max_batches is not None and batches >= max_batches:
                break
            xb, yb, starts, ends = get_gutenberg_memory_batch(sequences, batch_start, batch_size, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                logits, memory, language_loss, recall_loss = memory_forward(model, xb, yb, starts, ends)
                if evaluate_no_read:
                    with disabled_memory_reader(model):
                        off_logits, _, _ = model(xb[:, -1], memory=memory, update_memory=False)
            rows = recall_rows(logits, yb[:, -1], starts, ends)
            details['Normal'].extend(rows)
            for sample, row in zip(sequences[batch_start:batch_start + batch_size], rows):
                window_groups.setdefault(evidence_window_label(sample), []).append(row)
            if evaluate_no_read:
                details['NoRead'].extend(recall_rows(off_logits, yb[:, -1], starts, ends))
            total_language_loss += language_loss.item()
            total_recall_loss += recall_loss.item()
            batches += 1
    finally:
        model.train(was_training)
    if not batches:
        raise RuntimeError('Memory validation set does not contain a complete batch.')
    language_loss = total_language_loss / batches
    recall_loss = total_recall_loss / batches
    return dict(
        total_loss=language_loss + RECALL_LOSS_WEIGHT * recall_loss,
        language_loss=language_loss,
        recall_loss=recall_loss,
        conditions={key: aggregate_recall(rows) for key, rows in details.items()},
        window_groups={key: aggregate_recall(rows) for key, rows in window_groups.items()},
        window_group_basis='effective_evidence_window (support copy if present)',
    )


def print_memory_diagnostics(metrics):
    for label, row in metrics['conditions'].items():
        print(f"  {label:<7} | Answer CE {row['ce']:.4f} | First {100 * row['first_accuracy']:.2f}% | "
              f"Exact (teacher-forced) {100 * row['exact_accuracy']:.2f}% | n={row['samples']}")
    for label, row in sorted(metrics['window_groups'].items()):
        if not row['samples']:
            print(f"  Evidence {label:<7} | N/A | n=0")
            continue
        print(f"  Evidence {label:<7} | First {100 * row['first_accuracy']:.2f}% | "
              f"Exact (teacher-forced) {100 * row['exact_accuracy']:.2f}% | n={row['samples']}")


def memory_training():
    """Dedicated recurrent W0 -> W4 training path selected by MEMORY_TEST."""
    print("Setting up training path...")
    print("Recall loss includes answer + newline; validation condition accuracy/CE remain answer-only.")
    decoder_only = getattr(runtime_config, 'GUTENBERG_DECODER_ONLY', False)
    memory_only = getattr(runtime_config, 'GUTENBERG_MEMORY_ONLY', False)
    initial_checkpoint = prepare_training_stage(decoder_only, memory_only)
    train_files, valid_files = training_files(memory_mode=True)
    if USE_FACT_OFFSET_CURRICULUM:
        raise ValueError('Natural evidence cannot be relocated: set USE_FACT_OFFSET_CURRICULUM=False.')
    if type(MEMORY_WINDOWS_PER_SEQUENCE) is not int or not 2 <= MEMORY_WINDOWS_PER_SEQUENCE <= 5:
        raise ValueError('Gutenberg QA supports 2..5 windows including the final question window.')
    from gutenberg_qa_loader import read_records, build_sequences
    global tokenize_set, build_seeded_memory_sequences
    tokenize_set = read_records
    build_seeded_memory_sequences = build_sequences
    train_memories = valid_memories = []
    import hashlib
    signature = hashlib.sha256()
    for split_name, paths in [('train', train_files), ('validation', valid_files)]:
        signature.update(split_name.encode())
        for path in paths:
            signature.update(path.name.encode())
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    signature.update(chunk)
    MEMORY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    signature_path = MEMORY_CHECKPOINT_DIR / 'qa_dataset_signature.txt'
    digest = signature.hexdigest()
    if signature_path.exists() and signature_path.read_text().strip() != digest:
        raise ValueError('QA dataset changed: use a new checkpoint directory.')
    if MEMORY_LATEST_CHECKPOINT.exists() and not signature_path.exists():
        raise ValueError('Checkpoint has no QA dataset signature: use a new directory.')
    signature_path.write_text(digest + '\n')
    # A checkpoint in the NEW stage directory resumes normally. The source
    # checkpoint is used only on the first launch of this stage.
    model, learning, scaler, write_checkpoint = setup_training(
        initial_checkpoint, decoder_only=decoder_only, memory_only=memory_only)
    writing = writer_config()
    if not decoder_only:
        writes_per_window = block_size // writing['write_chunk_size']
        print(f'Memory writer: full-window encoder; {writing["write_chunk_size"]}-token '
              f'feature chunks; {writes_per_window} writes/window, '
              f'{writes_per_window * (MEMORY_WINDOWS_PER_SEQUENCE - 1)} writes/history. '
              f'W{MEMORY_WINDOWS_PER_SEQUENCE - 1} question-window writes disabled.')
    if decoder_only:
        print('Decoder-only evaluation: memory reads and writes are disabled.')
        if getattr(runtime_config, 'GUTENBERG_SUPPORT_WINDOW', None) not in ('question', MEMORY_WINDOWS_PER_SEQUENCE - 1):
            print('Note: use GUTENBERG_SUPPORT_WINDOW="question" for visible-evidence QA training.')
    print(f'Passage-grounded QA: {MEMORY_WINDOWS_PER_SEQUENCE - 1} history window(s), '
          f'one question in W{MEMORY_WINDOWS_PER_SEQUENCE - 1}; evidence-preserving contiguous crops.')
    if USE_FACT_OFFSET_CURRICULUM:
        print(f'Training fact offset schedule: {ACTIVE_FACT_OFFSET_SCHEDULE}')
        print('Validation: running on same offset schedule as training.')
    else:
        print('Fact offset curriculum disabled: using randomized base placements.')
        print('Validation: fixed seeded randomized placements; no offset relocation.')
    print('One queried detail per history; source randomly selected from cached candidates.')
    if USE_FACT_OFFSET_CURRICULUM and ALL_FACTS_SINGLE_QA:
        print("all W0 facts / one final QA enabled")
    print('Train/validation isolation: separate Gutenberg book IDs.')
    print('Building fixed memory validation set...')
    valid_limit = VALIDATION_BATCHES * batch_size if VALIDATION_BATCHES is not None else None
    valid_documents = read_records(valid_files, block_size, max_records=valid_limit)
    if not valid_documents:
        raise ValueError('Validation data set is empty. Add more validation files or check the dataset path.')
    valid_sequences = build_seeded_memory_sequences(valid_documents, valid_memories, block_size, pool_seed=42)
    if not valid_sequences:
        raise ValueError('No validation sequences could be built for this history length.')
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
          f'{built_validation_count:,} built / {len(valid_sequences):,} retained '
          f'{MEMORY_WINDOWS_PER_SEQUENCE}-window sequences')
    # Memory sequences own their X/Y storage; original document tensors are no
    # longer needed in memory mode (standard training still needs documents).
    del valid_documents
    gc.collect()
    # Optional extra full evaluation pass; disabled even if an older config
    # still defines TRAIN_DIAGNOSTIC_BATCHES.
    train_eval_sequences = None
    train_eval_batches = 0
    if getattr(runtime_config, 'GUTENBERG_EVAL_TRAIN_SUBSET', False):
        train_eval_batches = getattr(runtime_config, 'TRAIN_DIAGNOSTIC_BATCHES', 16)
        if type(train_eval_batches) is not int or train_eval_batches <= 0:
            raise ValueError('TRAIN_DIAGNOSTIC_BATCHES must be a positive integer.')
        train_eval_documents = read_records(
            train_files, block_size, max_records=train_eval_batches * batch_size)
        train_eval_sequences = build_seeded_memory_sequences(
            train_eval_documents, train_memories, block_size, pool_seed=42)
        del train_eval_documents
        del train_eval_sequences[train_eval_batches * batch_size:]
        compact_memory_sequences(train_eval_sequences)
        if len(train_eval_sequences) < batch_size:
            raise ValueError('Training diagnostics require at least one complete batch.')
    print('Window accuracy uses the support-copy window when assistance is enabled; '
          'otherwise it uses the original evidence window. Empty windows show N/A.')
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
        verify_pooling_resume(MEMORY_LATEST_CHECKPOINT, model)
        state = load_checkpoint(
            MEMORY_LATEST_CHECKPOINT,
            model=model,
            optimizer=learning,
            scaler=scaler,
            device=device,
        )
        if memory_only:
            verify_memory_only_parameters(model)
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
    running_train_batches = 0
    previous_offset_range = None
    # Evaluate the loaded weights before the first optimizer update of this
    # launch. On resume, retain the restored counters and evaluation schedule.
    print('\nMemory baseline evaluation (before training updates)...', flush=True)
    baseline_random_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=None if device == 'cuda' else []):
            scheduled_validation = scheduled_memory_samples(
                valid_sequences, tokens_seen, ACTIVE_FACT_OFFSET_SCHEDULE, seed=42,
                use_offset_curriculum=USE_FACT_OFFSET_CURRICULUM)
            val_metrics = evaluate_memory(
                model, scheduled_validation, batch_size, device,
                max_batches=VALIDATION_BATCHES)
            del scheduled_validation
    finally:
        random.setstate(baseline_random_state)
    baseline_lr = learning.param_groups[0]['lr']
    print(f'Memory Step: {global_steps} | Tokens: {tokens_seen:,} | '
          f'Tokens this run: {tokens_this_run:,} | LR: {baseline_lr:.3e} | Baseline')
    print(f"  Valid (dropout off) | LM {val_metrics['language_loss']:.4f} | "
          f"Combined loss {val_metrics['total_loss']:.4f} | Recall weight {RECALL_LOSS_WEIGHT:g}")
    print_memory_diagnostics(val_metrics)
    with (MEMORY_CHECKPOINT_DIR / 'memory_diagnostics.jsonl').open('a', encoding='utf-8') as log:
        log.write(json.dumps(dict(
            metrics_schema_version=3, event='baseline', recall_includes_newline=True,
            writer_config=writer_config(),
            support_label=getattr(runtime_config, 'GUTENBERG_SUPPORT_LABEL', True),
            tokens=tokens_seen, tokens_this_run=tokens_this_run, step=global_steps,
            lr=baseline_lr, recall_weight=RECALL_LOSS_WEIGHT,
            decoder_only=decoder_only, memory_only=memory_only,
            evidence_windows=getattr(runtime_config, 'GUTENBERG_EVIDENCE_WINDOWS', None),
            support_window=getattr(runtime_config, 'GUTENBERG_SUPPORT_WINDOW', None),
            validation=val_metrics, train_subset=None, train_interval=None)) + '\n')
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
                f'Memory Pool {pool_index} | {document_count:,} documents | {len(sequences):,} '
                f'{MEMORY_WINDOWS_PER_SEQUENCE}-window sequences | starting sequence {sequence_cursor:,}',
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
                xb, yb, answer_start, answer_end = get_gutenberg_memory_batch(offset_sequences, 0, batch_size, device)
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
                lr = cosine_decay(tokens_seen, WARMUP_TOKENS, MAX_TOKENS, learn_rate, MIN_LEARN_RATE)
                for group in learning.param_groups:
                    group["lr"] = lr * group["lr_scale"]
                scaler.scale(total_loss).backward()
                scaler.step(learning)
                scaler.update()
                global_steps += 1
                tokens_seen += xb.numel()
                tokens_this_run += xb.numel()
                running_language_loss += language_loss.item()
                running_recall_loss += recall_loss.item()
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
                    train_metrics = None
                    if train_eval_sequences is not None:
                        train_metrics = evaluate_memory(
                            model, train_eval_sequences, batch_size, device,
                            max_batches=train_eval_batches)
                    avg_language_loss = running_language_loss / running_train_batches
                    avg_recall_loss = running_recall_loss / running_train_batches
                    print(f'Memory Step: {global_steps} | Tokens: {tokens_seen:,} | Tokens this run: {tokens_this_run:,} | LR: {lr:.3e}')
                    print(
                        f'  Train interval | LM {avg_language_loss:.4f} | Answer CE {avg_recall_loss:.4f}',
                    )
                    if train_metrics is not None:
                        print(f"  Train subset (dropout off) | LM {train_metrics['language_loss']:.4f}")
                        print_memory_diagnostics(train_metrics)
                    print(
                        f"  Valid (dropout off) | LM {val_metrics['language_loss']:.4f} | Combined loss {val_metrics['total_loss']:.4f} | Recall weight {RECALL_LOSS_WEIGHT:g}",
                    )
                    print_memory_diagnostics(val_metrics)
                    MEMORY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                    with (MEMORY_CHECKPOINT_DIR / 'memory_diagnostics.jsonl').open('a', encoding='utf-8') as log:
                        log.write(
                            json.dumps(dict(metrics_schema_version=3, recall_includes_newline=True, writer_config=writer_config(), support_label=getattr(runtime_config, "GUTENBERG_SUPPORT_LABEL", True), tokens=tokens_seen, step=global_steps, lr=lr, recall_weight=RECALL_LOSS_WEIGHT, decoder_only=decoder_only, memory_only=memory_only, evidence_windows=getattr(runtime_config, "GUTENBERG_EVIDENCE_WINDOWS", None), support_window=getattr(runtime_config, "GUTENBERG_SUPPORT_WINDOW", None), validation=val_metrics, train_subset=train_metrics, train_interval=dict(lm=avg_language_loss, recall_ce=avg_recall_loss))) + '\n',
                        )
                    running_language_loss = 0.0
                    running_recall_loss = 0.0
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
                    if TOKEN_STOP_COUNT is not None and tokens_seen >= TOKEN_STOP_COUNT:
                        print("Token stop count reached, exiting training...")
                        return
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


def standard_settings():
    """Standard LM training uses independent stage controls and checkpoints."""
    decoder_only = getattr(runtime_config, 'STANDARD_DECODER_ONLY', False)
    if type(decoder_only) is not bool:
        raise ValueError('STANDARD_DECODER_ONLY must be True or False.')
    if getattr(runtime_config, 'STANDARD_MEMORY_ONLY', False):
        raise ValueError('Standard independent prose windows do not train recurrent recall. '
                         'Use MEMORY_TEST=True for memory-only training.')
    stop = TOKEN_STOP_COUNT
    pools = STOP_AFTER_POOL_CT
    if stop is not None and (type(stop) is not int or stop < 1):
        raise ValueError('TOKEN_STOP_COUNT must be positive or None.')
    if type(pools) is not int or pools < 0:
        raise ValueError('STOP_AFTER_POOL_CT must be nonnegative.')
    if VALIDATION_BATCHES is not None and (type(VALIDATION_BATCHES) is not int or VALIDATION_BATCHES < 1):
        raise ValueError('VALIDATION_BATCHES must be positive or None.')
    if EVAL_INTERVAL_TOKENS <= 0 or CHECKPOINT_INTERVAL_TOKENS <= 0:
        raise ValueError('Evaluation/checkpoint token intervals must be positive.')
    return decoder_only, stop, pools


def compact_standard_documents(documents):
    """Narrow CPU int64 tensors without changing document/container layout.

    Python token lists stay lists because external data_loader implementations
    may concatenate them. No token IDs are changed. GPU batches remain long.
    """
    if vocab_size > torch.iinfo(torch.int32).max:
        return documents
    def compact(value):
        if isinstance(value, torch.Tensor):
            if value.device.type == 'cpu' and value.dtype == torch.int64:
                return value.to(dtype=torch.int32)
            return value
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        if isinstance(value, list):
            if not value or isinstance(value[0], int):
                return value
            return [compact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(compact(item) for item in value)
        return value
    return compact(documents)


def build_standard_validation(paths):
    """Materialize only evaluated windows; release each book before the next.

    Uses the installed loader's non-random, non-shuffled per-document windows.
    VALIDATION_BATCHES bounds retained token storage (None retains all windows).
    No giant validation corpus or unused window list remains resident.
    """
    limit = VALIDATION_BATCHES * batch_size if VALIDATION_BATCHES is not None else None
    xs, ys, retained, books = [], [], 0, 0
    for path in paths:
        documents = tokenize_set([path], block_size)
        if documents is None or len(documents) == 0:
            continue
        documents = compact_standard_documents(documents)
        windows = build_context_window(documents, block_size, random_offset=False, shuffle=False)
        if windows is None:
            raise ValueError('Validation context windows could not be built.')
        books += len(documents)
        take = len(windows) if limit is None else min(len(windows), limit-retained)
        for start in range(0, take, batch_size):
            count = min(batch_size, take-start)
            x, y = materialize_window_slice(documents, windows, start, count, block_size)
            if x is None or y is None or x.shape[0] != count or y.shape != x.shape:
                raise RuntimeError('Invalid validation window materialization.')
            dtype = torch.int32 if vocab_size <= torch.iinfo(torch.int32).max else torch.int64
            xs.append(x.to(device='cpu', dtype=dtype))
            ys.append(y.to(device='cpu', dtype=dtype))
            retained += count
        del documents, windows
        if limit is not None and retained >= limit:
            break
    usable = retained // batch_size * batch_size
    if usable == 0:
        raise ValueError('Validation set has no complete optimizer-size batch.')
    x = torch.cat(xs, dim=0)[:usable].contiguous()
    y = torch.cat(ys, dim=0)[:usable].contiguous()
    del xs, ys
    gc.collect()
    print(f'Standard validation: {books:,} books inspected | {usable:,} cached windows | '
          f'{usable*block_size*2*x.element_size()/1024**2:.1f} MiB token tensors')
    return x, y


@torch.no_grad()
def evaluate_standard_cached(model, x_cpu, y_cpu):
    previous_mode = model.training
    model.eval()
    total, batches = 0.0, 0
    try:
        for start in range(0, len(x_cpu), batch_size):
            xb = x_cpu[start:start+batch_size].to(device=device, dtype=torch.long)
            yb = y_cpu[start:start+batch_size].to(device=device, dtype=torch.long)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                _, loss, _ = model(xb, yb, update_memory=False)
            total += loss.item()
            batches += 1
    finally:
        model.train(previous_mode)
    return total / batches


def prepare_standard_stage(decoder_only):
    """Do not restore an optimizer across a change of training mode."""
    if CHECKPOINT_DIR.resolve() == MEMORY_CHECKPOINT_DIR.resolve():
        raise ValueError('STANDARD_CHECKPOINT_DIR must differ from GUTENBERG_CHECKPOINT_DIR.')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    remove_stale_temp_checkpoint(LATEST_CHECKPOINT)
    resuming = checkpoint_exists(LATEST_CHECKPOINT)
    initial = None if resuming else getattr(runtime_config, 'STANDARD_INIT_CHECKPOINT', None)
    if initial and not Path(initial).is_file():
        raise FileNotFoundError(f'Initial checkpoint not found: {initial}')
    marker = CHECKPOINT_DIR / 'standard_training_stage.json'
    current = dict(mode='decoder_only' if decoder_only else 'standard', block_size=block_size,
                   batch_size=batch_size, pooling_config=writer_config(),
                   train_set_size=TRAIN_SET_SIZE, validation_set_size=VALIDATION_SET_SIZE,
                   validation_batches=VALIDATION_BATCHES,
                   data_dir=str(Path(DATA_DIR).resolve()), pool_size=POOL_SIZE,
                   validation_layout='per_document_fixed_v1')
    if resuming:
        if marker.exists():
            previous = json.loads(marker.read_text(encoding='utf-8'))
            if previous != current:
                raise ValueError('Standard training stage settings changed. Use a new '
                                 'STANDARD_CHECKPOINT_DIR and STANDARD_INIT_CHECKPOINT for weights only.')
        elif decoder_only:
            raise ValueError('Unmarked legacy checkpoint cannot verify decoder-only optimizer layout. '
                             'Use a new STANDARD_CHECKPOINT_DIR and STANDARD_INIT_CHECKPOINT.')
        else:
            print('Resuming legacy standard checkpoint; previous validation layout is unverified.')
    if not resuming or not marker.exists():
        marker.write_text(json.dumps(current, indent=2)+'\n', encoding='utf-8')
    return initial


def save_standard_checkpoint(*, model, optimizer, scaler, **kwargs):
    result = save_checkpoint(model=model, optimizer=optimizer, scaler=scaler, **kwargs)
    path = Path(kwargs['path'])
    metadata = dict(metadata_schema_version=1, training_path='standard',
                    decoder_only=getattr(model, '_decoder_only_enabled', False),
                    memory_only=False, writer_config=writer_config(),
                    tokens_seen=kwargs.get('tokens_seen'))
    sidecar = path.with_name(path.name+'.writer.json')
    temporary = sidecar.with_name(sidecar.name+'.tmp')
    temporary.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    temporary.replace(sidecar)
    return result


def standard_training():
    decoder_only, token_stop, pool_stop = standard_settings()
    initial_checkpoint = prepare_standard_stage(decoder_only)
    train_files, valid_files = training_files(memory_mode=False)
    if not train_files or not valid_files:
        raise ValueError('Standard training requires nonempty training and validation splits.')
    model, learning, scaler, _ = setup_training(initial_checkpoint, decoder_only=decoder_only)
    write_checkpoint = partial(save_standard_checkpoint, model=model, optimizer=learning, scaler=scaler)
    if not decoder_only:
        print('Standard LM mode: independent windows, memory writes disabled; '
              'memory reader remains enabled. Use STANDARD_DECODER_ONLY=True to bypass it.')
    print('Building bounded fixed validation cache...')
    valid_x, valid_y = build_standard_validation(valid_files)
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
        verify_pooling_resume(LATEST_CHECKPOINT, model)
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
    tokens_at_launch = tokens_seen
    pools_completed_this_run = 0
    if token_stop is not None and tokens_seen >= token_stop:
        print(f'Token stop already reached: {tokens_seen:,} >= {token_stop:,}; no updates performed.')
        return
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
            if documents is None or len(documents) == 0:
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
            documents = compact_standard_documents(documents)
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
                    param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
                scaler.scale(loss).backward()
                scaler.step(learning)
                scaler.update()
                global_steps += 1
                tokens_seen += xb.numel()
                running_train_loss += loss.item()
                running_train_batches += 1
                if tokens_seen >= next_eval:
                    val_loss = evaluate_standard_cached(model, valid_x, valid_y)
                    avg_train_loss = running_train_loss / running_train_batches
                    print(
                        f'Training Step: {global_steps} | Tokens: {tokens_seen:,} | Tokens this run: {tokens_seen-tokens_at_launch:,} | LR: {lr:.3e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}',
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
                stopping = token_stop is not None and tokens_seen >= token_stop
                if tokens_seen >= next_checkpoint or stopping:
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
                    if stopping:
                        print(f'Token stop reached at completed batch: {tokens_seen:,} '
                              f'(requested {token_stop:,}); checkpoint saved for resume.')
                        return
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
            # Drop last CPU/GPU batches and graph outputs before loading another pool.
            batch_x_cpu = batch_y_cpu = new_x = new_y = xb = yb = loss = _ = None
            gc.collect()
            pools_completed_this_run += 1
            if pool_stop and pools_completed_this_run >= pool_stop:
                print(f'RAM pool stop reached: {pools_completed_this_run}; checkpoint saved.')
                return
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
