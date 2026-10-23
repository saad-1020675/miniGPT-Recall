"""Generate W0-W3 prose, write each full window once, then answer in W4.

Run beside config.py, checkpoint.py and miniGPT_model_RNN.py:
    python generate_RNN.py --prompt "One dark stormy night" --compare-memory
    python generate_RNN.py --mode paste --max-answer-tokens 24

The history length includes the seed prompt. Default: 4 * config.block_size.
Use --raw-question-prompt to supply the dataset's exact cloze prompt format.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random

# Loaded only in main so layout/sequence helpers can be tested without a GPU
# or PyTorch installation. Actual generation requires torch and tiktoken.
torch = None
 

STAGE2_FILLER_TEXT = (
    "The afternoon passed quietly in the small town. People moved between "
    "the shops and houses, carrying baskets, tools, letters, and packages. "
    "A light wind crossed the road and stirred the leaves along the fences. "
    "Near the market square, several merchants packed away their goods while "
    "children walked home from school. A wagon rolled slowly toward the bridge.\n"
)


def read_multilines(prompt):
    print(prompt)
    print("Type END on its own line when finished.")
    lines = []
    while True:
        line = input()
        if line == "END":
            return "\n".join(lines)
        lines.append(line)


def generate_history(forward, sample, prompt_tokens, width, count, on_window=None):
    """Framework-free orchestration; forward(tokens, memory, write) -> logits,state.

    Decode against fixed incoming memory inside each window. Only completed,
    nonoverlapping windows are written. The last logit of a completed window
    predicts the next window's first token, as in shifted-target training.
    """
    if width < 1 or count < 1:
        raise ValueError("Window width and history-window count must be positive.")
    if not prompt_tokens or len(prompt_tokens) > width:
        raise ValueError("Seed prompt must contain 1..block_size tokens.")
    memory = None
    history = []
    window = list(prompt_tokens)
    for index in range(count):
        while len(window) < width:
            logits, _ = forward(window, memory, False)
            window.append(sample(logits))
        logits, memory = forward(window, memory, True)
        history.extend(window)
        if on_window:
            on_window(index, list(window))
        if index + 1 < count:
            window = [sample(logits)]
    return history, memory


def ingest_history(forward, tokens, width, count):
    if len(tokens) != width * count:
        raise ValueError(f"Expected exactly {width * count} history tokens.")
    memory = None
    for start in range(0, len(tokens), width):
        _, memory = forward(tokens[start:start + width], memory, True)
    return memory


def build_question_prefix(enc, question, width, answer_budget, rng,
                          offset=None, raw=False):
    """Use the loader's filler -> QA layout, stopping before the first answer.

    The answer budget replaces gold-answer length for placement. The answer
    itself is never supplied. A random valid insertion offset mirrors training;
    --question-offset sets a controlled position instead.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if answer_budget < 1:
        raise ValueError("Answer budget must be positive.")
    text = question if raw else (
    "\nQuestion: Earlier in the passage, what word completes this sentence?\n"
    + question.strip()
    + "\nAnswer:"
    )
    prompt = enc.encode_ordinary(text)
    available = width - len(prompt) - answer_budget - 1  # reserve newline
    if available < 0:
        raise ValueError("Question plus answer budget exceeds block_size; shorten either.")
    filler = enc.encode_ordinary(STAGE2_FILLER_TEXT)
    filler = (filler * (width // len(filler) + 2))[:width]
    offsets = list(range(available + 1)) if offset is None else [offset]
    if offset is not None and not 0 <= offset <= available:
        raise ValueError(f"Question offset must be in 0..{available}.")
    if offset is None:
        rng.shuffle(offsets)
    for position in offsets:
        try:
            enc.decode(filler[:position], errors="strict")
        except UnicodeDecodeError:
            continue
        return filler[:position] + prompt, position, text
    raise ValueError("No character-safe filler boundary for this question offset.")


def sample_next(logits, temperature, top_k, generator):
    if temperature == 0:
        return int(logits.argmax(dim=-1).item())
    scores = logits.float() / temperature
    if top_k:
        k = min(top_k, scores.shape[-1])
        cutoff = torch.topk(scores, k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


@contextmanager
def memory_reads(model, enabled=True):
    handles = []
    try:
        if not enabled:
            # Zero memory is not a true NoRead control: learned LN biases and
            # attention projections can still produce nonzero contributions.
            def zero_output(module, inputs, output):
                return torch.zeros_like(output)
            for block in model.blocks:
                handles.append(block.memory_attention.register_forward_hook(zero_output))
            if not handles:
                raise ValueError("No decoder memory_attention modules found for NoRead.")
        yield
    finally:
        for handle in handles:
            handle.remove()


def answer_question(forward, enc, prefix, memory, budget):
    """Greedy free generation; fixed memory, no teacher forcing or W4 writes."""
    answer = []
    first_logits = None
    for _ in range(budget):
        logits, _ = forward(prefix + answer, memory, False)
        if first_logits is None:
            first_logits = logits
        token = int(logits.argmax(dim=-1).item())
        if token == enc.eot_token:
            break
        answer.append(token)
        # Training ends answers with a newline. Inspect bytes so partial UTF-8
        # characters do not break the stopping logic.
        if b"\n" in b"".join(enc.decode_single_token_bytes(t) for t in answer):
            break
    text = enc.decode(answer).split("\n", 1)[0].strip()
    return text, answer, first_logits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["generate", "paste", "replay"], default="generate")
    parser.add_argument("--prompt", help="Story seed; otherwise requested interactively.")
    parser.add_argument("--history-file", type=Path, help="UTF-8 history for --mode paste.")
    parser.add_argument("--history-windows", type=int, default=4,
                        help="Default 4, matching W0-W3. Other values test a different horizon.")
    parser.add_argument("--question", help="Otherwise requested after prose is printed.")
    parser.add_argument("--raw-question-prompt", action="store_true",
                        help="Use question text verbatim, including your cloze/answer prefix.")
    parser.add_argument("--question-offset", type=int, help="W4 filler length; default seeded random.")
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--expected-answer", help="Optional first-token diagnostics; never supplied as input.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Prose sampling; 0 = greedy.")
    parser.add_argument("--top-k", type=int, default=40, help="Prose top-k; 0 = unrestricted.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to config.MEMORY_CHECKPOINT_PATH.")
    parser.add_argument("--compare-memory", action="store_true",
                        help="Also generate NoRead and zero-state answers from the identical W4 prefix.")
    parser.add_argument("--save-json", type=Path, help="Save history IDs, answers, and inference settings.")
    parser.add_argument("--manifest", type=Path, help="Materialized validation manifest for --mode replay.")
    parser.add_argument("--sample-index", type=int, help="Replay one zero-based sample; omit for all samples.")
    parser.add_argument("--limit", type=int, help="Maximum replay samples.")
    args = parser.parse_args()
    if args.mode == "replay" and args.manifest is None:
        parser.error("--mode replay requires --manifest")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.history_windows < 1 or args.max_answer_tokens < 1:
        parser.error("History-window count and answer budget must be positive.")
    if args.temperature < 0 or args.top_k < 0:
        parser.error("Temperature and top-k must be nonnegative.")
    return args


def main():
    global torch
    args = parse_args()
    import torch
    import tiktoken
    import config
    from miniGPT_model_RNN import MiniGPTRecall
    from checkpoint import load_model

    width = config.block_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    enc = tiktoken.get_encoding("gpt2")
    checkpoint = args.checkpoint or Path(config.MEMORY_CHECKPOINT_PATH)
    model = MiniGPTRecall(vocab_size=enc.n_vocab).to(device)
    model = load_model(checkpoint, model, device=device)
    model.to(device).eval()

    def forward(tokens, memory, write):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _, state = model(x, memory=memory, update_memory=write)
        return logits[:, -1, :], state

    required = args.history_windows * width
    print(f"\nHistory: {args.history_windows} windows x {width} = {required} tokens.")
    print("Memory writes occur once per full window. W4 never writes to memory.")
    if args.history_windows != 4:
        print("Note: this history length differs from the four-window training layout.")

    with torch.inference_mode():
        if args.mode == "replay":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            if manifest["block_size"] != width:
                raise ValueError("Manifest block_size differs from config/model; select matching config.")
            samples = manifest["samples"]
            indices = list(range(len(samples))) if args.sample_index is None else [args.sample_index]
            if any(i < 0 or i >= len(samples) for i in indices):
                raise ValueError("Sample index outside manifest.")
            if args.limit is not None:
                indices = indices[:args.limit]
            records = []
            for i in indices:
                sample = samples[i]
                tokens = sample["tokens"]
                count = manifest["sequence_windows"] - 1
                question_start = count * width
                memory = ingest_history(forward, tokens[:question_start], width, count)
                # start is a Y-coordinate. This input predicts the first answer;
                # no gold answer token is supplied to the decoder.
                prefix = tokens[question_start:question_start + sample["start"] + 1]
                expected = enc.decode(sample["answer_tokens"])
                generated, answer_tokens, _ = answer_question(
                    forward, enc, prefix, memory, min(args.max_answer_tokens, width-len(prefix)))
                row = dict(index=i, expected=expected, generated=generated,
                    full_answer_match=generated.strip() == expected.strip(),
                    answer_token_prefix_match=answer_tokens[:len(sample["answer_tokens"])] == sample["answer_tokens"],
                    generated_token_ids=answer_tokens)
                records.append(row)
                print(f"Sample {i} | expected={expected!r} | generated={generated!r} | exact={row['full_answer_match']}")
            if not records:
                raise ValueError("Empty replay selection.")
            rate = sum(r["full_answer_match"] for r in records) / len(records)
            print(f"Greedy full-answer accuracy: {rate:.2%} | n={len(records)}")
            if args.save_json:
                args.save_json.parent.mkdir(parents=True, exist_ok=True)
                args.save_json.write_text(json.dumps(dict(mode="replay", checkpoint=str(checkpoint),
                    manifest=str(args.manifest), full_answer_accuracy=rate, samples=records), indent=2) + "\n", encoding="utf-8")
            return

        if args.mode == "generate":
            prompt = args.prompt if args.prompt is not None else input("Story seed: ")
            seed_tokens = enc.encode_ordinary(prompt)
            print(f"History length includes the {len(seed_tokens)} seed tokens.", flush=True)
            def show_window(index, tokens):
                print(f"\n--- W{index} ---\n{enc.decode(tokens)}", flush=True)
            history, memory = generate_history(
                forward,
                lambda logits: sample_next(logits, args.temperature, args.top_k, generator),
                seed_tokens, width, args.history_windows, show_window)
        else:
            text = (args.history_file.read_text(encoding="utf-8")
                    if args.history_file else read_multilines("Paste the history:"))
            history = enc.encode_ordinary(text)
            if len(history) < required:
                raise ValueError(f"History needs {required} tokens; received {len(history)}.")
            if len(history) > required:
                print(f"Using the first {required} of {len(history)} pasted tokens.")
            history = history[:required]
            memory = ingest_history(forward, history, width, args.history_windows)
            print("\n--- HISTORY USED ---\n" + enc.decode(history))

        # One decode of the full token stream avoids display artifacts when
        # UTF-8 characters straddle window boundaries.
        if args.mode == "generate":
            print("\n--- COMPLETE PROSE ---\n" + enc.decode(history))
        print(f"\nSaved memory shape: {tuple(memory.shape)}")
        question = args.question if args.question is not None else input("\nQuestion: ")
        prefix, offset, prompt_text = build_question_prefix(
            enc, question, width, args.max_answer_tokens,
            random.Random(args.seed + 1), args.question_offset, args.raw_question_prompt)
        print("\n--- QUESTION DECODER INPUT ---")
        print("Total tokens:", len(prefix))
        print("Filler tokens:", offset)
        print("Question/wrapper tokens:", len(prefix) - offset)
        print("Block size:", width)
        print("Full input:", repr(enc.decode(prefix)))
        print("Without filler:", repr(enc.decode(prefix[offset:])))
        print("--- END INPUT ---\n")
        print(
                f"W{args.history_windows}: {len(prefix)} total input tokens "
                f"({offset} filler + {len(prefix) - offset} question/wrapper); "
                f"up to {args.max_answer_tokens} answer tokens."
            )

        conditions = [("Normal", memory, True)]
        if args.compare_memory:
            conditions += [("NoRead", memory, False),
                           ("ZeroState", torch.zeros_like(memory), True)]
        results = {}
        expected_ids = (enc.encode_ordinary(" " + args.expected_answer.strip())
                        if args.expected_answer and args.expected_answer.strip() else [])
        for label, state, enabled in conditions:
            with memory_reads(model, enabled):
                text, ids, first = answer_question(
                    forward, enc, prefix, state, args.max_answer_tokens)
            results[label] = dict(text=text, token_ids=ids)
            print(f"{label}: {text}")
            if expected_ids:
                target = expected_ids[0]
                rank = int((first[0] > first[0, target]).sum().item()) + 1
                probability = float(torch.softmax(first.float(), dim=-1)[0, target].item())
                results[label]["expected_first_token"] = dict(id=target, rank=rank, probability=probability)
                print(f"  Expected first token {enc.decode([target])!r}: rank={rank}, P={probability:.6g}")

    if args.save_json:
        payload = dict(checkpoint=str(checkpoint), block_size=width,
            history_windows=args.history_windows, history_token_ids=history,
            history_text=enc.decode(history), question_prompt=prompt_text,
            question_prefix_token_ids=prefix, filler_tokens=offset,
            max_answer_tokens=args.max_answer_tokens, seed=args.seed,
            temperature=args.temperature, top_k=args.top_k, mode=args.mode, answers=results)
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved {args.save_json}")


if __name__ == "__main__":
    main()
