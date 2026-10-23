from pathlib import Path
from config import (MEMORY_WINDOWS_PER_SEQUENCE, 
                    MEMORY_MAX_ASSOCIATIONS, 
                    MEMORY_MIN_ASSOCIATIONS, 
                    FACT_OFFSET_SCHEDULE,
                    ALL_FACTS_SINGLE_QA,
                    FACT_OFFSET_SCHEDULE,
                    USE_FACT_OFFSET_CURRICULUM,) 
import config as runtime_config
import random
import torch
import json

from tokenizer import encode

# False preserves build_memory_windows' randomized fact/QA placements.
# Set True in config.py to enable the token-based single-fact curriculum.
USE_FACT_OFFSET_CURRICULUM = getattr(
    runtime_config, "USE_FACT_OFFSET_CURRICULUM", False)

def fact_offset_range(tokens_seen, schedule=None):
    """Return inclusive (min_start, max_start) for the active token stage."""
    raw_schedule = FACT_OFFSET_SCHEDULE if schedule is None else schedule
    if raw_schedule is None:
        raise ValueError("Fact offset schedule cannot be empty.")
    if not isinstance(raw_schedule, (list, tuple)):
        raise ValueError("Fact offset schedule cannot be empty.")
    schedule_items = list(raw_schedule)
    if not schedule_items:
        raise ValueError("Fact offset schedule cannot be empty.")

    if not isinstance(tokens_seen, int) or tokens_seen < 0:
        raise ValueError("tokens_seen must be a nonnegative integer.")

    previous = -1
    active: tuple[int, int | None] | None = None
    for entry in schedule_items:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError("Each schedule item must be a (threshold, low, high) tuple.")
        threshold, low, high = entry
        if not isinstance(threshold, int) or threshold < 0 or threshold <= previous:
            raise ValueError("Schedule token thresholds must increase strictly.")
        if not isinstance(low, int) or low < 0:
            raise ValueError("Minimum fact offset must be a nonnegative integer.")
        if high is not None and (not isinstance(high, int) or high < low):
            raise ValueError("Maximum fact offset must be >= minimum, or None.")
        if previous == -1 and threshold != 0:
            raise ValueError("Fact offset schedule must start at token 0.")
        if tokens_seen >= threshold:
            active = (low, high)
        previous = threshold
    if active is None:
        raise ValueError("Fact offset schedule has no active stage for tokens_seen.")
    return active


def _randomize_w0_facts(sample, rng):
    """Reposition all W0 facts without overlap; preserve filler order and W1..W4."""
    placements = [dict(item) for item in sample["placements"]]
    facts = sorted((p for p in placements if p["window"] == 0),
                   key=lambda p: p["local_start"])
    if len(facts) != sample["association_count"] or any(p["kind"] != "fact" for p in facts):
        raise ValueError("Expected all associations to have exactly one fact in W0.")
    x, y = sample["x"], sample["y"]
    pieces, chunks, cursor = [], [], 0
    for p in facts:
        start, end = p["local_start"], p["local_end"]
        if not cursor <= start < end <= x.shape[1]:
            raise ValueError("Invalid or overlapping W0 fact metadata.")
        pieces.append(x[0, cursor:start])
        chunks.append((p, x[0, start:end]))
        cursor = end
    pieces.append(x[0, cursor:])
    filler = torch.cat(pieces)
    rng.shuffle(chunks)
    cuts = sorted(rng.randint(0, len(filler)) for _ in chunks)
    boundaries = [0] + cuts + [len(filler)]
    assembled, local, filler_pos = [], 0, 0
    for i, (p, tokens) in enumerate(chunks):
        gap = boundaries[i + 1] - boundaries[i]
        assembled.extend((filler[filler_pos:filler_pos + gap], tokens))
        filler_pos += gap
        local += gap
        p["local_start"], p["local_end"] = local, local + len(tokens)
        local += len(tokens)
    assembled.append(filler[filler_pos:])
    w0 = torch.cat(assembled)
    if len(w0) != x.shape[1]:
        raise RuntimeError("W0 size changed during fact relocation.")
    result = dict(sample)
    result["x"], result["y"] = x.clone(), y.clone()
    result["x"][0] = w0
    result["y"][0] = torch.cat((w0[1:], x[1, :1]))
    result["placements"] = placements
    result["fact_offset_range"] = (0, None)
    return result


def apply_fact_offset(sample, tokens_seen, schedule=None, rng=None):
    """Move one complete target fact within W0 without changing W1..W4.

    Return a copy; rebuild W0 next-token targets and update placement metadata.
    This controlled experiment requires exactly one association per sequence.
    Call immediately before batching so prebuilt RAM pools respect stage changes.
    """
    rng = random if rng is None else rng
    low, high = fact_offset_range(tokens_seen, schedule=((0,0,None),))
    if sample.get("all_facts_w0_single_qa"):
        if (low, high) != (0, None):
            raise ValueError("All-facts-W0 mode requires FACT_OFFSET_SCHEDULE = ((0, 0, None),).")
        return _randomize_w0_facts(sample, rng)
    if sample.get("association_count") != 1:
        raise ValueError("Offset curriculum requires one association per sequence.")
    placements = [dict(item) for item in sample["placements"]]
    w0_chunks = [item for item in placements if item["window"] == 0]
    if len(w0_chunks) != 1 or w0_chunks[0]["kind"] != "fact" or not w0_chunks[0]["is_target"]:
        raise ValueError("Expected exactly one target fact chunk in W0.")
    fact = w0_chunks[0]
    start, end = fact["local_start"], fact["local_end"]
    x, y = sample["x"], sample["y"]
    if x.shape[0] < 2 or x.shape != y.shape:
        raise ValueError("Expected matching X/Y tensors with at least two windows.")
    width = x.shape[1]
    if not 0 <= start < end <= width:
        raise ValueError("Invalid fact placement metadata.")
    fact_tokens = x[0, start:end]
    filler = torch.cat((x[0, :start], x[0, end:]))
    max_start = len(filler)
    effective_high = max_start if high is None else min(high, max_start)
    if low > effective_high:
        raise ValueError("Fact offset range cannot fit this complete fact in W0.")
    offset = rng.randint(low, effective_high)
    new_w0 = torch.cat((filler[:offset], fact_tokens, filler[offset:]))
    result = dict(sample)
    result["x"] = x.clone()
    result["y"] = y.clone()
    result["x"][0] = new_w0
    # W0's final target remains W1's first input token.
    result["y"][0] = torch.cat((new_w0[1:], x[1, :1]))
    fact["local_start"] = offset
    fact["local_end"] = offset + len(fact_tokens)
    result["placements"] = placements
    result["fact_offset_range"] = (low, effective_high)
    return result


# Find Dataset files

def find_data_files(data_dir):

    files = sorted(Path(data_dir).rglob("*.txt"))

    if not files:
        raise RuntimeError(f"No .txt files found in {data_dir}")

    return files


# Create a permanent training and validation dataset split

def split_set(files, train_ratio=0.9, seed=42):
    """Group Gutenberg editions by book, honoring a new QA build manifest.

    Generic non-Gutenberg corpora retain the legacy seeded file split.
    This improves NEW-run split hygiene; it changes the old file-level split.
    """
    from dataset_utils import canonical_book_id, book_split
    files = sorted(files)
    try:
        books = {str(p): canonical_book_id(Path(p).name) for p in files}
    except ValueError:
        rng = random.Random(seed)
        rng.shuffle(files)
        split = int(train_ratio * len(files))
        return files[:split], files[split:]
    manifest_path = Path(getattr(runtime_config, 'GUTENBERG_QA_DIR', './gutenberg_qa')) / 'build_manifest.json'
    destinations = {}
    if manifest_path.is_file():
        metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
        destinations = {str(item['book_id']): item['split'] for item in metadata.get('books', [])}
    train, valid = [], []
    for path in files:
        book = books[str(path)]
        destination = destinations.get(book, book_split(book, seed, 1-train_ratio))
        (valid if destination == 'validation' else train).append(path)
    return train, valid


# Divide total memory of files into RAM-sized pools

def make_ram_pool(files, pool_bytes_limit):

    # Group files into <= pool_bytes size
    ## If file doesn't fit it is stored as remainder ("long division")

    pool = []
    pool_bytes = 0

    for path in files:
        file_size = path.stat().st_size

        # Incase current file exceeds pool budget

        if (pool and pool_bytes + file_size > pool_bytes_limit):
            yield pool
            pool = []
            pool_bytes = 0

        # Incase single file is larger than pool budget itself
        if file_size > pool_bytes_limit:
            # Flush pool buffer
            if pool:
                yield pool
                pool = []
                pool_bytes = 0

            yield [path]
            continue

        pool.append(path)
        pool_bytes += file_size

    # Remainder
    if pool:
        yield pool

# Load and tokenize files in RAM pool

def tokenize_set(pool_paths, block_size):

    # Load files from disk into RAM and tokenize them
    ## Document boundaries are preserved to training never crosses file boundaries
    ### Tokens are in int32 format for tensor compatibility

    document = []
    for path in pool_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        token_ids = encode(text)

        # To make X and Y tensor we need block_size and block_size + 1 tokens
        if len(token_ids) > block_size:
            tokens = torch.tensor(token_ids, dtype=torch.int32)
            document.append(tokens)

        del text
        del token_ids
        
    return document

def load_memory_set(path):

    # This function will load QA records from a jsonl file as a list called memories

    memories = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                memories.append(json.loads(line))

    if not memories:
        raise ValueError(f"Memory set is empty: {path}")

    return memories

def randomize_record(item):

    # This function will randomize the records contained in the memories list

    subject = random.choice(
        item["subject_pool"]
    )

    answer = random.choice(
        item["answer_pool"]
    )

    fact = item["fact_template"].format(
        subject=subject,
        answer=answer,
    )

    question = item["question_template"].format(
        subject=subject,
    )

    return {
        "id": item["id"],
        "relation": item["relation"],
        "subject": subject,
        "fact": fact,
        "question": question,
        "answer": answer,
    }

def count_tokens(documents):

    # Number of tokens available in RAM pool

    return sum(len(doc) for doc in documents)


# Controlled randomization for the persistent-memory task.
#
# We deliberately keep one anchor association whose FACT is in W0 and whose
# recall QA is in the final window.  This guarantees that every sample still
# contains a known long-range dependency.  The total number of associations,
# their local positions, and all distractor fact/question windows are random.
#
# Increase MEMORY_MAX_ASSOCIATIONS later if you want to put more associations
# into the stream than there are memory slots.
MEMORY_MIN_ASSOCIATIONS = MEMORY_MIN_ASSOCIATIONS
MEMORY_MAX_ASSOCIATIONS = MEMORY_MAX_ASSOCIATIONS


def _encode_memory_record(item, dtype):
    """Tokenize one randomized association into a fact and a QA chunk."""

    fact_tokens = torch.tensor(
        encode("\n" + item["fact"] + "\n"),
        dtype=dtype,
    )

    question_prefix_tokens = torch.tensor(
        encode(
            "\n\nQuestion: "
            + item["question"]
            + "\nAnswer:"
        ),
        dtype=dtype,
    )

    answer_tokens = torch.tensor(
        encode(" " + item["answer"]),
        dtype=dtype,
    )

    qa_end_tokens = torch.tensor(
        encode("\n"),
        dtype=dtype,
    )

    qa_tokens = torch.cat((
        question_prefix_tokens,
        answer_tokens,
        qa_end_tokens,
    ))

    return {
        "fact_tokens": fact_tokens,
        "question_prefix_tokens": question_prefix_tokens,
        "answer_tokens": answer_tokens,
        "qa_end_tokens": qa_end_tokens,
        "qa_tokens": qa_tokens,
    }


def _draw_unique_memory_records(memories, count, distinct_subjects_answers=False):
    """
    Draw randomized associations while avoiding duplicate
    (relation, subject) keys inside one recurrent sample.

    Duplicate keys with different answers would make the recall target
    genuinely ambiguous, which is not useful for this experiment.
    """

    records = []
    used_keys = set()
    used_subjects = set()
    used_answers = set()

    max_attempts = max(100, count * 50)
    attempts = 0

    while len(records) < count and attempts < max_attempts:
        attempts += 1

        template = random.choice(memories)
        item = randomize_record(template)

        key = (
            item.get("relation"),
            item.get("subject"),
        )

        if key in used_keys:
            continue
        if distinct_subjects_answers and (item["subject"] in used_subjects or item["answer"] in used_answers):
            continue

        used_keys.add(key)
        used_subjects.add(item["subject"])
        used_answers.add(item["answer"])
        records.append(item)

    if len(records) != count:
        raise RuntimeError(
            "Could not draw enough unique memory associations. "
            f"Requested {count}, got {len(records)}."
        )

    return records


def _random_gap_sizes(total_filler, gap_count):
    """
    Randomly divide filler tokens among gaps surrounding artificial chunks.

    For N artificial chunks there are N+1 filler gaps.  Random cut points
    make the chunks appear at different local positions instead of always
    at the beginning/end of a context window.
    """

    if gap_count <= 0:
        raise ValueError("gap_count must be positive")

    if gap_count == 1:
        return [total_filler]

    cuts = sorted(
        random.randint(0, total_filler)
        for _ in range(gap_count - 1)
    )

    boundaries = [0] + cuts + [total_filler]

    return [
        boundaries[i + 1] - boundaries[i]
        for i in range(gap_count)
    ]


def _assemble_memory_window(
    filler_tokens,
    chunks,
    block_size,
    window_index,
):
    """
    Assemble one fixed-size context window from Gutenberg filler plus a
    randomized ordering/placement of artificial chunks.

    Returns:
        window: [block_size]
        placements: metadata describing where each artificial chunk landed.
    """

    chunks = chunks.copy()
    random.shuffle(chunks)

    artificial_tokens = sum(
        len(chunk["tokens"])
        for chunk in chunks
    )

    filler_needed = block_size - artificial_tokens

    if filler_needed < 0:
        raise ValueError(
            f"Artificial memory text does not fit in W{window_index}: "
            f"{artificial_tokens} > {block_size} tokens"
        )

    if len(filler_tokens) != filler_needed:
        raise RuntimeError(
            f"Wrong filler length for W{window_index}: "
            f"expected {filler_needed}, got {len(filler_tokens)}"
        )

    gap_sizes = _random_gap_sizes(
        filler_needed,
        len(chunks) + 1,
    )

    pieces = []
    placements = []

    filler_pos = 0
    local_pos = 0

    for chunk_index, chunk in enumerate(chunks):
        gap_len = gap_sizes[chunk_index]

        if gap_len:
            pieces.append(
                filler_tokens[
                    filler_pos:
                    filler_pos + gap_len
                ]
            )
            filler_pos += gap_len
            local_pos += gap_len

        chunk_start = local_pos

        pieces.append(chunk["tokens"])
        local_pos += len(chunk["tokens"])

        placements.append({
            "association_index": chunk["association_index"],
            "kind": chunk["kind"],
            "is_target": chunk["is_target"],
            "window": window_index,
            "local_start": chunk_start,
            "local_end": local_pos,
            "answer_offset_start": chunk.get("answer_offset_start"),
            "answer_offset_end": chunk.get("answer_offset_end"),
        })

    final_gap = gap_sizes[-1]

    if final_gap:
        pieces.append(
            filler_tokens[
                filler_pos:
                filler_pos + final_gap
            ]
        )
        filler_pos += final_gap
        local_pos += final_gap

    if filler_pos != len(filler_tokens):
        raise RuntimeError(
            f"Did not consume all filler in W{window_index}"
        )

    if local_pos != block_size:
        raise RuntimeError(
            f"W{window_index} has {local_pos} tokens, "
            f"expected {block_size}"
        )

    if pieces:
        window = torch.cat(pieces)
    else:
        # This is only reachable if block_size is zero, but keeps the helper
        # structurally complete.
        window = filler_tokens

    return window, placements


def build_memory_windows(
    document,
    memories,
    block_size,
    min_associations=MEMORY_MIN_ASSOCIATIONS,
    max_associations=MEMORY_MAX_ASSOCIATIONS,
    *,
    tokens_seen=None,
    fact_offset_schedule=None,
    all_facts_w0_single_qa=ALL_FACTS_SINGLE_QA,
):
    """
    Build randomized recurrent W0 -> W4 memory samples.

    Optional curriculum: supply tokens_seen and fact_offset_schedule for
    one-association samples. Omit tokens_seen for unrestricted placement.

    Set all_facts_w0_single_qa=True to put every fact in W0 and inject
    only the randomly selected target QA in the final window.

    Controlled randomization (legacy mode):
      * Number of associations varies from min_associations..max_associations.
      * One randomly chosen TARGET fact is guaranteed to appear somewhere in W0.
      * Its recall QA is guaranteed to appear somewhere in the final window.
      * Every artificial chunk is randomly positioned inside its window.
      * Distractor facts can appear in any non-final window.
      * Each distractor QA appears in a later window than its own fact.
      * Distractor QAs are allowed in the final window, so the target is not
        uniquely identifiable just because it is the only question there.

    Why keep the target fact in W0 / target QA in W4 for now?
      This guarantees a known long-range dependency while removing fixed count
      and fixed local-position shortcuts.  Fully randomizing the target windows
      should be a later experiment because the current training/evaluation code
      expects the supervised recall answer span to live in W4.
    """

    if min_associations < 1:
        raise ValueError(
            "min_associations must be at least 1"
        )

    if max_associations < min_associations:
        raise ValueError(
            "max_associations must be >= min_associations"
        )

    if MEMORY_WINDOWS_PER_SEQUENCE < 2:
        raise ValueError(
            "Memory recall requires at least two context windows"
        )

    sequences = []

    sequence_size = (
        MEMORY_WINDOWS_PER_SEQUENCE
        * block_size
    )

    final_window_index = (
        MEMORY_WINDOWS_PER_SEQUENCE - 1
    )

    final_window_start = (
        final_window_index
        * block_size
    )

    # Tracks only real Gutenberg tokens consumed. Artificial memory text
    # reduces how many Gutenberg tokens are needed for a fixed-size sample.
    doc_pos = 0

    while True:

        association_count = random.randint(
            min_associations,
            max_associations,
        )

        items = _draw_unique_memory_records(
            memories,
            association_count,
            distinct_subjects_answers=all_facts_w0_single_qa,
        )

        # The target is not tied to association index 0.
        target_index = random.randrange(
            association_count
        )

        encoded_items = [
            _encode_memory_record(
                item,
                document.dtype,
            )
            for item in items
        ]

        # Each element is a list of artificial chunks assigned to that window.
        chunks_by_window = [
            []
            for _ in range(
                MEMORY_WINDOWS_PER_SEQUENCE
            )
        ]

        # -----------------------------------------------------------------
        # Assign artificial chunks to windows.
        # -----------------------------------------------------------------
        #
        # Target:
        #   fact -> W0
        #   QA   -> final window
        #
        # Distractors:
        #   fact -> random W0..W(N-2)
        #   QA   -> random later window
        #
        # Thus every distractor is itself a valid memory association, but only
        # the target answer receives the extra recall loss.
        for association_index, (
            item,
            encoded,
        ) in enumerate(
            zip(items, encoded_items)
        ):

            is_target = (
                association_index == target_index
            )

            if is_target or all_facts_w0_single_qa:
                fact_window = 0
                qa_window = final_window_index
            else:
                fact_window = random.randint(
                    0,
                    final_window_index - 1,
                )

                qa_window = random.randint(
                    fact_window + 1,
                    final_window_index,
                )

            fact_tokens = encoded["fact_tokens"]
            qa_tokens = encoded["qa_tokens"]

            if len(fact_tokens) >= block_size:
                raise ValueError(
                    "Fact is too large for one context window: "
                    f"{len(fact_tokens)} tokens"
                )

            if len(qa_tokens) >= block_size:
                raise ValueError(
                    "QA is too large for one context window: "
                    f"{len(qa_tokens)} tokens"
                )

            chunks_by_window[fact_window].append({
                "tokens": fact_tokens,
                "association_index": association_index,
                "kind": "fact",
                "is_target": is_target,
            })

            if all_facts_w0_single_qa and not is_target:
                continue

            qa_chunk = {
                "tokens": qa_tokens,
                "association_index": association_index,
                "kind": "qa",
                "is_target": is_target,
            }

            # Only the target QA needs exact answer offsets for the recall loss.
            if is_target:
                answer_offset_start = len(
                    encoded["question_prefix_tokens"]
                )

                answer_offset_end = (
                    answer_offset_start
                    + len(encoded["answer_tokens"])
                )

                qa_chunk[
                    "answer_offset_start"
                ] = answer_offset_start

                qa_chunk[
                    "answer_offset_end"
                ] = answer_offset_end

            chunks_by_window[qa_window].append(
                qa_chunk
            )

        artificial_per_window = [
            sum(
                len(chunk["tokens"])
                for chunk in window_chunks
            )
            for window_chunks in chunks_by_window
        ]

        if any(
            artificial_tokens > block_size
            for artificial_tokens
            in artificial_per_window
        ):
            if all_facts_w0_single_qa:
                raise ValueError("Combined W0 facts exceed block_size; shorten fact templates or increase block_size.")
            # This is unlikely with the short synthetic records, but randomly
            # clustered chunks can theoretically overflow one window.
            # Redraw the sample instead of biasing placement with a fallback.
            continue

        artificial_total = sum(
            artificial_per_window
        )

        input_filler_needed = (
            sequence_size
            - artificial_total
        )

        # One extra natural token is required after the final input token so
        # the final Y position has a next-token target.
        required_tokens = (
            input_filler_needed + 1
        )

        if (
            doc_pos + required_tokens
            > len(document)
        ):
            break

        filler = document[
            doc_pos:
            doc_pos + required_tokens
        ]

        input_filler = filler[:-1]
        extra_y_token = filler[-1:]

        window_tensors = []
        all_placements = []

        filler_pos = 0

        for window_index in range(
            MEMORY_WINDOWS_PER_SEQUENCE
        ):
            filler_count = (
                block_size
                - artificial_per_window[
                    window_index
                ]
            )

            window_filler = input_filler[
                filler_pos:
                filler_pos + filler_count
            ]

            filler_pos += filler_count

            window, placements = (
                _assemble_memory_window(
                    window_filler,
                    chunks_by_window[
                        window_index
                    ],
                    block_size,
                    window_index,
                )
            )

            window_tensors.append(window)
            all_placements.extend(
                placements
            )

        if filler_pos != len(input_filler):
            raise RuntimeError(
                "Did not consume all input filler tokens"
            )

        # Fixed input stream plus one extra natural next-token target.
        sequence = torch.cat(
            window_tensors
            + [extra_y_token]
        )

        assert (
            len(sequence)
            == sequence_size + 1
        )

        # Locate the randomly positioned target QA inside W4.
        target_qa_placement = None

        for placement in all_placements:
            if (
                placement["is_target"]
                and placement["kind"] == "qa"
            ):
                target_qa_placement = placement
                break

        if target_qa_placement is None:
            raise RuntimeError(
                "Target QA placement was not recorded"
            )

        target_qa_global_start = (
            final_window_start
            + target_qa_placement[
                "local_start"
            ]
        )

        answer_start_global = (
            target_qa_global_start
            + target_qa_placement[
                "answer_offset_start"
            ]
        )

        answer_end_global = (
            target_qa_global_start
            + target_qa_placement[
                "answer_offset_end"
            ]
        )

        # Build standard next-token X/Y windows.
        X = []
        Y = []

        for window_idx in range(
            MEMORY_WINDOWS_PER_SEQUENCE
        ):
            start = (
                window_idx * block_size
            )

            x = sequence[
                start:
                start + block_size
            ]

            y = sequence[
                start + 1:
                start + block_size + 1
            ]

            X.append(x)
            Y.append(y)

        X = torch.stack(X)
        Y = torch.stack(Y)

        # W4 Y[0] predicts global sequence position final_window_start + 1.
        # Therefore the global answer span maps into W4 target coordinates by
        # subtracting that origin.  This +1 is the normal X->Y next-token shift,
        # not the old prepended-memory-token offset.
        answer_start = (
            answer_start_global
            - (final_window_start + 1)
        )

        answer_end = (
            answer_end_global
            - (final_window_start + 1)
        )

        if not (
            0
            <= answer_start
            < answer_end
            <= block_size
        ):
            raise RuntimeError(
                "Computed answer span is outside W4 targets: "
                f"[{answer_start}, {answer_end})"
            )

        target_item = items[target_index]

        sequences.append({
            "x": X,
            "y": Y,

            # Preserve the existing evaluator/training interface:
            # "memory" still identifies the supervised target association.
            "memory": target_item,
            "answer_start": answer_start,
            "answer_end": answer_end,

            # Extra metadata is useful for debugging the randomized task and
            # does not affect existing training code.
            "memories": items,
            "association_count": association_count,
            "all_facts_w0_single_qa": all_facts_w0_single_qa,
            "target_index": target_index,
            "placements": all_placements,
        })

        doc_pos += required_tokens

    if tokens_seen is not None:
        sequences = [apply_fact_offset(sample, tokens_seen, fact_offset_schedule)
                     for sample in sequences]
    return sequences

# Get number of context windows from each doc and shuffle them
## Guarantees each document gets used almost entirely 

def build_context_window(documents, block_size, random_offset=True, shuffle=True, clip_windows=False):
    
    windows = []

    for doc_idx, doc in enumerate(documents):
        if len(doc) <= block_size:
            continue
        if random_offset:
            max_offset = min(
                block_size - 1, len(doc) - block_size - 1
            )

            offset = random.randint(0, max_offset)
        else:
            offset = 0

        num_windows = (len(doc) - offset - 1) // block_size

        for window_idx in range(num_windows):
            start = (
                offset + window_idx * block_size
            )

            windows.append((doc_idx, start))

    if shuffle:
        random.shuffle(windows)
    
    return windows

# Create fixed-size training batches
def get_batch(documents, windows, batch_start, batch_size, block_size, device):

    # Create a [B, T] input tensor and one [B, T] output tensor
    # Each step contains block_size * batch_size prediction tokens

    if not documents:
        raise ValueError("Cannot create batch from empty document pool")

    X = []
    Y = []

    batch_windows = windows[batch_start: batch_start + batch_size]

    for doc_idx, start in batch_windows:
        doc = documents[doc_idx]

        x = doc[start: start + block_size]
        y = doc[start + 1: start + block_size + 1]

        X.append(x)
        Y.append(y)

    # Tensor creation on CPU
    X = torch.stack(X)
    Y = torch.stack(Y)

    # Embedding needs int64 / torch.long so conversion for GPU batch
    X = X.long().to(device, non_blocking=True)
    Y = Y.long().to(device, non_blocking=True)

    return X, Y

# Reproducible pool construction and batching, shared by training modes.
def build_seeded_training_windows(documents, block_size, pool_seed):
    previous = random.getstate()
    try:
        random.seed(pool_seed)
        return build_context_window(documents, block_size, random_offset=True, shuffle=True)
    finally:
        random.setstate(previous)


def build_seeded_memory_sequences(documents, memories, block_size, pool_seed):
    """Build two W0 facts and one randomly selected final-window QA."""
    previous = random.getstate()
    try:
        random.seed(pool_seed)
        sequences = []
        for document in documents:
            sequences.extend(build_memory_windows(
                document, memories, block_size, min_associations=2, max_associations=2,
                all_facts_w0_single_qa=ALL_FACTS_SINGLE_QA))
        random.shuffle(sequences)
        return sequences
    finally:
        random.setstate(previous)


def materialize_window_slice(documents, windows, start_index, count, block_size):
    """Reuse the regular loader batch function for CPU carry buffers."""
    if count <= 0:
        return None, None
    if len(windows[start_index:start_index+count]) != count:
        raise ValueError("Requested more context windows than remain in the pool.")
    return get_batch(documents, windows, start_index, count, block_size, "cpu")


def carry_count(carry_x):
    return 0 if carry_x is None else carry_x.shape[0]


def append_to_carry(carry_x, carry_y, new_x, new_y):
    if new_x is None:
        return carry_x, carry_y
    if carry_x is None:
        return new_x, new_y
    return torch.cat((carry_x, new_x)), torch.cat((carry_y, new_y))


def get_memory_batch(sequences, batch_start, batch_size, device):
    """Stack one full recurrent batch, validating shapes and answer spans."""
    selected = sequences[batch_start:batch_start+batch_size]
    if not selected or len(selected) != batch_size:
        raise ValueError("Requested more memory sequences than remain in the pool.")
    width = selected[0]["x"].shape[1]
    for sample in selected:
        expected = (MEMORY_WINDOWS_PER_SEQUENCE, width)
        if tuple(sample["x"].shape) != expected or tuple(sample["y"].shape) != expected:
            raise ValueError(f"Expected recurrent X/Y shape {expected}.")
        if not 0 <= sample["answer_start"] < sample["answer_end"] <= width:
            raise ValueError("Invalid final-window answer span.")
    tensors = (
        torch.stack([sample["x"] for sample in selected]),
        torch.stack([sample["y"] for sample in selected]),
        torch.tensor([sample["answer_start"] for sample in selected], dtype=torch.long),
        torch.tensor([sample["answer_end"] for sample in selected], dtype=torch.long),
    )
    return tuple(item.long().to(device, non_blocking=True) for item in tensors)


def scheduled_memory_samples(sequences, tokens_seen, schedule=None, seed=42,
                             *, use_offset_curriculum=None):
    """Apply optional offsets, or keep randomized base placements unchanged.

    Disabled mode supports multiple associations in either placement mode.
    Returns a new list referencing the base samples; callers must not mutate
    their tensors or placement metadata. get_memory_batch stacks fresh tensors.
    """
    enabled = (USE_FACT_OFFSET_CURRICULUM if use_offset_curriculum is None
               else use_offset_curriculum)
    if not isinstance(enabled, bool):
        raise ValueError("USE_FACT_OFFSET_CURRICULUM must be True or False.")
    if not enabled:
        return list(sequences)
    rng = random.Random(seed)
    return [apply_fact_offset(sample, tokens_seen, schedule, rng) for sample in sequences]