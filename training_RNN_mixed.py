"""Mixed recurrent recall / independent prose training.

Add to config.py:
    MIXED_RECALL_RATIO = 0.5       # fraction of optimizer updates, 0..1
    MIXED_PROSE_BATCH_SIZE = 64   # recall and validation use existing batch_size
    MIXED_CHECKPOINT_DIR = './checkpoints_mixed'
    MIXED_INIT_CHECKPOINT = './previous/best.pt'  # weights only, first launch
    MIXED_ACTIVATION_CHECKPOINTING = True  # recompute recall activations in backward
    MIXED_VERIFY_DATA_HASH = True  # False skips full-content hashing for debug
    # For older source checkpoints without saved attention-head metadata:
    # GUTENBERG_INIT_HEAD_NUM = 4  # SOURCE head count, not the target count

Run: python training_RNN_mixed.py
Existing MEMORY_TEST / exclusive-training mode flags do not select this loop.
Use both decoder and memory unfrozen; exclusive flags must be False.
Existing dataset, writer, assistance, loss, LR-scale, pool, evaluation,
checkpoint, size-limit, epoch and token-stop macros remain in use.
POOL_SIZE is split between the two training streams (not a hard process RAM cap).
STOP_AFTER_POOL_CT counts completed nonempty pools across both streams per launch.
The run ends when the selected stream reaches num_epochs, or a stop limit.
Validation uses batch_size for BOTH objectives, so increasing prose training
batch size does not silently change either validation sample count.
Checkpoints have conventional model_state_dict for generators and weight loading;
full mixed resume requires this script, not the original single-stream trainer.
Book splits are aligned using numeric Gutenberg IDs or matching filename stems.
QA record IDs may use BOOK:offset; explicit book/source-file metadata is supported.
Selection loss for best.pt is the equal-weight mean of recall combined loss
and prose LM loss; latest.pt always contains the most recent saved state."""

from contextlib import contextmanager
import types
import gc
from pathlib import Path
import json
import random
import math
import torch
from torch.utils.checkpoint import checkpoint
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
    RECALL_LOSS_WEIGHT,
    WARMUP_TOKENS,
    MAX_TOKENS,
    MIN_LEARN_RATE,
    STOP_AFTER_POOL_CT,
    USE_FACT_OFFSET_CURRICULUM,
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
    build_context_window,
    build_seeded_training_windows,
    materialize_window_slice,
    carry_count,
    append_to_carry,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class MixedProgress:
    """Low-frequency progress output, flushed for terminals and redirected logs."""
    def __init__(self, label, interval=30.0):
        import time
        self.clock = time.monotonic
        self.label, self.interval = label, interval
        self.started = self.last = self.clock()
        print(f'{label}...', flush=True)

    def update(self, detail, force=False):
        now = self.clock()
        if force or now-self.last >= self.interval:
            print(f'  {self.label} | {detail} | elapsed {now-self.started:.0f}s', flush=True)
            self.last = now

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



def resize_grouped_parameter(source, target, groups=None):
    """Overlap-copy independently packed groups, including changed head counts."""
    from itertools import product
    groups = groups or {}
    if source.ndim != target.ndim:
        raise ValueError('Parameter ranks differ during width migration.')
    packing = [groups.get(i, (1, 1)) for i in range(source.ndim)]
    if any(source.shape[i] % old or target.shape[i] % new
           for i, (old, new) in enumerate(packing)):
        raise ValueError('Width does not divide into the declared attention heads/gates.')
    result = target.detach().cpu().clone()
    source = source.detach().cpu().to(dtype=result.dtype)
    for coordinates in product(*(range(min(old, new)) for old, new in packing)):
        src, dst = [], []
        for axis, group in enumerate(coordinates):
            old, new = packing[axis]
            a, b = source.shape[axis]//old, result.shape[axis]//new
            size = min(a, b)
            src.append(slice(group*a, group*a+size))
            dst.append(slice(group*b, group*b+size))
        result[tuple(dst)].copy_(source[tuple(src)])
    return result


def migrate_backbone_width(state, target, old_c, new_c, old_m, new_m, old_d, new_d,
                           old_heads, new_heads, pooling_heads, allow_memory_shrink=False):
    """Known MiniGPTRecall tensors only; initialize new coordinates from target.

    Widening is an initialization, not a function-preserving transformation.
    Decoder depth and vocabulary changes remain strict errors.
    """
    import re
    if new_c < old_c:
        raise ValueError('Backbone-width migration supports expansion only, not embedding shrink.')
    if (new_m < old_m or new_d < old_d) and not allow_memory_shrink:
        raise ValueError('Memory shrink requires GUTENBERG_ALLOW_MEMORY_SHRINK=True.')
    for c, h in ((old_c, old_heads), (new_c, new_heads)):
        if type(h) is not int or h < 1 or c % h or (c//h) % 2:
            raise ValueError('Source/target attention head layout is invalid for RoPE. '
                             'Set GUTENBERG_INIT_HEAD_NUM to the source checkpoint head count.')
    for key in ('token_embedding_table.weight', 'lm_head.weight'):
        if key not in state or key not in target or state[key].shape[0] != target[key].shape[0]:
            raise ValueError('Vocabulary changes are not supported by width migration.')
    if not torch.equal(state['token_embedding_table.weight'], state['lm_head.weight']):
        raise ValueError('Expected tied source embedding/output weights; values differ.')
    state = dict(state)
    legacy = 'memory_encoder.block.'
    if any(k.startswith(legacy) for k in state):
        if any(k.startswith('memory_encoder.blocks.') for k in state):
            raise ValueError('Mixed legacy/stacked encoder checkpoint.')
        state = {('memory_encoder.blocks.0.'+k[len(legacy):] if k.startswith(legacy) else k):v
                 for k,v in state.items()}
    names = ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')
    shared = {k for k in state if k.startswith('memory.gru.')}
    separate = {k for k in state if k.startswith('memory.grus.')}
    if shared:
        if separate or shared != {'memory.gru.'+name for name in names}:
            raise ValueError('Incomplete/mixed source GRU layout.')
        for slot in range(old_m):
            for name in names:
                state[f'memory.grus.{slot}.{name}'] = state['memory.gru.'+name].clone()
        for key in shared:
            del state[key]
    expected = {f'memory.grus.{slot}.{name}' for slot in range(old_m) for name in names}
    if {k for k in state if k.startswith('memory.grus.')} != expected:
        raise ValueError('Incomplete source per-slot GRU layout.')
    migrated = {k:v for k,v in state.items() if not k.startswith('memory.grus.')}
    for slot in range(new_m):
        for name in names:
            key = f'memory.grus.{slot}.{name}'
            source = state[f'memory.grus.{slot % old_m}.{name}']
            shape = (3*old_d, old_d) if name.startswith('weight') else (3*old_d,)
            if tuple(source.shape) != shape or key not in target:
                raise ValueError(f'Invalid GRU tensor {key}.')
            migrated[key] = resize_grouped_parameter(source, target[key], {0:(3,3)})
    ph = (pooling_heads, pooling_heads)
    layouts = {
        'slot_embeddings': ((old_m,old_d),(new_m,new_d),{}),
        'input_norm.weight': ((old_c,),(new_c,),{}),
        'input_norm.bias': ((old_c,),(new_c,),{}),
        'slot_norm.weight': ((old_d,),(new_d,),{}),
        'slot_norm.bias': ((old_d,),(new_d,),{}),
        'query.weight': ((old_d,old_d),(new_d,new_d),{}),
        'key.weight': ((old_d,old_c),(new_d,new_c),{}),
        'value.weight': ((old_d,old_c),(new_d,new_c),{}),
        'multihead_query.weight': ((old_d,old_d),(new_d,new_d),{0:ph}),
        'multihead_key.weight': ((old_d,old_c),(new_d,new_c),{0:ph}),
        'multihead_value.weight': ((old_d,old_c),(new_d,new_c),{0:ph}),
        'multihead_out.weight': ((old_d,old_d),(new_d,new_d),{1:ph}),
        'refine_gru.weight_ih': ((3*old_d,old_d),(3*new_d,new_d),{0:(3,3)}),
        'refine_gru.weight_hh': ((3*old_d,old_d),(3*new_d,new_d),{0:(3,3)}),
        'refine_gru.bias_ih': ((3*old_d,),(3*new_d,),{0:(3,3)}),
        'refine_gru.bias_hh': ((3*old_d,),(3*new_d,),{0:(3,3)}),
        'refine_delta.0.weight': ((old_d,),(new_d,),{}),
        'refine_delta.0.bias': ((old_d,),(new_d,),{}),
        'refine_delta.1.weight': ((2*old_d,old_d),(2*new_d,new_d),{}),
        'refine_delta.1.bias': ((2*old_d,),(2*new_d,),{}),
        'refine_delta.3.weight': ((old_d,2*old_d),(new_d,2*new_d),{}),
        'refine_delta.3.bias': ((old_d,),(new_d,),{}),
    }
    count = 0
    for key, value in list(migrated.items()):
        if key.startswith('memory.grus.') or key not in target:
            continue
        spec = None
        if key in ('token_embedding_table.weight','lm_head.weight'):
            spec = ((value.shape[0],old_c),(value.shape[0],new_c),{})
        elif key.startswith('memory_compressor.'):
            spec = layouts.get(key[len('memory_compressor.'):])
        else:
            match = re.fullmatch(r'(?:blocks\.\d+|memory_encoder\.blocks\.\d+)\.(.+)',key)
            suffix = match.group(1) if match else ''
            if suffix.startswith(('at_head.','attention.')):
                part = suffix.split('.',1)[1]
                if part in ('query.weight','key.weight','value.weight','proj.weight'):
                    # Q/K preserve the paired halves used by this model's RoPE.
                    groups = ({1:(old_heads,new_heads)} if part=='proj.weight' else
                              {0:(old_heads*2,new_heads*2)} if part in ('query.weight','key.weight') else
                              {0:(old_heads,new_heads)})
                    spec = ((old_c,old_c),(new_c,new_c),groups)
                elif part=='proj.bias':
                    spec = ((old_c,),(new_c,),{})
            elif suffix.startswith('memory_attention.') and suffix.endswith('.weight'):
                part = suffix.split('.')[1]
                if part in ('query','key','value','proj'):
                    old_in,new_in = (old_d,new_d) if part in ('key','value') else (old_c,new_c)
                    groups = {1:(old_heads,new_heads)} if part=='proj' else {0:(old_heads,new_heads)}
                    spec = ((old_c,old_in),(new_c,new_in),groups)
            elif suffix in ('ffwd.net.0.weight','ffwd.net.2.weight'):
                spec = (((4*old_c,old_c),(4*new_c,new_c),{}) if suffix=='ffwd.net.0.weight' else
                        ((old_c,4*old_c),(new_c,4*new_c),{}))
            elif suffix in ('ffwd.net.0.bias','ffwd.net.2.bias'):
                multiplier = 4 if suffix=='ffwd.net.0.bias' else 1
                spec = ((multiplier*old_c,),(multiplier*new_c,),{})
            elif re.fullmatch(r'(ln[123]|memory_ln)\.(weight|bias)',suffix):
                a,b = (old_d,new_d) if suffix.startswith('memory_ln.') else (old_c,new_c)
                spec = ((a,),(b,),{})
            elif key in ('ln_final.weight','ln_final.bias','memory_encoder.ln_final.weight','memory_encoder.ln_final.bias'):
                spec = ((old_c,),(new_c,),{})
        if spec is None:
            if value.shape != target[key].shape:
                raise ValueError(f'Unsupported width-dependent tensor: {key}.')
            continue
        old_shape,new_shape,groups = spec
        if tuple(value.shape) != old_shape or tuple(target[key].shape) != new_shape:
            raise ValueError(f'Unexpected source/target shape for {key}: '
                             f'{tuple(value.shape)} -> {tuple(target[key].shape)}.')
        migrated[key] = resize_grouped_parameter(value,target[key],groups)
        count += 1
    # Copy the widened embedding once; strict loading retains the model's tie.
    migrated['lm_head.weight'] = migrated['token_embedding_table.weight'].clone()
    print(f'Migrated backbone {old_c}D -> {new_c}D, heads {old_heads} -> {new_heads}, '
          f'memory {old_m}x{old_d} -> {new_m}x{new_d}; {count} interface tensors. '
          'Added coordinates retain target initialization; retraining is required.')
    return migrated


def mixed_model_config(model):
    return dict(emb_dims=model.token_embedding_table.weight.shape[1],
                head_num=model.blocks[0].at_head.head_num,
                num_blocks=len(model.blocks), num_encoder_blocks=len(model.memory_encoder.blocks),
                block_size=block_size)


def load_initial_weights(model, path):
    """Load weights with backbone/head, memory, and encoder-depth migration.

    Use a new MIXED_CHECKPOINT_DIR and MIXED_INIT_CHECKPOINT for
    architecture changes; ordinary optimizer resume requires the same layout.
    Width expansion is partial weight transfer, not exact function preservation.
    Block-size changes rebuild nonpersistent RoPE caches in the target model.
    """
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint containing model_state_dict.")

    state = dict(checkpoint["model_state_dict"])
    target = model.state_dict()
    # RoPE buffers are derived from target dimensions/context, not learned.
    cache_suffixes = ('.inv_freq', '.cos_cached', '.sin_cached')
    for name in list(state):
        if (name.startswith('rope.') or '.rope.' in name) and name.endswith(cache_suffixes) and name not in target:
            del state[name]
    old_c = state['token_embedding_table.weight'].shape[1]
    new_c = target['token_embedding_table.weight'].shape[1]
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
    sidecar = Path(path).with_name(Path(path).name + '.writer.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    source_config = checkpoint.get('model_config', metadata.get('model_config', {}))
    old_block = source_config.get('block_size', metadata.get('writer_config',
        checkpoint.get('mixed_stage', {}).get('writer_config', {})).get('block_size'))
    if old_block is not None and old_block != block_size:
        print(f'Context size {old_block} -> {block_size}: target model rebuilds RoPE caches; '
              'no learned position table to resize. QA data must use the new block_size.')
    new_heads = model.blocks[0].at_head.head_num
    old_heads = source_config.get('head_num', getattr(runtime_config, 'GUTENBERG_INIT_HEAD_NUM', new_heads))
    if old_c != new_c or old_heads != new_heads:
        if 'head_num' not in source_config:
            print(f'No saved source attention-head metadata: using {old_heads} heads. '
                  'Set GUTENBERG_INIT_HEAD_NUM to the source head count if different.')
        source_pooling = metadata.get('writer_config', checkpoint.get('mixed_stage', {}).get('writer_config', {})).get(
            'pooling_heads', getattr(runtime_config, 'GUTENBERG_INIT_POOLING_HEADS', model.memory_compressor.pooling_heads))
        if source_pooling != model.memory_compressor.pooling_heads:
            raise ValueError('Width migration requires an unchanged memory pooling head count.')
        state = migrate_backbone_width(state, target, old_c, new_c, old_count, new_count,
            old_width, new_width, old_heads, new_heads, source_pooling, allow_shrink)
        state = migrate_encoder_weights(state, target)
        state = migrate_routing_weights(state, target)
        state = migrate_multihead_weights(state, target)
        model.load_state_dict(state, strict=True)
        print(f'Loaded migrated weights from {path}; target RoPE caches rebuilt for '
              f'block_size={block_size}. Fresh optimizer/scaler/counters.')
        return
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



def verify_model_writer(model):
    expected = writer_config()
    actual = getattr(model, 'memory_write_chunk_size', block_size)
    if actual != expected['write_chunk_size']:
        raise ValueError(f'Model uses write chunk size {actual}, but config requests '
                         f'{expected["write_chunk_size"]}. Load the updated model source.')
    if actual != block_size and not callable(getattr(model, 'write_memory_features', None)):
        raise ValueError('Chunked writing requires the updated model write_memory_features method.')



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



def setup_training(initial_checkpoint=None):
    """Initialize the joint model, grouped optimizer and mixed-precision scaler."""
    model = MiniGPTRecall(vocab_size=vocab_size).to(device)
    verify_model_writer(model)
    if initial_checkpoint:
        load_initial_weights(model, initial_checkpoint)
    learning = build_grouped_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, learning, scaler



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


def memory_training_forward(model, xb, yb, starts, ends, checkpointing=True):
    """Training-only window checkpoints; losses stay inside recomputed regions.

    No memory detach: the final QA loss still trains every earlier write.
    Do not return vocabulary logits from checkpointed windows. Evaluation uses
    memory_forward unchanged so its logits and diagnostics remain available.
    """
    if not checkpointing:
        _, memory, lm_loss, recall_loss = memory_forward(model, xb, yb, starts, ends)
        return None, memory, lm_loss, recall_loss

    def history_step(tokens, targets, state):
        _, lm_loss, next_state = model(
            tokens, targets, memory=state, update_memory=True)
        return lm_loss, next_state

    def question_step(tokens, targets, state, answer_start, answer_end):
        logits, lm_loss, _ = model(
            tokens, targets, memory=state, update_memory=False)
        recall_loss = compute_recall_loss(logits, targets, answer_start, answer_end)
        return lm_loss, recall_loss

    memory = None
    language_sum = 0.0
    for index in range(MEMORY_WINDOWS_PER_SEQUENCE - 1):
        lm_loss, memory = checkpoint(
            history_step, xb[:, index], yb[:, index], memory,
            use_reentrant=False, preserve_rng_state=True)
        language_sum = language_sum + lm_loss
    lm_loss, recall_loss = checkpoint(
        question_step, xb[:, -1], yb[:, -1], memory, starts, ends,
        use_reentrant=False, preserve_rng_state=True)
    return None, memory, (language_sum + lm_loss) / MEMORY_WINDOWS_PER_SEQUENCE, recall_loss



def cosine_decay(tokens_this_run, warm_up, max_tokens, learn_rate, min_learn_rate):
    if tokens_this_run < warm_up:
        return learn_rate * tokens_this_run / warm_up
    if tokens_this_run >= max_tokens:
        return min_learn_rate
    progress = (tokens_this_run - warm_up) / (max_tokens - warm_up)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learn_rate + coeff * (learn_rate - min_learn_rate)



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



def validate_mixed_controls():
    """Validate the shared stop, evaluation and checkpoint controls."""
    if TOKEN_STOP_COUNT is not None and (type(TOKEN_STOP_COUNT) is not int or TOKEN_STOP_COUNT < 1):
        raise ValueError('TOKEN_STOP_COUNT must be positive or None.')
    if type(STOP_AFTER_POOL_CT) is not int or STOP_AFTER_POOL_CT < 0:
        raise ValueError('STOP_AFTER_POOL_CT must be nonnegative.')
    if VALIDATION_BATCHES is not None and (type(VALIDATION_BATCHES) is not int or VALIDATION_BATCHES < 1):
        raise ValueError('VALIDATION_BATCHES must be positive or None.')
    if EVAL_INTERVAL_TOKENS <= 0 or CHECKPOINT_INTERVAL_TOKENS <= 0:
        raise ValueError('Evaluation/checkpoint token intervals must be positive.')



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
    progress = MixedProgress('Building prose validation cache')
    limit = VALIDATION_BATCHES * batch_size if VALIDATION_BATCHES is not None else None
    xs, ys, retained, books = [], [], 0, 0
    for file_index, path in enumerate(paths, 1):
        progress.update(f'file {file_index:,}/{len(paths):,}: {Path(path).name}; {retained:,} windows cached')
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
            progress.update(f'{file_index:,}/{len(paths):,} files; {retained:,} windows cached')
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



@contextmanager
def mixed_prose_mode(model):
    """Bypass memory and freeze its parameters only for this forward/backward.

    Avoid the original mode toggle's per-update prints. grad=None also prevents
    AdamW momentum/weight-decay updates to inactive memory parameters.
    """
    frozen = [model.memory_encoder, model.memory_compressor, model.memory]
    for block in model.blocks:
        frozen.extend([block.memory_attention, block.memory_ln, block.ln2])
    parameters = {id(p): p for module in frozen for p in module.parameters()}
    flags = [(p, p.requires_grad) for p in parameters.values()]
    try:
        for p, _ in flags:
            p.requires_grad_(False)
            p.grad = None
        with disabled_memory_reader(model):
            yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)



def mixed_recall_update(step, ratio):
    """Evenly distribute recall updates without RNG; first update is recall.

    0.5 -> recall, prose; 0.25 -> recall, prose, prose, prose.
    Counts differ from the requested cumulative fraction by less than one.
    """
    from fractions import Fraction
    fraction = Fraction(str(ratio))
    n, d = fraction.numerator, fraction.denominator
    return ((step + 1) * n + d - 1) // d > (step * n + d - 1) // d



class MixedStream:
    """Reconstructible bounded pool with independent cursor and prose carry."""
    def __init__(self, kind, files, size, pool_bytes, state=None):
        self.kind, self.files, self.size = kind, list(files), size
        self.pool_bytes = pool_bytes
        self.epoch = self.pool = self.cursor = self.completed = 0
        self.carry_x = self.carry_y = None
        self.documents = self.windows = self.sequences = None
        self.loaded = False
        if state is not None:
            for key in ('epoch', 'pool', 'cursor', 'completed', 'carry_x', 'carry_y'):
                setattr(self, key, state[key])

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ('epoch', 'pool', 'cursor', 'completed', 'carry_x', 'carry_y')}

    def load_pool(self):
        from gutenberg_qa_loader import read_records, build_sequences
        while self.epoch < num_epochs:
            files = self.files.copy()
            random.Random(42 + self.epoch).shuffle(files)
            paths = next((p for i, p in enumerate(make_ram_pool(files, self.pool_bytes))
                          if i == self.pool), None)
            if paths is None:
                self.epoch += 1
                self.pool = self.cursor = 0
                continue
            seed = 42 + self.epoch * 1_000_003 + self.pool
            print(f'Loading {self.kind} pool {self.pool} | epoch {self.epoch+1} | '
                  f'{len(paths):,} files...', flush=True)
            if self.kind == 'recall':
                documents = read_records(paths, block_size)
                print(f'Building recall sequences from {len(documents):,} records...', flush=True)
                self.sequences = (build_sequences(documents, [], block_size, pool_seed=seed)
                                  if documents else [])
                del documents
                compact_memory_sequences(self.sequences)
                count = len(self.sequences)
                if not 0 <= self.cursor <= count:
                    raise ValueError('Recall cursor exceeds reconstructed pool.')
                for i in range(self.cursor):
                    self.sequences[i] = None
            else:
                self.documents = compact_standard_documents(tokenize_set(paths, block_size))
                self.windows = (build_seeded_training_windows(self.documents, block_size, seed)
                                if self.documents is not None and len(self.documents) else [])
                count = len(self.windows)
                if not 0 <= self.cursor <= count:
                    raise ValueError('Prose cursor exceeds reconstructed pool.')
            self.loaded = True
            print(f'{self.kind.title()} pool {self.pool} | epoch {self.epoch+1} | '
                  f'{count:,} items | cursor {self.cursor:,}', flush=True)
            return
        raise StopIteration

    def finish_pool(self):
        count = len(self.sequences) if self.kind == 'recall' else len(self.windows)
        if count:
            self.completed += 1
        self.documents = self.windows = self.sequences = None
        self.loaded = False
        self.pool += 1
        self.cursor = 0
        gc.collect()

    def next_batch(self):
        while True:
            if not self.loaded:
                self.load_pool()
            if self.kind == 'recall':
                if self.cursor + self.size > len(self.sequences):
                    self.finish_pool()
                    continue
                batch = get_gutenberg_memory_batch(
                    self.sequences, self.cursor, self.size, device)
                for i in range(self.cursor, self.cursor + self.size):
                    self.sequences[i] = None
                self.cursor += self.size
                # Release completed pools now, so pool-stop includes this batch.
                if self.cursor + self.size > len(self.sequences):
                    self.finish_pool()
                return batch
            have = carry_count(self.carry_x)
            take = min(self.size-have, len(self.windows)-self.cursor)
            if take:
                x, y = materialize_window_slice(self.documents, self.windows,
                                                self.cursor, take, block_size)
                self.carry_x, self.carry_y = append_to_carry(
                    self.carry_x, self.carry_y, x, y)
                self.cursor += take
            if carry_count(self.carry_x) == self.size:
                x, y = self.carry_x, self.carry_y
                self.carry_x = self.carry_y = None
                if self.cursor == len(self.windows):
                    self.finish_pool()
                return (x.to(device=device, dtype=torch.long),
                        y.to(device=device, dtype=torch.long))
            self.finish_pool()



def mixed_file_list_signature(splits):
    """Cheap ordered split/path identity; never open dataset contents."""
    import hashlib
    listing = [(label, [str(Path(path).resolve()) for path in paths])
               for label, paths in splits.items()]
    return hashlib.sha256(json.dumps(listing, ensure_ascii=True).encode()).hexdigest()


def verify_mixed_stage(saved, current):
    """Hash verification may toggle without resetting optimizer or cursors."""
    ignored = {'data_signature', 'data_files_signature'}
    if (not isinstance(saved, dict) or
            {k:v for k,v in saved.items() if k not in ignored} !=
            {k:v for k,v in current.items() if k not in ignored}):
        raise ValueError('Mixed stage settings changed; use a new MIXED_CHECKPOINT_DIR '
                         'with MIXED_INIT_CHECKPOINT for weights-only initialization.')
    old_files, new_files = saved.get('data_files_signature'), current.get('data_files_signature')
    if old_files is not None and old_files != new_files:
        raise ValueError('Dataset file list/order changed; cannot resume saved data cursors.')
    old_hash, new_hash = saved.get('data_signature'), current.get('data_signature')
    if old_hash is not None and new_hash is not None and old_hash != new_hash:
        raise ValueError('Dataset contents changed; cannot resume saved data cursors.')
    if old_files is None and new_hash is None:
        print('Legacy checkpoint has no saved file-list signature; content checking is disabled. '
              'Recording the current file list; previous dataset identity is unverified.', flush=True)
    elif old_hash is None and new_hash is not None:
        print('Full hashing enabled: establishing a content baseline; '
              'the preceding checkpoint did not verify file contents.', flush=True)


def mixed_data_signature(splits, verify_contents=True):
    """Pin exact file bytes/order for safe pool reconstruction on resume."""
    if not verify_contents:
        print('Full dataset hashing disabled (MIXED_VERIFY_DATA_HASH=False); '
              'checking ordered file paths and training settings only.', flush=True)
        return None
    import hashlib
    progress = MixedProgress('Hashing dataset files for resume verification')
    total_files = sum(len(paths) for paths in splits.values())
    completed = total_bytes = 0
    digest = hashlib.sha256()
    for label, paths in splits.items():
        digest.update(label.encode())
        for path in paths:
            digest.update(str(Path(path).resolve()).encode())
            with Path(path).open('rb') as source:
                for chunk in iter(lambda: source.read(1024*1024), b''):
                    digest.update(chunk)
                    total_bytes += len(chunk)
                    progress.update(f'{label}: {completed:,}/{total_files:,} files complete; '
                                    f'{total_bytes/1024**3:.2f} GiB read; current {Path(path).name}')
            completed += 1
    progress.update(f'complete: {completed:,} files; {total_bytes/1024**3:.2f} GiB read', force=True)
    return digest.hexdigest()



def mixed_align_book_splits(splits):
    """Align matching books without requiring numeric-only record IDs."""
    import re
    def book_key(value):
        # Normalize Windows paths even when this trainer runs on Linux.
        name = str(value).strip().replace('\\', '/').rsplit('/', 1)[-1]
        if not name:
            raise ValueError('Empty book identifier in QA metadata.')
        stem = Path(name).stem if Path(name).suffix.lower() in ('.txt', '.text') else name
        # Gutenberg exports use ROWINDEX_BOOKID-ENCODING.txt, e.g.
        # 060495_20462-8.txt. The row index is not the Gutenberg book ID.
        exported = re.fullmatch(
            r'\d+_(?:pg)?(\d+)(?:[-_](?:images|noimages|0|8))?', stem, re.IGNORECASE)
        if exported:
            return str(int(exported.group(1)))
        match = re.fullmatch(r'(?:pg)?(\d+)(?:[-_](?:images|noimages|0|8))?', stem, re.IGNORECASE)
        return str(int(match.group(1))) if match else stem.casefold()

    def prose_id(path):
        return book_key(Path(path).name)

    def record_book(record):
        for key in ('book_id', 'gutenberg_id', 'source_file', 'source_path', 'book_path'):
            value = record.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return book_key(value)
        value = record.get('id')
        if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
            raise ValueError('QA record needs a book identifier or nonempty id.')
        # Strip the final numeric passage offset, preserving filenames and paths.
        book = re.sub(r':\d+$', '', str(value).strip())
        return book_key(book)
    qa_ids = {}
    progress = MixedProgress('Scanning QA book IDs to align train/validation splits')
    for split in ('train', 'valid'):
        ids = set()
        records_scanned = 0
        paths = splits['recall_'+split]
        print(f'  QA {split}: scanning {len(paths):,} files...', flush=True)
        for file_index, path in enumerate(paths, 1):
            progress.update(f'{split}: file {file_index:,}/{len(paths):,}; '
                            f'{records_scanned:,} records; current {Path(path).name}')
            with Path(path).open(encoding='utf-8') as source:
                for line in source:
                    record = json.loads(line)
                    ids.add(record_book(record))
                    records_scanned += 1
                    if records_scanned % 1024 == 0:
                        progress.update(f'{split}: file {file_index:,}/{len(paths):,}; '
                                        f'{records_scanned:,} records; {len(ids):,} book IDs')
        qa_ids[split] = ids
        progress.update(f'{split} complete: {records_scanned:,} records; {len(ids):,} book IDs', force=True)
    if qa_ids['train'] & qa_ids['valid']:
        raise ValueError('QA train and validation splits contain the same Gutenberg book.')
    # Reassign matching books rather than relying on two independent shuffles.
    all_prose = list(dict.fromkeys(splits['prose_train']+splits['prose_valid']))
    matched = {prose_id(path) for path in all_prose} & (qa_ids['train'] | qa_ids['valid'])
    if not matched:
        raise ValueError('No QA book identifiers match prose filenames. '
                         f'QA examples: {sorted(qa_ids["train"] | qa_ids["valid"])[:3]!r}; '
                         f'prose examples: {[prose_id(p) for p in all_prose[:3]]!r}. '
                         'Provide matching book_id/source_file metadata to align the splits.')
    import hashlib
    # Split otherwise-unmatched books by a stable hash of BOOK ID, not path.
    # All editions/encodings of a book therefore receive the same destination.
    destinations = {}
    splits['prose_train'], splits['prose_valid'] = [], []
    for path in all_prose:
        progress.update(f'Aligning prose files; current {Path(path).name}')
        book = prose_id(path)
        if book not in destinations:
            bucket = int.from_bytes(hashlib.sha256(('42:'+book).encode()).digest()[:8], 'big') % 10
            destinations[book] = ('valid' if book in qa_ids['valid'] else
                                  'train' if book in qa_ids['train'] else
                                  'valid' if bucket == 0 else 'train')
        destination = destinations[book]
        splits['prose_'+destination].append(path)
    # Multiple files for one otherwise-unmatched book must not cross the split.
    if ({prose_id(p) for p in splits['prose_train']} &
            {prose_id(p) for p in splits['prose_valid']}):
        raise ValueError('Duplicate prose book IDs cross the train/validation split.')
    print(f'Book splits aligned for {len(matched):,} matching recall/prose book IDs; '
          f'{len(destinations):,} unique prose books grouped across '
          f'{len(all_prose):,} files. Unmatched books use a deterministic 90/10 split.', flush=True)



def mixed_training():
    from gutenberg_qa_loader import read_records, build_sequences
    from contextlib import nullcontext
    ratio = getattr(runtime_config, 'MIXED_RECALL_RATIO', 0.5)
    prose_batch = getattr(runtime_config, 'MIXED_PROSE_BATCH_SIZE', batch_size)
    activation_checkpointing = getattr(runtime_config, 'MIXED_ACTIVATION_CHECKPOINTING', True)
    verify_data_hash = getattr(runtime_config, 'MIXED_VERIFY_DATA_HASH', True)
    if type(verify_data_hash) is not bool:
        raise ValueError('MIXED_VERIFY_DATA_HASH must be True or False.')
    if type(activation_checkpointing) is not bool:
        raise ValueError('MIXED_ACTIVATION_CHECKPOINTING must be True or False.')
    print(f'Recall activation checkpointing: {activation_checkpointing} | '
          'memory remains connected across windows.', flush=True)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('MIXED_RECALL_RATIO must be a finite number in 0..1.')
    for name, value in [('batch_size', batch_size), ('MIXED_PROSE_BATCH_SIZE', prose_batch),
                        ('POOL_SIZE', POOL_SIZE), ('num_epochs', num_epochs)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    # Reuse validation of common stop/evaluation macros, without selecting a mode.
    validate_mixed_controls()
    for flag in ('GUTENBERG_DECODER_ONLY', 'GUTENBERG_MEMORY_ONLY',
                 'STANDARD_DECODER_ONLY', 'STANDARD_MEMORY_ONLY'):
        if getattr(runtime_config, flag, False):
            raise ValueError(f'Set {flag}=False: mixed training controls memory per batch.')
    if USE_FACT_OFFSET_CURRICULUM:
        raise ValueError('Natural evidence cannot be relocated; disable USE_FACT_OFFSET_CURRICULUM.')
    directory = Path(getattr(runtime_config, 'MIXED_CHECKPOINT_DIR', './checkpoints_mixed'))
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / 'latest.pt'
    initial = getattr(runtime_config, 'MIXED_INIT_CHECKPOINT', None)
    splits = {}
    splits['recall_train'], splits['recall_valid'] = training_files(memory_mode=True)
    splits['prose_train'], splits['prose_valid'] = training_files(memory_mode=False)
    mixed_align_book_splits(splits)
    if any(not paths for paths in splits.values()):
        raise ValueError('Both datasets require nonempty training and validation splits.')
    stage = dict(schema=1, ratio=ratio, ratio_basis='optimizer_updates',
                 recall_batch=batch_size, prose_batch=prose_batch, writer_config=writer_config(),
                 pool_size=POOL_SIZE,
                 data_signature=mixed_data_signature(splits, verify_contents=verify_data_hash),
                 data_files_signature=mixed_file_list_signature(splits),
                 validation_batches=VALIDATION_BATCHES, recall_weight=RECALL_LOSS_WEIGHT,
                 evidence_windows=getattr(runtime_config, 'GUTENBERG_EVIDENCE_WINDOWS', None),
                 support_window=getattr(runtime_config, 'GUTENBERG_SUPPORT_WINDOW', None),
                 support_label=getattr(runtime_config, 'GUTENBERG_SUPPORT_LABEL', True),
                 memory_lr_scale=runtime_config.MEMORY_LR_SCALE,
                 decoder_lr_scale=runtime_config.DECODER_LR_SCALE,
                 warmup=WARMUP_TOKENS, max_tokens=MAX_TOKENS, peak_lr=learn_rate,
                 min_lr=MIN_LEARN_RATE, num_epochs=num_epochs)
    print('Initializing model, optimizer and scaler...', flush=True)
    if initial and not latest.exists():
        print(f'Loading initial weights from {initial}', flush=True)
    model, learning, scaler = setup_training(None if latest.exists() else initial)
    counters = dict(step=0, tokens=0, recall_tokens=0, prose_tokens=0,
                    recall_updates=0, prose_updates=0)
    best = float('inf')
    restored = None
    if latest.exists():
        print(f'Restoring mixed checkpoint: {latest}', flush=True)
        restored = torch.load(latest, map_location='cpu', weights_only=True)
        verify_mixed_stage(restored.get('mixed_stage'), stage)
        if restored.get('model_config') is not None and restored['model_config'] != mixed_model_config(model):
            raise ValueError('Model architecture changed: use a new MIXED_CHECKPOINT_DIR '
                             'and MIXED_INIT_CHECKPOINT for weights-only migration.')
        model.load_state_dict(restored['model_state_dict'], strict=True)
        learning.load_state_dict(restored['optimizer_state_dict'])
        scaler.load_state_dict(restored['scaler_state_dict'])
        counters = restored['mixed_counters']
        best = restored['best_val_loss']
    streams = {
        'recall': MixedStream('recall', splits['recall_train'], batch_size,
                              max(1, POOL_SIZE//2), restored['mixed_streams']['recall'] if restored else None),
        'prose': MixedStream('prose', splits['prose_train'], prose_batch,
                             max(1, POOL_SIZE//2), restored['mixed_streams']['prose'] if restored else None)}
    launch_tokens = counters['tokens']
    launch_pools = sum(s.completed for s in streams.values())
    print('Reading fixed recall validation records...', flush=True)
    records = read_records(splits['recall_valid'], block_size,
                           max_records=VALIDATION_BATCHES*batch_size if VALIDATION_BATCHES else None)
    print(f'Building fixed recall validation sequences from {len(records):,} records...', flush=True)
    valid = build_sequences(records, [], block_size, pool_seed=42)
    del records
    if VALIDATION_BATCHES is not None:
        del valid[VALIDATION_BATCHES*batch_size:]
    compact_memory_sequences(valid)
    print(f'Recall validation ready: {len(valid):,} sequences.', flush=True)
    prose_x, prose_y = build_standard_validation(splits['prose_valid'])
    train_valid = None
    if getattr(runtime_config, 'GUTENBERG_EVAL_TRAIN_SUBSET', False):
        print('Building fixed recall training diagnostic subset...', flush=True)
        diagnostic_batches = getattr(runtime_config, 'TRAIN_DIAGNOSTIC_BATCHES', 16)
        if type(diagnostic_batches) is not int or diagnostic_batches < 1:
            raise ValueError('TRAIN_DIAGNOSTIC_BATCHES must be positive.')
        records = read_records(splits['recall_train'], block_size,
                               max_records=diagnostic_batches*batch_size)
        train_valid = build_sequences(records, [], block_size, pool_seed=42)
        del records
        del train_valid[diagnostic_batches*batch_size:]
        compact_memory_sequences(train_valid)
    gc.collect()

    def save(path):
        payload = dict(model_state_dict=model.state_dict(),
                       model_config=mixed_model_config(model),
                       optimizer_state_dict=learning.state_dict(), scaler_state_dict=scaler.state_dict(),
                       tokens_seen=counters['tokens'], best_val_loss=best,
                       mixed_stage=stage, mixed_counters=dict(counters),
                       activation_checkpointing=activation_checkpointing,
                       mixed_streams={k: s.state_dict() for k,s in streams.items()},
                       rng_python=random.getstate(), rng_torch=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [])
        temp = path.with_name(path.name+'.tmp')
        torch.save(payload, temp)
        temp.replace(path)
        sidecar = path.with_name(path.name+'.writer.json')
        temp = sidecar.with_name(sidecar.name+'.tmp')
        temp.write_text(json.dumps(dict(metadata_schema_version=1, recall_includes_newline=True,
            model_config=mixed_model_config(model),
            writer_config=writer_config(), tokens_seen=counters['tokens'], decoder_only=False,
            memory_only=False, support_window=stage['support_window'],
            support_label=stage['support_label'], evidence_windows=stage['evidence_windows']), indent=2)+'\n')
        temp.replace(sidecar)
        print(f'Saved {path.name} at {counters["tokens"]:,} tokens', flush=True)

    interval = {'recall_lm': 0., 'recall_ce': 0., 'prose_lm': 0., 'recall_n': 0, 'prose_n': 0}
    def evaluate_mixed(event):
        print(f'Running {event} evaluation: recall...', flush=True)
        python_rng = random.getstate()
        try:
            with torch.random.fork_rng(devices=None if device=='cuda' else []):
                metrics = evaluate_memory(model, valid, batch_size, device, max_batches=VALIDATION_BATCHES)
                train_metrics = (evaluate_memory(model, train_valid, batch_size, device,
                    max_batches=diagnostic_batches) if train_valid is not None else None)
                print(f'Running {event} evaluation: prose...', flush=True)
                with mixed_prose_mode(model):
                    prose_loss = evaluate_standard_cached(model, prose_x, prose_y)
        finally:
            random.setstate(python_rng)
        # Equal task weighting for selection, independent of the update ratio.
        score = 0.5 * (metrics['total_loss'] + prose_loss)
        print(f'\nMixed {event} | Step {counters["step"]} | Tokens {counters["tokens"]:,} | '
              f'Tokens this run {counters["tokens"]-launch_tokens:,}')
        print(f'  Prose LM {prose_loss:.4f} | Recall LM {metrics["language_loss"]:.4f} | '
              f'Recall combined {metrics["total_loss"]:.4f} | Selection loss {score:.4f}')
        print_memory_diagnostics(metrics)
        if train_metrics is not None:
            print('  Fixed recall train subset:')
            print_memory_diagnostics(train_metrics)
        train_interval = dict(
            recall_lm=interval['recall_lm']/interval['recall_n'] if interval['recall_n'] else None,
            recall_ce=interval['recall_ce']/interval['recall_n'] if interval['recall_n'] else None,
            prose_lm=interval['prose_lm']/interval['prose_n'] if interval['prose_n'] else None)
        with (directory/'mixed_diagnostics.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(dict(event=event, metrics_schema_version=3,
                recall_includes_newline=True, **counters, tokens_this_run=counters['tokens']-launch_tokens,
                recall_ratio=ratio, ratio_basis='optimizer_updates', writer_config=writer_config(),
                activation_checkpointing=activation_checkpointing,
                lr_by_group={g['name']:g['lr'] for g in learning.param_groups},
                validation=metrics, prose_validation_loss=prose_loss, selection_loss=score,
                train_subset=train_metrics, train_interval=train_interval))+'\n')
        for key in interval:
            interval[key] = 0
        return score

    # Restore RNG after setup/cache construction so resume does not perturb dropout.
    if restored:
        random.setstate(restored['rng_python'])
        torch.set_rng_state(restored['rng_torch'])
        if device=='cuda' and restored['rng_cuda']:
            torch.cuda.set_rng_state_all(restored['rng_cuda'])
    baseline_score = evaluate_mixed('baseline')
    if baseline_score < best:
        best = baseline_score
        save(directory/'best.pt')
    if TOKEN_STOP_COUNT is not None and counters['tokens'] >= TOKEN_STOP_COUNT:
        print('Token stop already reached; no updates performed.')
        return
    next_eval = (counters['tokens']//EVAL_INTERVAL_TOKENS+1)*EVAL_INTERVAL_TOKENS
    next_save = (counters['tokens']//CHECKPOINT_INTERVAL_TOKENS+1)*CHECKPOINT_INTERVAL_TOKENS
    model.train()
    training_progress = MixedProgress('Mixed training updates')
    while True:
        kind = 'recall' if mixed_recall_update(counters['step'], ratio) else 'prose'
        try:
            batch = streams[kind].next_batch()
        except StopIteration:
            save(latest)
            print(f'{kind.title()} stream reached num_epochs; training complete.')
            return
        learning.zero_grad(set_to_none=True)
        lr = cosine_decay(counters['tokens'], WARMUP_TOKENS, MAX_TOKENS, learn_rate, MIN_LEARN_RATE)
        for group in learning.param_groups:
            group['lr'] = lr * group.get('lr_scale', 1.)
        # Keep the mode active through optimizer.step: inactive params have no
        # gradients, so AdamW cannot move them through weight decay or momentum.
        with mixed_prose_mode(model) if kind=='prose' else nullcontext():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device=='cuda'):
                if kind=='recall':
                    xb, yb, starts, ends = batch
                    logits, memory, language_loss, recall_loss = memory_training_forward(
                        model, xb, yb, starts, ends, checkpointing=activation_checkpointing)
                    loss = language_loss + RECALL_LOSS_WEIGHT*recall_loss
                    interval['recall_lm'] += language_loss.item()
                    interval['recall_ce'] += recall_loss.item()
                else:
                    xb, yb = batch
                    logits, loss, memory = model(xb, yb, update_memory=False)
                    interval['prose_lm'] += loss.item()
            scaler.scale(loss).backward()
            scaler.step(learning)
            scaler.update()
        interval[kind+'_n'] += 1
        count = xb.numel()
        counters['step'] += 1
        counters['tokens'] += count
        counters[kind+'_tokens'] += count
        counters[kind+'_updates'] += 1
        training_progress.update(
            f'step {counters["step"]:,}; tokens {counters["tokens"]:,}; '
            f'last batch {kind}; loss {loss.item():.4f}; LR {lr:.3e}; '
            f'recall/prose updates {counters["recall_updates"]:,}/{counters["prose_updates"]:,}')
        # Free graphs and GPU inputs before evaluation or the next pool load.
        batch = xb = yb = logits = memory = loss = None
        starts = ends = language_loss = recall_loss = None
        if counters['tokens'] >= next_eval:
            score = evaluate_mixed('interval')
            if score < best:
                best = score
                save(directory/'best.pt')
            while counters['tokens'] >= next_eval:
                next_eval += EVAL_INTERVAL_TOKENS
        stopped = ((TOKEN_STOP_COUNT is not None and counters['tokens'] >= TOKEN_STOP_COUNT)
                   or (STOP_AFTER_POOL_CT and
                       sum(s.completed for s in streams.values())-launch_pools >= STOP_AFTER_POOL_CT))
        if counters['tokens'] >= next_save or stopped:
            save(latest)
            while counters['tokens'] >= next_save:
                next_save += CHECKPOINT_INTERVAL_TOKENS
        if stopped:
            print('Stop limit reached; latest checkpoint saved.')
            return



if __name__ == '__main__':
    mixed_training()
