"""Export inference/weights-only checkpoint without optimizer, RNG or file paths.

Run on your own training checkpoint. This export cannot resume optimizer state.
Retains only scalar architecture and writer metadata from an allowlist.
"""
import argparse
import json
from pathlib import Path


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve():
        p.error('Choose a separate output file; keep your resumable checkpoint.')
    data = torch.load(args.input, map_location='cpu', weights_only=True)
    state = data['model_state_dict']
    config_keys = ('emb_dims','head_num','num_blocks','num_encoder_blocks','block_size')
    writer_keys = ('block_size','memory_slots','memory_slot_dims','sequence_windows',
                   'routing_passes','pooling_heads','write_chunk_size','encoder_scope')
    sidecar = args.input.with_name(args.input.name+'.writer.json')
    saved = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    def select(source, keys):
        return {k:source[k] for k in keys if k in source and isinstance(source[k], (int,float,str,bool))}
    architecture = select(data.get('model_config',saved.get('model_config',{})), config_keys)
    writer = select(data.get('writer_config',saved.get('writer_config',{})), writer_keys)
    payload = dict(model_state_dict=state, model_config=architecture, writer_config=writer)
    if isinstance(data.get('tokens_seen'), int):
        payload['tokens_seen'] = data['tokens_seen']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload,args.output)
    args.output.with_name(args.output.name+'.writer.json').write_text(
        json.dumps(dict(model_config=architecture,writer_config=writer),indent=2)+'\n')
    print(f'Saved {args.output.name}: {args.output.stat().st_size/1024**2:.1f} MiB')
    if not writer or not architecture:
        print('Source lacks complete metadata; distribute its matching config separately.')


if __name__ == '__main__':
    main()
