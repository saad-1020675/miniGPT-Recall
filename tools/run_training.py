"""Seed a selected original training loop without duplicating its implementation."""
import argparse
import os
from pathlib import Path
import random
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['standard','recall','mixed'])
    p.add_argument('--config',type=Path,help='JSON config overrides; applied before model import.')
    p.add_argument('--seed',type=int,default=42)
    stages=p.add_mutually_exclusive_group()
    stages.add_argument('--decoder-only',action='store_true')
    stages.add_argument('--memory-only',action='store_true')
    args=p.parse_args()
    if args.mode=='mixed' and (args.decoder_only or args.memory_only):
        p.error('Mixed training updates both objectives; exclusive modes are for single-stream training.')
    if args.mode=='standard' and args.memory_only:
        p.error('--memory-only requires recall mode')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    if args.config:os.environ['MINIGPT_CONFIG']=str(args.config.resolve())
    import torch
    import config
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    # Full resume restores the saved RNG state after these startup defaults.
    config.GUTENBERG_DECODER_ONLY=args.decoder_only if args.mode=='recall' else False
    config.GUTENBERG_MEMORY_ONLY=args.memory_only if args.mode=='recall' else False
    config.STANDARD_DECODER_ONLY=args.decoder_only if args.mode=='standard' else False
    config.MEMORY_TEST=args.mode=='recall'
    module='training_RNN_mixed.py' if args.mode=='mixed' else 'training_RNN_gutenberg.py'
    print(f'Training mode={args.mode}, startup seed={args.seed}, loop={module}',flush=True)
    runpy.run_path(str(root/module),run_name='__main__')


if __name__=='__main__':
    main()
