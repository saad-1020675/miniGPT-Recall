"""Print unique parameter counts, including tied embeddings only once."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from miniGPT_model_RNN import MiniGPTRecall
from tokenizer import vocab_size

model = MiniGPTRecall(vocab_size)
print(f'Backbone width: {config.emb_dims}; memory: {config.memory_slots} x {config.memory_slot_dims}')
print(f'Encoder/decoder blocks: {config.num_encoder_blocks}/{config.num_blocks}')
print(f'Unique parameters: {sum(p.numel() for p in model.parameters()):,}')
for name in ('token_embedding_table', 'blocks', 'memory_encoder', 'memory_compressor', 'memory'):
    print(f'{name}: {sum(p.numel() for p in getattr(model,name).parameters()):,}')
print('lm_head shares token_embedding_table.weight; do not add it again.')
