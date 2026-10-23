################################### TINY TRANSFORMER MODEL THAT SHOULD BE ABLE TO RECALL INFORMATION OUTSIDE OF CONTEXT WINDOW ####################################

import torch
import torch.nn as nn
from torch.nn import functional as F
import config as model_config
from config import (
    block_size,
    emb_dims,
    dropout,
    head_num,
    num_blocks,
    num_encoder_blocks,
    memory_slots,
    memory_slot_dims,
    memory_routing_passes,
    memory_pooling_heads,
    TEMPERATURE,
    TOP_K
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

amp_dtype = torch.float16 # Use FP16

class MemoryEncoderBlock(nn.Module):

    # This will create an encoding transformer that is bidirectional; it processes a complete window to build representations for memory
    ## The representation is then fed to the GRU to be compressed and fed into memory slots to update persistent memory

    def __init__(self, emb_dims, head_num, dropout, rope):
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dims)
        self.attention = MultiHeadAttention(head_num, emb_dims, dropout, rope)
        self.ln2 = nn.LayerNorm(emb_dims)
        self.ffwd = FeedForward(emb_dims)

    def forward(self, x):

        x = x + self.attention(self.ln1(x), is_causal=False)
        x = x + self.ffwd(self.ln2(x))
        return x

class MemoryEncoder(nn.Module):
    
    def __init__(self, emb_dims, head_num, dropout, rope, num_encoder_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([
            MemoryEncoderBlock(emb_dims, head_num, dropout, rope)
            for _ in range(num_encoder_blocks)
        ])
        self.ln_final = nn.LayerNorm(emb_dims)

    def forward(self, x):

        # Each block is [B, T, emb_dims]
        for block in self.blocks:
            x = block(x)

        return self.ln_final(x)
    
class MemoryCrossAttention(nn.Module):

    def __init__(self, emb_dims, memory_dims, head_num, dropout):
        super().__init__()

        assert emb_dims % head_num == 0

        self.emb_dims = emb_dims
        self.head_num = head_num
        self.head_size = emb_dims // head_num
        self.dropout = dropout

        # Current Transformer tokens will create queries
        self.query = nn.Linear(emb_dims, emb_dims, bias=False)

        # The Persisten Memory creates keys and values
        # Each Dm-wide slot is projected into C decoder attention channels.
        # This changes feature width; it does not restore discarded information.
        self.key = nn.Linear(memory_dims, emb_dims, bias=False)

        self.value = nn.Linear(memory_dims, emb_dims, bias=False)

        self.proj = nn.Linear(emb_dims, emb_dims, bias=False)

        self.dropout_residual = nn.Dropout(dropout) 

    def forward(self, x, memory):

        # x: [B, T, C]
        # memory: [B, M, Dm]

        B, T, C = x.shape
        M = memory.shape[1]

        # [B, T, C]
        q = self.query(x)

        # [B, M, C]
        k = self.key(memory)
        v = self.value(memory)

        q = q.view(B, T, self.head_num, self.head_size)
        k = k.view(B, M, self.head_num, self.head_size)
        v = v.view(B, M, self.head_num, self.head_size)

        # [B, T, H, D] ---> [B, H, T, D]
        q = q.transpose(1, 2)
        # [B, M, H, D] ---> [B, H, M, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # No causal mask because we are only dealing with memory tokens so we can safely attend to all tokens
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=(self.dropout if self.training else 0.0), is_causal=False)

        # [B, H, T, D] ---> [B, T, H, D]
        out = out.transpose(1, 2)
        out = out.contiguous().view(B, T, C,)
        out = self.proj(out)

        return self.dropout_residual(out)

class MemoryCompressor(nn.Module):
    """Competitive writer whose slot queries depend on incoming memory.

    B = batch size, T = tokens, C = decoder channels,
    M = memory slots, Dm = memory channels, Dk = write-attention channels.
    Iterative competitive pooling with an additive multi-head pooling branch.
    Legacy diagnostics describe only the original single-head branch.
    """

    def __init__(self, emb_dims, memory_slots, memory_dims, attention_dims=None, memory_routing_passes=3, pooling_heads=4):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dims = memory_dims
        self.attention_dims = memory_dims if attention_dims is None else attention_dims

        # [M, Dm]: distinct learned identities for the write queries only.
        # Incoming GRU memory still starts at zero; decoder reads no future writes.
        self.slot_embeddings = nn.Parameter(torch.empty(memory_slots, memory_dims))
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)

        # Normalize each vector's channels; neither operation mixes tokens/slots.
        self.input_norm = nn.LayerNorm(emb_dims)   # [B, T, C] -> [B, T, C]
        self.slot_norm = nn.LayerNorm(memory_dims) # [B, M, Dm] -> [B, M, Dm]

        self.query = nn.Linear(memory_dims, self.attention_dims, bias=False)
        # Slot features: [B, M, Dm] -> write queries: [B, M, Dk].
        self.key = nn.Linear(emb_dims, self.attention_dims, bias=False)
        # Token features: [B, T, C] -> write keys: [B, T, Dk].

        # Keep the original value projection and its checkpoint parameter name.
        self.value = nn.Linear(emb_dims, memory_dims, bias=False)
        # Decoder representations: [B, T, C] -> write values: [B, T, Dm].

        if memory_routing_passes < 1:
            raise ValueError("memory routing passes must be at least 1")

        self.routing_iterations = memory_routing_passes

        # Not persisent per-slot GRU
        ## Shared GRU across slots for refinement iterations
        self.refine_gru = nn.GRUCell(input_size=memory_dims, hidden_size=memory_dims)

        self.refine_delta = nn.Sequential(
            nn.LayerNorm(memory_dims),
            nn.Linear(memory_dims, 2 * memory_dims),
            nn.GELU(),
            nn.Linear(2 * memory_dims, memory_dims),
        )

        # No inital change from refinement to working slot queries.
        # Sequential indexing is typed as Module | Tensor, so guard the final layer.
        refine_out = self.refine_delta[-1]
        if not isinstance(refine_out, nn.Linear):
            raise TypeError("Expected the final refinement layer to be nn.Linear.")
        nn.init.zeros_(refine_out.weight)
        nn.init.zeros_(refine_out.bias)

        # Independent token routing for each head; concatenation keeps Dm fixed.
        if not isinstance(pooling_heads, int) or pooling_heads < 1:
            raise ValueError("memory_pooling_heads must be a positive integer")
        if memory_dims % pooling_heads or self.attention_dims % pooling_heads:
            raise ValueError("Memory and attention widths must divide evenly by pooling heads")
        self.pooling_heads = pooling_heads
        self.pooling_key_dims = self.attention_dims // pooling_heads
        self.pooling_value_dims = memory_dims // pooling_heads
        self.multihead_query = nn.Linear(memory_dims, self.attention_dims, bias=False)
        self.multihead_key = nn.Linear(emb_dims, self.attention_dims, bias=False)
        self.multihead_value = nn.Linear(emb_dims, memory_dims, bias=False)
        self.multihead_out = nn.Linear(memory_dims, memory_dims, bias=False)
        # Preserve the old function on migration. This projection learns first;
        # gradients reach the new Q/K/V projections once it becomes nonzero.
        nn.init.zeros_(self.multihead_out.weight)

        self.last_assignments = None
        self.last_weights = None
        self.last_head_assignments = None
        self.last_head_weights = None

    def forward(self, x, memory=None):
        # x: encoder features [B, T, C]
        B, T, C = x.shape
        M = self.memory_slots
        D = self.memory_dims

        if T == 0:
            raise ValueError("Cannot compress an empty window.")

        identities = self.slot_embeddings.unsqueeze(0).expand(B, -1, -1)

        if memory is None:
            working_slots = identities
        else:
            if tuple(memory.shape) != (B, M, D):
                raise ValueError(
                    "Incoming memory must have shape [B, M, Dm]."
                )

            # Temporary working state; does not modify persistent memory.
            working_slots = memory + identities

        # Fixed throughout this window's routing iterations.
        normalized_x = self.input_norm(x)
        keys = self.key(normalized_x)  # [B, T, Dk]
        values = self.value(x)               # [B, T, D]
        H = self.pooling_heads
        head_keys = self.multihead_key(normalized_x).reshape(
            B, T, H, self.pooling_key_dims
        ).transpose(1, 2)  # [B, H, T, Dk/H]
        head_values = self.multihead_value(x).reshape(
            B, T, H, self.pooling_value_dims
        ).transpose(1, 2)  # [B, H, T, Dm/H]
        pooled = None
        assignments = None
        weights = None
        head_assignments = None
        head_weights = None

        for iteration in range(self.routing_iterations):
            normalized_slots = self.slot_norm(working_slots)
            queries = self.query(normalized_slots)  # [B, M, Dk]
            head_queries = self.multihead_query(normalized_slots).reshape(
                B, M, H, self.pooling_key_dims
            ).transpose(1, 2)  # [B, H, M, Dk/H]

            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = (
                    keys.float()
                    @ queries.float().transpose(1, 2)
                )
                scores = scores * (self.attention_dims ** -0.5)

                # Each token competes across slots.
                assignments = scores.softmax(dim=-1)  # [B, T, M]

                # Normalize across tokens for each slot.
                slot_mass = assignments.sum(dim=1, keepdim=True)
                weights = assignments / slot_mass.clamp_min(1e-8)

            pooled = (
                weights.to(values.dtype).transpose(1, 2) @ values
            )  # [B, M, D]

            with torch.autocast(device_type=x.device.type, enabled=False):
                head_scores = (
                    head_keys.float() @ head_queries.float().transpose(-2, -1)
                ) * (self.pooling_key_dims ** -0.5)
                # Independent slot competition within each head.
                head_assignments = head_scores.softmax(dim=-1)  # [B,H,T,M]
                head_weights = head_assignments / head_assignments.sum(
                    dim=-2, keepdim=True
                ).clamp_min(1e-8)
            head_pooled = head_weights.to(head_values.dtype).transpose(-2, -1) @ head_values
            head_pooled = head_pooled.transpose(1, 2).contiguous().reshape(B, M, D)
            pooled = pooled + self.multihead_out(head_pooled)

            # Refine from the combined candidate; update persistent GRUs only
            # once, after all routing iterations finish.
            # No refinement is needed after the final pooling pass.
            if iteration + 1 < self.routing_iterations:
                proposal = self.refine_gru(
                    pooled.reshape(B * M, D),
                    working_slots.to(pooled.dtype).reshape(B * M, D),
                ).reshape(B, M, D)

                working_slots = (
                    working_slots + self.refine_delta(proposal)
                )

        if pooled is None:
            raise RuntimeError("Memory routing must run at least one iteration.")

        # Diagnostics describe the FINAL routing pass.
        self.last_assignments = assignments.detach() if assignments is not None else None
        self.last_weights = weights.detach() if weights is not None else None

        self.last_head_assignments = (
            head_assignments.detach() if head_assignments is not None else None
        )
        self.last_head_weights = head_weights.detach() if head_weights is not None else None
        # Older audits that rebuild a candidate using value + last_assignments
        # omit the residual branch and must be updated for such interventions.

        # Persistent GRUs receive the final pooled candidate, as before.
        return pooled

class PersistentMemory(nn.Module):

    # Separate GRUCell weights update M independent slot states, each with Dm channels.

    def __init__ (self, memory_slots, memory_dims):
        super().__init__()

        self.memory_slots = memory_slots
        self.memory_dims = memory_dims


        # Each memory slot is now assigned a GRU
        self.grus = nn.ModuleList([
            nn.GRUCell(
                input_size=memory_dims, 
                hidden_size=memory_dims
            )
            for _ in range(memory_slots)                       
        ])

    def init_memory(self, batch_size, device, dtype):

        # [B, M, Dm]
        return torch.zeros(batch_size, self.memory_slots, self.memory_dims, device=device, dtype=dtype)

    def forward(self, block_state, memory):
        # Both inputs: [B, M, Dm]
        updated_slots = [
            gru(
                block_state[:, slot, :],  # [B, Dm]
                memory[:, slot, :],       # [B, Dm]
            )
            for slot, gru in enumerate(self.grus)
        ]

        # M tensors of [B, Dm] -> [B, M, Dm]
        return torch.stack(updated_slots, dim=1)

class RotaryPositionalEmbedding(nn.Module):
    
    # Rotary Positional Embedding (RoPE) is a method for encoding positional information that rotates the vectors from the channel dim instead of recalculating the absolute posiiton of vectors every time to update affinity
    # This model uses RoPE rather than a learned absolute-position table.

    def __init__(self, dim, block_size, base=10000):
        super().__init__()

        assert dim % 2 == 0

        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor

        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.update_cos_sin_cache(block_size)

    def update_cos_sin_cache(self, block_size):

        if not isinstance(self.inv_freq, torch.Tensor):
            raise TypeError("inv_freq must be a tensor buffer")
        t = torch.arange(block_size, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq) # [T] outer [D/2] ---> [T, D/2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, input_block_size):

        if input_block_size > self.cos_cached.shape[0]:
            self.update_cos_sin_cache(input_block_size)

        # [T, D/2] ---> [1, T, 1, D/2]
        cos = self.cos_cached[:input_block_size, :].to(device=x.device).unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:input_block_size, :].to(device=x.device).unsqueeze(0).unsqueeze(2)

        return cos, sin

    def apply_rope(self, x, cos, sin):

        # x: [B, T, H, D]

        original_dtype = x.dtype

        x1 = x[..., : self.dim // 2].float() # [B, T, H, D/2]
        x2 = x[..., self.dim // 2:].float() # [B, T, H, D/2]

        cos = cos.float()
        sin = sin.float()

        # Apply 2D rotation to corresponding halves
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        # [B, T, H, D/2] + [B, T, H, D/2] ---> [B, T, H, D]
        return torch.cat((out1, out2), dim=-1).to(original_dtype) # keeps the original FP32 for accurate sin cos calculations and outputs back to original dtype
        

class MultiHeadAttention(nn.Module):

    # Multiple Attention Heads in Parallel; Takes place of AttentionHead class with integrated attention scores
    ## Now uses FlashAttention call via Pytorch for better I/O performance


    def __init__(self, head_num, emb_dims, dropout, rope):
        super().__init__()

        assert emb_dims % head_num == 0, "Ensure emb_dims / head_num divides without remainder in hyperparameters"

        self.emb_dims = emb_dims
        self.head_num = head_num
        self.head_size = emb_dims // head_num
        self.dropout = dropout

        # self.head_num = head_num # H = number of attention heads
        # self.head_size = head_size # D = how many channels each head contains


        # Linear Projection:
        ## Input: [B, T, C]
        ## Output: [B, T, C]
        self.query = nn.Linear(emb_dims, emb_dims, bias=False)
        self.key = nn.Linear(emb_dims, emb_dims, bias=False)
        self.value = nn.Linear(emb_dims, emb_dims, bias=False)

        # RoPE over each head's D dimension (channels)
        self.rope = rope

        # Final projection after recombining all heads
        ## [B, T, C] ---> [B, T, C] 
        self.proj = nn.Linear(emb_dims, emb_dims)

        # Final application of dropout
        self.dropout_residual = nn.Dropout(dropout)

    def forward(self, x, is_causal=True):

        # x: [B, T, C]
        ## where:
        ## B = batch_size, T = sequence dimension of tokens, C = total embedding dimension
        B, T, C = x.shape

        # [B, T, C] ---> [B, T, C]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # C = H * D
        # [B, T, C] ---> [B, T, H, D] (C dimension is split into H and D dimensions)
        q = q.view(B, T, self.head_num, self.head_size)
        k = k.view(B, T, self.head_num, self.head_size)
        v = v.view(B, T, self.head_num, self.head_size)


        # RoPE Application
        # q = [B, T, H, D]
        # cos/sin from RoPE = [1, T, 1, D/2]
        ## Broadcasting:
        ### q:   [B, T, H, D]
        ### cos: [1, T, 1, D]
        ###      [B, T, H, D] 
        cos, sin = self.rope(q, T) 

        # Before: q, k = [B, T, H, D]
        # After : q, k = [B, T, H, D]
        # values along dim D are rotated according to their position in dim T
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)

        # Rearrange for Attention Score Computation
        ## [B, T, H, D] ---> [B, H, T, D]
        q = q.transpose(1,2) 
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # FlashAttention AKA scaled dot-product attention
        ## Bypasses the need to load the whole [T,T] matrix in HBM memory in favor of chunks of [T,T] being loaded into smaller but faster SRAM

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        # is_causal handles lower triangular masking for decoder transformers

        # # Put Token Dim back before Head Dim
        out = out.transpose(1,2)

        # # Recombine heads (H * D = C)
        # # [B, T, H, D] ---> [B, T, C]
        out = out.contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.dropout_residual(out)

        return out
    

class FeedForward(nn.Module):

    # Attention values passed through a simple MLP

    def __init__(self, emb_dims):
         super().__init__()
         self.net = nn.Sequential(
             nn.Linear(emb_dims, 4 *emb_dims),
             nn.ReLU(),
             nn.Linear(4 * emb_dims, emb_dims), # project back to residual pathway; x4 used because attention paper used it
             nn.Dropout(dropout)
         )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    # Creates a Transformer Block

    def __init__(self, emb_dims, head_num, rope, memory_dims,):
        super().__init__()
        self.at_head = MultiHeadAttention(head_num, emb_dims, dropout, rope)
        self.memory_attention = MemoryCrossAttention(emb_dims, memory_dims, head_num, dropout)
        self.ffwd = FeedForward(emb_dims)
        # LayerNorm is applied to the channel dimension C as opposed to BatchNorm which is applied to dimension B
        self.ln1 = nn.LayerNorm(emb_dims) 
        self.ln2 = nn.LayerNorm(emb_dims)
        self.memory_ln = nn.LayerNorm(memory_dims)
        self.ln3 = nn.LayerNorm(emb_dims)

    def forward(self, x, memory):
        # Layer Normalization is applied before the attention and feedforward layers, and skip connections are added after each layer to reduce backpropagation load
        x = x + self.at_head(self.ln1(x))
        # Seperate query into persistent memory 
        x = x + self.memory_attention(self.ln2(x), self.memory_ln(memory),)
        # MLP
        x = x + self.ffwd(self.ln3(x))
        return x


# RNN(GRU) + CrossAttention Transformer Hybrid
class MiniGPTRecall(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        head_size = emb_dims // head_num
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dims)
        # self.position_embedding_table = nn.Embedding(block_size, emb_dims) # An embedded matrix that stores positional info of tokens ie. where they exist in a chunk of context
        self.rope = RotaryPositionalEmbedding(head_size, block_size)
        # Stacking num_blocks of Transformer blocks
        self.blocks = nn.ModuleList([Block(emb_dims, head_num, self.rope, memory_slot_dims) for _ in range(num_blocks)]) 
        self.ln_final = nn.LayerNorm(emb_dims) # Final layer normalization before output
        self.lm_head = nn.Linear(emb_dims, vocab_size, bias=False)
        self.lm_head.weight = (self.token_embedding_table.weight) # Makes the final layer share weights with the embedding table     
        # Encoder - writer to memory
        self.memory_encoder = MemoryEncoder(emb_dims, head_num, dropout, self.rope, num_encoder_blocks)  
        # Use weighted average
        self.memory_compressor = MemoryCompressor(emb_dims=emb_dims, memory_slots=memory_slots, memory_dims=memory_slot_dims, memory_routing_passes=memory_routing_passes, pooling_heads=memory_pooling_heads)
        # Persistent memory mechanism
        self.memory = PersistentMemory(memory_slots=memory_slots, memory_dims=memory_slot_dims)
        nn.init.normal_(
            self.token_embedding_table.weight,
            mean=0.0,
            std=0.02
        ) # Reduce the initial scale of embedding matrix and output matrix to prevent large early losses

        # Import chunk size to model
        self.memory_write_chunk_size = getattr(
            model_config,
            "memory_write_chunk_size",
            block_size,
        )

        if (
            type(self.memory_write_chunk_size) is not int
            or self.memory_write_chunk_size <= 0
            or block_size % self.memory_write_chunk_size != 0
        ):
            raise ValueError(
                "memory_write_chunk_size must be a positive integer "
                "that divides block_size."
            )


    def forward(self, idx, targets=None, memory=None, update_memory=True):
        B,T = idx.shape

        token_embedding = self.token_embedding_table(idx) # [B,T,C]; C here is embed C

        # Initialize memory at beginning of a sequence
        ## [B, memory_slots, memory_slot_dims]
        if memory is None:
            memory = self.init_memory(batch_size=B, device=token_embedding.device, dtype=token_embedding.dtype)

        # Normal Transformer
        x = token_embedding

        for block in self.blocks:
            x = block(x, memory,)

        x = self.ln_final(x)

        # Decoder features retain [B, T, C]; no extra memory token is appended.
        token_output = x
        logits = self.lm_head(token_output)

        # We want to avoid updating the memory on every token, instead we update on new chunks
        if update_memory:

            # Encoder reads the entire input window bidirectionally and sends to compressor [B, T, C]
            memory_features = self.memory_encoder(token_embedding) 

            # # Write queries use incoming memory [B, M, Dm].
            # # Completed-window features [B, T, C] -> pooled writes [B, M, Dm].
            # block_state = self.memory_compressor(memory_features, memory)

            # # Pool [B, M, Dm] + incoming state [B, M, Dm]
            # # -> next-window state [B, M, Dm]. Current logits read only old memory.
            # new_memory = self.memory(block_state, memory)

            # Integrate smaller pieces sequentially
            new_memory = self.write_memory_features(memory_features, memory,)

        else:
            new_memory = memory
    
        # Language Model Loss
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_memory

    def init_memory(self, batch_size, device, dtype):
        
        # Initializes memory as a run-time state rather than part of model's parameter

        return self.memory.init_memory(batch_size, device, dtype)

    def write_memory_features(self, memory_features, memory):

        # Memory_features = [B, T, emb_dims]
        # Memory = [B, memory_slots, memory_slot_dims]
        # Integrates consecutive chunks of a single window processed using shared compressor/GRU weights

        for chunk in memory_features.split(self.memory_write_chunk_size, dim=1):
            # Each chunk routes against the most recently updated memory
            candidate = self.memory_compressor(chunk, memory)
            # Each per-slot GRU integrates the candidate as it sees fit
            memory = self.memory(candidate, memory)
        
        return memory
    
    @torch.no_grad()
    def generate(self, idx, max_tokens, memory=None):
        for _ in range(max_tokens):
            # clip the idx size to block_size so only use most recent tokens
            idx_clip = idx[:, -block_size:] 
            logits, _, _ = self(idx_clip, memory=memory, update_memory=False) # forward pass
            # logit selection for only next-token prediction
            logits = logits[:, -1, :] 
            # apply temperature adjustment
            logits = logits / TEMPERATURE
            # apply top-k filtering
            values, _ = torch.topk(logits, TOP_K)
            # apply mask to cutoff any logit that is smaller than the minimum logit allowed for top_k
            logits = logits.masked_fill(
                logits < values[:, [-1]],
                float("-inf")
            )
            # sample from one the candidates left
            probs = F.softmax(logits, dim=-1)
            # idx_next = torch.multinomial(probs, num_samples=1)
            # TEMPORARY CHANGE
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            # append token to full generated sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


