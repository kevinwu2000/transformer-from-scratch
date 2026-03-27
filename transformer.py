import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = n_embd // n_head # dimension for key

        self.q_linear = nn.Linear(n_embd, n_embd)
        self.k_linear = nn.Linear(n_embd, n_embd)
        self.v_linear = nn.Linear(n_embd, n_embd)
        
        self.out_proj = nn.Linear(n_embd, n_embd)
        
    def forward(self, x, mask=None):
        # Batch size, Sentence length, Embedding dimension
        B, L, D = x.size()

        # Project x to Q, K, V and reshape for multi-head: (B, L, n_head, d_k) -> (B, n_head, L, d_k)
        q = self.q_linear(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        
        # Calculate Attention Scores (Q * K^T / sqrt(d_k))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # Multiply by Values
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads back together: (B, L, n_embd)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        
        # Return output, attention weights
        return self.out_proj(output), attn_weights

class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_embd, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * n_embd  # Feed forward dimension
            
        self.linear1 = nn.Linear(n_embd, d_ff)
        self.linear2 = nn.Linear(d_ff, n_embd)
   
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff=None):
        super().__init__()
        self.mha = MultiHeadAttention(n_embd, n_head)
        self.ffn = PositionWiseFeedForward(n_embd, d_ff)
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x, mask=None):
        # Self-Attention with Pre-LayerNorm and Residual Connection
        attn_out, attn_weights = self.mha(self.ln1(x), mask)
        x = x + attn_out
        
        # Feed-Forward with Pre-LayerNorm and Residual Connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        
        # Stack of n_layer EncoderBlocks
        self.blocks = nn.ModuleList([EncoderBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # batch_size, sentence length
        B, L = x.size()
        
        # Generate the padding mask
        # (x != 0) creates a boolean tensor where True = valid word, False = <pad> token
        # unsqueeze twice to shape it as (B, 1, 1, L) so it broadcasts over heads and sequence length
        pad_mask = (x != 0).unsqueeze(1).unsqueeze(2)
        
        # Generate position indices: [0, 1, 2, ..., L-1] and copy for the whole batch
        positions = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, L)
        
        # Add Token Embeddings and Positional Embeddings
        out = self.token_embedding(x) + self.pos_embedding(positions)
        
        # Keep track of attention maps for the sanity check
        all_attn_maps = []
        
        # Pass data through each encoder block
        for block in self.blocks:
            out, attn_weights = block(out, mask=pad_mask)
            all_attn_maps.append(attn_weights)
            
        # Apply the final layer normalization
        out = self.ln_out(out)
        
        # Return the final embeddings and the attention maps
        return out, all_attn_maps
    
class SpeechClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        
    def forward(self, x):
        return self.network(x)
    

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        
        # Stack EncoderBlocks, set d_ff=100
        self.blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head, d_ff=100) for _ in range(n_layer)
        ])
        self.ln_out = nn.LayerNorm(n_embd)
        
        # LM Head: maps embeddings back to vocabulary probabilities
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x, targets=None):
        # batch_size, sentence length
        B, L = x.size()
        
        # Mask to block future
        look_ahead_filter = torch.tril(torch.ones(L, L, device=x.device))
        
        # Reshape to (1, 1, L, L) so it broadcasts over Batch and Heads
        mask = look_ahead_filter.unsqueeze(0).unsqueeze(1)
        
        # Embeddings
        positions = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, L)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        all_attn_maps = []
        
        # Pass through blocks with the filter
        for block in self.blocks:
            x, attn_weights = block(x, mask=mask)
            all_attn_maps.append(attn_weights)
            
        x = self.ln_out(x)
        
        # Compute Logits
        logits = self.lm_head(x) # Shape: (B, L, V)
        
        # Compute Loss (if targets are provided)
        loss = None
        if targets is not None:
            B, L, V = logits.shape
            # Flattening for the loss function
            loss = F.cross_entropy(logits.view(B * L, V), targets.view(B * L))
            
        return logits, loss, all_attn_maps
    
class WindowedTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head, d_ff=100) for _ in range(n_layer)
        ])
        
        self.ln_out = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x, targets=None):
        B, T = x.size()
        
        # Mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        
        # Apply local window: Zero out anything further back than window_size
        # torch.triu keeps the upper part of the matrix from the specified diagonal
        window_mask = torch.triu(causal_mask, diagonal=-self.window_size + 1)
        
        # Reshape for broadcasting
        mask = window_mask.unsqueeze(0).unsqueeze(1)
        
        # Embeddings & blocks
        positions = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        all_attn_maps = []
        for block in self.blocks:
            x, attn_weights = block(x, mask=mask)
            all_attn_maps.append(attn_weights)
            
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = torch.nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            
        return logits, loss, all_attn_maps