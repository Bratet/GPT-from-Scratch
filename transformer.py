import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % h == 0
        
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, value, key, mask=None, causal=False):
        batch_size = query.size(0)
        
        # Linear Projections
        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        key = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if causal:
            # Create a mask to prevent attending to future positions
            seq_len = scores.size(-1)
            future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(scores.device)
            scores.masked_fill_(future_mask.bool(), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, value)
        
        # Output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.linear_out(attention_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Define layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Define dropout layers (optional but common for regularization)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Multi-head attention followed by residual connection and normalization
        attn_output = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network followed by residual connection and normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Masked multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, h)
        
        # Multi-head attention over encoder's output
        self.src_attn = MultiHeadAttention(d_model, h)
        
        # Position-wise feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Define layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Define dropout layers for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked multi-head self-attention with causal=True
        attn_output1 = self.self_attn(x, x, x, mask=tgt_mask, causal=True)
        x = x + self.dropout1(attn_output1)
        x = self.norm1(x)
        
        # Multi-head attention over encoder's output
        attn_output2 = self.src_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = x + self.dropout2(attn_output2)
        x = self.norm2(x)
        
        # Position-wise feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x
    

class Transformer(nn.Module):
    def __init__(self, src_token_size, tgt_token_size, d_model, h, d_ff, num_layers, dropout=0.1, max_len=500):
        super(Transformer, self).__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_token_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_token_size, d_model)
        
        # Positional Encodings
        self.src_positional_encoding = PositionalEncoding(d_model, max_len)
        self.tgt_positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder stacks
        self.encoder = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        
        # Final linear layer for Decoder
        self.output_layer = nn.Linear(d_model, tgt_token_size)
        
        self.d_model = d_model
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Add embeddings and positional encodings
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src * math.sqrt(self.d_model))
        
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt * math.sqrt(self.d_model))
        
        # Pass through Encoder
        for layer in self.encoder:
            src = layer(src)
            
        # Pass through Decoder
        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask=src_mask, tgt_mask=tgt_mask)
            
        # Output layer
        output = self.output_layer(tgt)
        
        return output
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        idx = idx.to(self.device)
        for _ in range(max_new_tokens):
            logits = self(idx, idx)  
            logits = logits[:, -1, :] / temperature  
            
            
            if top_k is not None:
                top_vals, top_indices = logits.topk(top_k, dim=-1)
                mask = logits < top_vals[:, -1, None]
                logits[mask] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
