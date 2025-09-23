
import math, torch, torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from transformers import AutoTokenizer, AutoModel
finbert_model_name = "yiyanghkust/finbert-tone"
import torch.nn as nn

from torch import Tensor
class GatedCrossAttentionFusion2D(nn.Module):
    """
    Fusion module for MSGCA with 2D inputs:
    1. Cross-attention between two modalities.
    2. Gated feature selection guided by a primary modality.
    Inputs are 2D: (batch_size, d_model)
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Cross-attention projections (for queries, keys, values)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Gating mechanism projections
        self.gate_proj = nn.Linear(d_model, d_model)  # For primary modality
        self.unstable_proj = nn.Linear(d_model, d_model)  # For unstable features

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self,
                primary: torch.Tensor,
                auxiliary: torch.Tensor,
                ) -> torch.Tensor:
        """
        Args:
            primary:   (batch_size, d_model)
            auxiliary: (batch_size, d_model)
        Returns:
            fused:     (batch_size, d_model)
        """
        # --- Step 1: Unstable Cross-Attention Fusion ---
        Q = self.q_proj(primary)  # (batch, d_model)
        K = self.k_proj(auxiliary)
        V = self.v_proj(auxiliary)

        # Multi-head attention for 2D inputs
        batch_size, _ = Q.size()

        # Reshape for multi-head attention
        # (batch, n_heads, head_dim)
        Q = Q.view(batch_size, self.n_heads, self.head_dim)
        K = K.view(batch_size, self.n_heads, self.head_dim)
        V = V.view(batch_size, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        # (batch, n_heads, n_heads)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # (batch, n_heads, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Restore original shape: (batch, d_model)
        attn_output = attn_output.view(batch_size, self.d_model)

        # --- Step 2: Stable Gated Feature Selection ---
        # (batch, d_model)
        unstable_features = self.unstable_proj(attn_output)
        gate = torch.sigmoid(self.gate_proj(primary))

        # Element-wise gating
        # (batch, d_model)
        fused = unstable_features * gate

        # Output projection
        fused = self.out_proj(fused)
        return fused

class MSGCAFusion(nn.Module):
    """
    Complete fusion module for MSGCA:
    1. Fuses indicators + documents.
    2. Fuses (indicators + documents) + graph.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # First fusion: Indicators (primary) + Documents (auxiliary)
        self.fusion1 = GatedCrossAttentionFusion2D(d_model, n_heads)


    def forward(self,
                indicators: Tensor,  # (batch, seq_len, d_model)
                documents: Tensor  # (batch, seq_len, d_model)
                ) -> Tensor:
        # First fusion stage
        fused_id = self.fusion1(primary=indicators, auxiliary=documents)  # (batch, seq_len, d_model)

        return fused_id
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MarketNewsFusionWeightModel(nn.Module):
    def __init__(self, ts_input_dim, num_stocks, d_model=64, nhead=4, num_layers=2,
                 news_embed_dim=768, hidden_dim=64, count_dim=0, max_len=500):
        super().__init__()
        self.S = num_stocks
        self.d_model = d_model
        self.inp = nn.Linear(ts_input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.ts_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.news_proj = nn.Linear(news_embed_dim+self.S, 64)
        self.news_lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)
        self.count_dim = count_dim
        if count_dim > 0:
            self.count_lstm = nn.LSTM(input_size=count_dim, hidden_size=count_dim, batch_first=True)
        else:
            self.count_lstm = None
        fused_dim = d_model + hidden_dim + (count_dim if count_dim>0 else 0)
        self.fusion = nn.ModuleList([MSGCAFusion(self.d_model, n_heads=4) for _ in range(num_stocks)])

        self.stock_heads = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(self.S)
        ])
        
    def forward(self, ts_input, count_input, news_input):
        x = self.inp(ts_input); 
        x = self.pos(x); 
        ts_emb = self.ts_enc(x)[:, -1, :].squeeze(1)   
        
        concat_news = torch.cat([news_input, count_input], dim=2)
        news_proj = self.news_proj(concat_news)
        _, (hn, _) = self.news_lstm(news_proj); 
        news_emb = hn[-1]
        
        fused = [fusion(ts_emb, news_emb) for fusion in self.fusion]
        # fused = torch.cat([ts_emb, news_emb], dim=1)   # [B, 128]

        outputs = [self.stock_heads[ij](fused[ij]).squeeze(-1) for ij in range(self.S)]

        return torch.stack(outputs, dim=1)

