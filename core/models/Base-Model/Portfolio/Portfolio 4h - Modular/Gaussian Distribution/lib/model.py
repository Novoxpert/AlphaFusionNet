import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
from transformers import AutoTokenizer, AutoModel
finbert_model_name = "yiyanghkust/finbert-tone"
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerReturnPredictor(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, max_len=500):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        x = self.input_proj(x)             # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)            # Add positional encoding
        x = self.transformer(x)            # [batch_size, seq_len, d_model]
        x = x[:, -1, :]                    # Use representation of last time step
        return self.output_layer(x).squeeze(-1)


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


class MarketNewsFusionModel(nn.Module):
    def __init__(self, ts_input_dim, num_stocks, d_model=64, nhead=4, num_layers=2,
                 news_embed_dim=768, hidden_dim=64, count_dim=0, max_len=500):

        super().__init__()
        self.num_stocks = num_stocks

        self.d_model = d_model
        self.input_proj = nn.Linear(ts_input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ts_out = nn.Linear(128, hidden_dim)
        news_embed_dim = 768

        # 2. News LSTM
        self.news_proj = nn.Linear(news_embed_dim+self.num_stocks, 64)
        self.news_lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)

        # self.news_proj = nn.Linear(news_embed_dim, 64)
        # self.count_lstm = nn.LSTM(input_size=len(seleted_crypto), hidden_size=len(seleted_crypto), batch_first=True)
        self.fusion = nn.ModuleList([MSGCAFusion(self.d_model, n_heads=4) for _ in range(num_stocks)])

        # 3. Stock-specific regression heads (1 per stock)
        # self.stock_heads = nn.Sequential(
        #         nn.Linear(self.d_model+hidden_dim+len(seleted_crypto), 64),
        #         nn.ReLU(),
        #         nn.Linear(64, num_stocks)
        #     )

        # 3. Stock-specific regression heads (1 per stock)
        self.stock_heads = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            for _ in range(self.num_stocks)
        ])

    def forward(self, ts_input, count_input, news_input):

                # Transformer on OHLCV
        x = self.input_proj(ts_input)           # [B, T, d_model]
        x = self.pos_encoder(x)
        x = self.transformer(x)                  # [B, T, d_model]
        ts_emb = x[:, -1, :].squeeze(1)                   # [B, d_model]

        concat_news = torch.cat([news_input, count_input], dim=2)   # [B, 128]
        news_proj = self.news_proj(concat_news)         # [B, 30, 64]

        _, (hn, _) = self.news_lstm(news_proj)        # hn: [1, B, 64]
        news_emb = hn[-1]
        fused = [fusion(ts_emb, news_emb) for fusion in self.fusion]
        # fused = torch.cat([ts_emb, news_emb], dim=1)   # [B, 128]
        outputs = [self.stock_heads[ij](fused[ij]).squeeze(-1) for ij in range(self.num_stocks)]
        # print(outputs)
        logits = torch.stack(outputs, dim=1)
        mu = logits[..., 0]                 # [B, num_stocks]
        sigma_raw = logits[..., 1]          # [B, num_stocks]
        sigma = F.softplus(sigma_raw) + 1e-6
        return mu, sigma
