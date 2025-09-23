
import math, torch, torch.nn as nn, torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MarketNewsFusionModel(nn.Module):
    def __init__(self, ts_input_dim, num_stocks, d_model=64, nhead=4, num_layers=2,
                 news_embed_dim=768, hidden_dim=64, count_dim=0, max_len=500):
        super().__init__()
        self.S = num_stocks
        self.d_model = d_model

        self.inp = nn.Linear(ts_input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.ts_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.news_proj = nn.Linear(news_embed_dim, 64)
        self.news_lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)

        self.count_dim = count_dim
        if count_dim > 0:
            self.count_lstm = nn.LSTM(input_size=count_dim, hidden_size=count_dim, batch_first=True)
        else:
            self.count_lstm = None

        fused_dim = d_model + hidden_dim + (count_dim if count_dim>0 else 0)
        self.mean_head = nn.Linear(fused_dim, num_stocks)
        self.chol_head  = nn.Linear(fused_dim, num_stocks*(num_stocks+1)//2)

    def forward(self, ts_input, count_input, news_input):
        B = ts_input.size(0)
        x = self.inp(ts_input)
        x = self.pos(x)
        x = self.ts_enc(x)[:, -1, :]

        n = self.news_proj(news_input)
        _, (hn, _) = self.news_lstm(n)
        n_emb = hn[-1]

        if self.count_lstm is not None:
            _, (hc, _) = self.count_lstm(count_input)
            c_emb = hc[-1]
            fused = torch.cat([x, n_emb, c_emb], dim=1)
        else:
            fused = torch.cat([x, n_emb], dim=1)

        mu = self.mean_head(fused)
        chol_params = self.chol_head(fused)

        L = torch.zeros(B, self.S, self.S, device=ts_input.device)
        idx = torch.tril_indices(row=self.S, col=self.S, offset=0, device=ts_input.device)
        L[:, idx[0], idx[1]] = chol_params
        diag = torch.arange(self.S, device=ts_input.device)
        L[:, diag, diag] = F.softplus(L[:, diag, diag]) + 1e-3
        return mu, L
