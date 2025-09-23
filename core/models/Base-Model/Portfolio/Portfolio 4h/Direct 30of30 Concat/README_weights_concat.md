# Direct Portfolio Weight Forecasting with News + OHLCV (Concat Fusion, Sharpe-Ratio Loss)

This project directly forecasts **portfolio weights** for a group of assets using multi-modal data (news + OHLCV).  
Unlike return/distribution modeling, the network outputs weights end-to-end and is trained with a **Sharpe-ratio-based loss**.

---

## Architecture (Concat Fusion)

- **OHLCV encoder:** a **Transformer** over 3-minute bars.
- **News-text encoder:** an **LSTM** over per-bar **BigBird** news embeddings (per timestamp, multiple news are averaged; if no news, use a learned **[NO_NEWS]** embedding).
- **News-coverage encoder:** an **LSTM** over per-bar **one-hot** vectors indicating which stocks were mentioned.
- **Fusion:** **concatenate** the three embeddings → **Linear head** → **predicted weights** for all assets.

No cross-gating is used in this variant; fusion is a straightforward **concat + linear**.

---

## Data & Setup

- **Timeframe:** 3-minute bars  
- **Input Window:** 80 timestamps (≈ 4 hours)  
- **Prediction Horizon:** next 80 timestamps (≈ 4 hours)  
- **Assets:** configurable set of stocks

**News processing**
- Encode each article with **BigBird**.
- Aggregate multiple news per bar by **mean**.
- If no news in a bar, use a **learned [NO_NEWS] embedding**.

**Coverage one-hot**
- Per bar, create a binary vector (length = #assets) indicating which assets are mentioned.
- Feed the sequence of one-hots to the coverage **LSTM**.

---

## Loss (Training Objective)

We optimize a **Sharpe Ratio Loss** over the prediction horizon:
\[
\mathcal{L} = - \frac{\mathbb{E}[R_p]}{\sqrt{\operatorname{Var}(R_p) + \epsilon}},
\]
where \(R_p\) are realized portfolio returns using the **predicted weights**.

Regularization tips:
- Add turnover penalties or L2 on weight changes.
- Clip/logit-transform raw outputs before normalization if needed.

---

## Portfolio Construction (Test Phase)

- Use the **predicted weights** directly to form the portfolio.
- Optional **top-k selection**:
  1) Select top-k assets by absolute weight
  2) Re-normalize the selected weights to **sum to 1**
- Supports **long-only** or **long/short** depending on your normalization step.

---

## How to Run

1. Prepare data:
   - **OHLCV** (3-min bars) aligned across assets
   - **News table** with text + asset mentions (coverage)
2. Open the notebook:
   ```
   Transformer+LLM_Concat_portfolio_30of30_news.ipynb
   ```
3. Train:
   - Best checkpoint selected by **validation Sharpe** (or proxy metric).
4. Test:
   - Generate predicted weights for each horizon
   - (Optional) apply **top-k** filtering
   - Evaluate portfolio (Sharpe, P&L, drawdown, turnover)

---

## Dependencies

- Python 3.10+
- PyTorch 2.x
- Hugging Face `transformers` (BigBird)
- numpy, pandas, matplotlib, scikit-learn

Install:
```bash
pip install torch transformers numpy pandas matplotlib scikit-learn
```

---

## Outputs

- Predicted portfolio weights per timestamp (CSV/Parquet)
- Performance metrics:
  - Sharpe ratio
  - Cumulative P&L
  - Max Drawdown
  - Turnover / trading cost analysis (if enabled)
- Plots: equity curve, rolling Sharpe, weights heatmap

---

## Notes

- **Concat fusion** keeps the architecture simple and efficient.
- Consider adding **transaction-cost-aware** loss for production.
- Extensions:
  - Replace concat with **cross-gated fusion** or cross-attention
  - Multi-objective loss (Sharpe + turnover + drawdown)
  - Regime conditioning by news intensity or realized volatility

---
