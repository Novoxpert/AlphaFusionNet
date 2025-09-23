# 🔍 Architecture Exploration & Reasoning

As part of our algorithmic trading research, we explored different methods to fuse **news data** and **OHLCV time series** for predicting asset volatility. Our goal was to determine how to best represent, align, and integrate these two modalities. Below is a summary of our progression and reasoning that led to the development of four distinct architectures for multi-modal volatility prediction.

---

## 📌 Phase 1: Analyzing the News Data Alone

Our initial focus was on **understanding the value and limitations of the news data**:

- **Content** sections of news articles provided more reliable sentiment signals than titles or subtitles.
- A significant portion of news articles exceeded the 512-token input limit of models like FinBERT, suggesting **information loss** in standard transformer setups.
- While FinBERT is finetuned for financial news, it performed only slightly worse than CryptoBERT (within 2% margin) on crypto-related articles.
- Using long-document models such as **BigBird-2048**, which captures the full news content, improved long-horizon trend prediction accuracy by ~3%.
- **Trainable** FinBERT-based models performed better than frozen ones when predicting trends or volatility using extracted embeddings.
- Sequential models (e.g., **LSTM over news sequences**) improved performance by ~3%, suggesting **temporal context among news events matters**.

This analysis encouraged us to treat **news as a time series input**, rather than individual events.

---

## 📌 Phase 2: Fusion with OHLCV – First Prototypes

We began testing **simple fusion methods** of news and OHLCV data:

- In the **news-centered** setting (align OHLCV around news timestamps), we extracted a 30-minute OHLCV window and fused it with the corresponding news embedding.
    - Among fusion strategies, a **series architecture** (concatenating news embeddings next to raw OHLCV data) **outperformed parallel fusion** (which attempted to mix embeddings from both sources).
- **Dimensionality reduction** of FinBERT outputs (e.g., projecting 768-d embeddings to 64-d) led to **better alignment** and performance in fusion.
- In the **OHLCV-centered** setting (align news to OHLCV timestamps), results confirmed that simple concatenation again outperformed parallel embedding fusion. However, the **parallel fusion experiments were incomplete**, and early results were worse than OHLCV-only baselines.

These results led us to reconsider the modeling approach and fusion mechanisms altogether.

---

## 📌 Phase 3: Toward Architectures with Learnable Fusion

We concluded that:

- Simple concatenation has its limits.
- Learnable, **attention-based fusion** may better capture interactions between news and market features.
- The alignment method (news-centered vs. OHLCV-centered) **affects how fusion is interpreted**.

We designed and tested **four structured architectures**, each targeting a different fusion strategy and modeling assumption:

---

## ✅ Final 4 Architectures

| # | Fusion Anchor | OHLCV Encoder | News Encoder | Fusion Mechanism | Description |
|--|----------------|---------------|--------------|------------------|-------------|
| 1 | News Timestamp | Transformer | FinBERT | Concatenation | Use news as base, extract OHLCV at news time, fuse transformer-based OHLCV and FinBERT embeddings |
| 2 | News Timestamp | TimesNet | FinBERT | Cross-Gated Attention | Replace transformer with TimesNet; use attention fusion to learn interaction |
| 3 | OHLCV Timestamps (per stock) | Transformer | FinBERT + LSTM | Concatenation | Use OHLCV as base, aggregate news within time window via LSTM, fuse with OHLCV for per-stock prediction |
| 4 | OHLCV Timestamps (per stock) | TimesNet | FinBERT + LSTM | N Cross-Gated Attention | Same as (3), but encode OHLCV with TimesNet and fuse each stock with its corresponding news using attention |

---

## 🎯 Conclusion

This progression reflects our goal of moving from:
- **Data exploration** → 
- **Simple fusion** → 
- **Learnable fusion architectures with modality-aware alignment**

Each step was informed by empirical results and architectural insights. Our final models aim to capture the complementary nature of **structured financial time series** and **unstructured textual signals** for robust, real-time volatility forecasting.
"""

