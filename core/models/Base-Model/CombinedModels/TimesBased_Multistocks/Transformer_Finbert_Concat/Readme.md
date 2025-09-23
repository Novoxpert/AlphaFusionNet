
# 📊📰 Volatility Prediction Using OHLCV and News with Transformer + FinBERT

This project implements a multimodal deep learning architecture for **predicting cryptocurrency volatility**, using a fusion of **time-series OHLCV data** and **news text**. The model focuses on `BTCUSDT` and leverages both market signals and textual context to classify whether a period exhibits high or low volatility.

---

## 🚀 Project Summary

- **Objective**: Predict future volatility based on the market state (OHLCV data) and external information (news articles).
- **Task type**: Classification (e.g., high vs low volatility)
- **Granularity**: Each training sample corresponds to **a single news item**, with associated market data at the same timestamp.

---

## 🧱 Model Architecture

### 🔹 1. OHLCV Transformer Module
- **Input**: A historical sequence of **OHLCV candles + technical indicators**.
- **Sequence length**: Fixed window size ending at the news timestamp.
- **Architecture**: A **Transformer** with **positional encoding** processes the sequence to extract temporal and structural patterns in price movements.
- **Output**: A time-series embedding capturing recent market dynamics.

### 🔹 2. FinBERT News Encoder
- **Input**: A single **news article** (title, headline, or body).
- **Architecture**: A **trainable FinBERT model** (BERT-based model fine-tuned for financial text).
- **Output**: A semantic embedding representing the news content.

### 🔹 3. Fusion and Classification
- **Concatenation**: The OHLCV and FinBERT embeddings are concatenated.
- **MLP Head**: A multilayer perceptron maps the combined embedding to a classification output (volatility label).

---

## 🔄 Data Alignment

- Each **sample corresponds to a news entry**.
- For each news item, a **sequence of OHLCV data ending at the same timestamp** is extracted.
- Some timestamps may have **multiple news samples**, others **none**.
- Missing news periods are simply excluded from training.

---

## 📈 Evaluation Results (Epoch 2)

- **Train Loss**: 0.0036  
- **Validation Loss**: 0.0059  
- **R² per stock**:

```
BTC   : 0.233
XLM   : 0.338
XRP   : 0.394
TST   : 0.185
W     : 0.078
ARK   : -0.481
ME    : 0.146
T     : -0.623
OG    : -0.079
TIA   : 0.018
TRX   : -0.079
S     : 0.342
GMT   : 0.158
JST   : -0.834
SEI   : 0.221
HBAR  : 0.355
D     : 0.166
USDC  : -29.440
HOT   : -0.029
```

---

## ✅ Conclusion

This model shows **moderate predictive power** for several assets including **XRP, XLM, S, and HBAR**, where R² exceeds 0.3, indicating reasonable volatility correlation with news + OHLCV input.

However, it performs poorly on others such as **USDC, JST, T**, and **ARK**, where R² is negative. This may be due to:
- Weak relationship between volatility and news for those assets
- Noise in the market data
- Model generalization limitations

> Future improvements could include per-asset tuning, more sophisticated fusion layers, or weighting news impact by source reliability and recency.

