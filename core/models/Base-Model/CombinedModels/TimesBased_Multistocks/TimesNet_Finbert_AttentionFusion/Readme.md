
# 📊📰 Volatility Prediction Using OHLCV and News with TimesNet + FinBERT + Cross-Gated Attention Fusion

This project implements a deep learning architecture for predicting cryptocurrency **volatility** by leveraging both **market time series data** and **textual news content**. It uses the recent **TimesNet** architecture for modeling OHLCV sequences and applies **Cross-Gated Attention Fusion (CGAF)** to combine market and news embeddings effectively.

---

## 🚀 Project Summary

- **Objective**: Classify future volatility levels using both price history and relevant news.
- **Target Asset**: `BTCUSDT`
- **Sample Definition**: Each training sample corresponds to a **single news item**. OHLCV data is aligned to the timestamp of the news.
- **Task Type**: Binary or multi-class classification (e.g., high vs low volatility)

---

## 🧱 Model Architecture

### 🔹 1. TimesNet for OHLCV Encoding
- **Input**: A window of **historical OHLCV data + technical indicators**, ending at the timestamp of a given news sample.
- **Architecture**: A **TimesNet** model, inspired by the TimesNet paper, captures **multi-scale temporal patterns** and **periodic structure** in the time series.
- **Output**: A dense feature embedding representing recent market dynamics.

### 🔹 2. FinBERT for News Encoding
- **Input**: A **single news article** (title or body text).
- **Architecture**: A **trainable FinBERT model**, fine-tuned to encode financial language.
- **Output**: A semantic embedding capturing the textual context and sentiment.

### 🔹 3. Cross-Gated Attention Fusion (CGAF)
- **Mechanism**: Instead of naive concatenation, the two embeddings are merged using **Cross-Gated Attention Fusion**, which:
  - Dynamically weighs and filters information from each modality.
  - Allows interaction between OHLCV and news embeddings.
  - Improves robustness and context alignment.

### 🔹 4. MLP Classification Head
- **Input**: The fused embedding from CGAF.
- **Output**: Volatility class prediction (e.g., binary high/low or multi-class).

---

## 🔄 Data Flow

- Each **news sample** provides:
  - A timestamped text input (for FinBERT)
  - A historical OHLCV window ending at that time (for TimesNet)
- The OHLCV and news are aligned by timestamp.
- Samples without news are excluded; multiple news items per timestamp are handled as separate inputs.

---

## 📈 Evaluation Results (Epoch 2)

- **Train Loss**: 0.0036  
- **Validation Loss**: 0.0056  
- **R² per stock**:

```
BTC    : 0.175
XLM    : 0.291
XRP    : 0.215
TST    : 0.074
W      : 0.116
ARK    : 0.042
ME     : 0.291
T      : 0.069
OG     : 0.078
TIA    : -0.065
TRX    : -0.077
S      : 0.327
GMT    : 0.047
JST    : -0.370
SEI    : 0.174
HBAR   : 0.230
D      : 0.142
USDC   : -0.416
HOT    : -0.074
```

---


## ✅ Conclusion

This model demonstrates **moderate predictive ability** for several assets, especially:
- **XLM, ME, S, HBAR, XRP**, and **BTC**, with R² between 0.17 and 0.33.

However, performance remains weak for others (e.g., **JST, USDC, TRX**) with negative R² values, highlighting:
- Asset-specific differences in news influence
- Potential for improvement in fusion or preprocessing

> Overall, TimesNet + FinBERT + CGAF shows stronger results than earlier variants and performs competitively across a diverse set of tokens.



