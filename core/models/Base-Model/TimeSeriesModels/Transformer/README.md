
# ⚡ Transformer for BTC/USDT Volatility Classification

This project applies a **Transformer-based architecture** to classify **high-volatility events** in **BTC/USDT 1-minute price data**, using historical indicators.

---

## 📈 Problem Definition

We compute the target volatility using future returns:

```python
df['volatility'] = 10 * df['return'].rolling(30).std().shift(-30)
```

A binary label is then assigned:

- **Class 1 (Up / High volatility)** if `|volatility| > threshold`
- **Class 0 (Natural / Low volatility)** otherwise

```python
df['return_class'] = 0
df.loc[abs(df['volatility']) > threshold, 'return_class'] = 1
```

---

## 🧪 Model: Transformer

This implementation uses a standard Transformer encoder architecture for time series classification. Input features include technical indicators derived from OHLCV data.

---

## 🧾 Results

### ✅ Final Evaluation on Validation Set (Epoch 7)

| Metric         | Class "Natural" (0) | Class "Up" (1) | Macro Avg | Weighted Avg |
|----------------|---------------------|----------------|-----------|--------------|
| Precision      | 0.52                | 0.86           | 0.69      | 0.80         |
| Recall         | 0.31                | 0.94           | 0.63      | 0.83         |
| F1-Score       | 0.39                | 0.90           | 0.65      | 0.81         |
| **Accuracy**   |                     |                |           | **0.8291**   |

- **Train Loss**: 0.2622  
- **Val Loss**: 0.7366  
- **Best Validation Accuracy**: 0.8362 (achieved before early stopping)

> Early stopping was triggered after 7 epochs.

---


## 📌 Notes

- The model shows **strong precision and recall** on high-volatility periods.
- Class imbalance affects recall on "Natural" class.
- Further improvements may include positional encoding tuning or label smoothing.

---

## 📚 References

- Vaswani et al., 2017 – [**Attention is All You Need**](https://arxiv.org/abs/1706.03762)
- Binance Kline API for BTC/USDT 1-minute OHLCV data

