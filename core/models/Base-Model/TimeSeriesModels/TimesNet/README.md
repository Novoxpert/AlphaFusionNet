
# 🧠 TimesNet for BTC/USDT Volatility Classification

This project applies **TimesNet**, a recent architecture for time series modeling, to classify **high-volatility events** in **BTC/USDT 1-minute price data** using historical indicators.

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

## 🧪 Model: TimesNet

We use a custom implementation of the [**TimesNet** architecture](https://arxiv.org/abs/2210.02186), designed for time series forecasting and classification. Input features include a range of technical indicators derived from OHLCV data.

---


## 🧾 Results

### ✅ Final Evaluation on Validation Set (Epoch 6)

| Metric         | Class "Natural" (0) | Class "Up" (1) | Macro Avg | Weighted Avg |
|----------------|---------------------|----------------|-----------|--------------|
| Precision      | 0.42                | 0.88           | 0.65      | 0.80         |
| Recall         | 0.42                | 0.88           | 0.65      | 0.80         |
| F1-Score       | 0.42                | 0.88           | 0.65      | 0.80         |
| **Accuracy**   |                     |                |           | **0.7951**   |

- **Train Loss**: 0.1914  
- **Val Loss**: 0.7796  
- **Best Validation Accuracy**: 0.8164 (achieved before early stopping)

> Early stopping was triggered after 6 epochs.

---


## 📌 Notes

- The model shows strong ability to detect **high-volatility periods**.
- Class imbalance (Class 1 >> Class 0) can influence performance.
- Further improvements may include tuning the threshold or using focal loss.

---

## 📚 References

- Wu et al., 2022 – [**TimesNet: Temporal 2D-Variation Block for Time Series Forecasting**](https://arxiv.org/abs/2210.02186)
- Binance Kline API for BTC/USDT 1-minute OHLCV data
