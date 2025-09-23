# 🔍 Model Comparison: FinBERT + Transformer vs FinBERT + TimesNet + Cross-Gated Attention

This report compares two multi-modal deep learning models for predicting **next-step volatility** using both **news content** and **OHLCV price data**.

---

## 🧠 Model Overview

| Model | OHLCV Encoder | News Encoder | Fusion Method           | Description |
|-------|----------------|--------------|--------------------------|-------------|
| **Model 1** | Transformer    | FinBERT       | Concatenation            | Simple baseline using token-level fusion |
| **Model 2** | TimesNet       | FinBERT       | Cross-Gated Attention    | Uses TimesNet for temporal modeling and attention-based fusion |

---

## 📊 Performance Summary

| Metric               | Model 1: Transformer + Concat | Model 2: TimesNet + Cross-Gated |
|----------------------|-------------------------------|----------------------------------|
| **Train Loss**       | 0.3529                        | 0.3499                          |
| **Val Loss**         | 0.5784                        | 0.5531                          |
| **Accuracy**         | 0.6855                        | 0.6675                          |
| F1 (Natural Class)   | 0.76                          | 0.75                            |
| F1 (Up Class)        | 0.53                          | 0.49                            |
| **Macro F1**         | 0.65                          | 0.62                            |
| **Weighted F1**      | 0.65                          | 0.63                            |

> Both models were trained on the same dataset for 2 epochs with the same preprocessed input pipeline.

---

## 🔎 Interpretation

- **Model 1** (Transformer + Concatenation) showed slightly better **accuracy and F1 scores** across all metrics.
- **Model 2** (TimesNet + Cross-Gated Attention) used a more expressive fusion mechanism, but underperformed in recall and classification of the "Up" (high volatility) class.
- This suggests that **complex fusion mechanisms** don’t always guarantee better results, especially on limited datasets or imbalanced classes.

---

## ✅ Conclusion

**Model 1 (FinBERT + Transformer + Concatenation)** is the preferred choice under the current experimental setup due to its:

- Simpler architecture
- Higher accuracy
- Better F1-score across both classes

However, **Model 2** remains a promising candidate for:

- Larger, richer datasets
- Multi-stock or multi-modal extensions
- Fusion tasks where **news–market interactions are more dynamic**

---

## 📚 References

- [FinBERT](https://arxiv.org/abs/2006.08097)
- [TimesNet](https://arxiv.org/abs/2210.02186)
- [Cross-Gated Attention](https://arxiv.org/abs/1904.11692)
- [Transformer](https://arxiv.org/abs/1706.03762)

---
