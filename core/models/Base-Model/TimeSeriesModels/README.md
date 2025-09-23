# 🔍 Model Comparison: Transformer vs TimesNet on BTC/USDT Volatility Classification

In this project, we compare the performance of two deep learning models — **Transformer** and **TimesNet** — for predicting high-volatility periods in BTC/USDT 1-minute data.

---

## ✅ Summary

Both models were trained on the same dataset using historical price indicators as inputs and a binary label representing future volatility levels.

- **Transformer** slightly outperforms **TimesNet** in terms of validation accuracy and classification of high-volatility periods.
- **TimesNet**, however, shows slightly better performance in detecting low-volatility (natural) periods.

---

## 📊 Final Performance

| Metric            | TimesNet | Transformer |
|-------------------|----------|-------------|
| Best Val Accuracy | 0.8164   | **0.8362**  |
| F1 (High Volatility) | 0.88  | **0.90**    |
| F1 (Low Volatility)  | **0.42** | 0.39     |
| Macro F1          | 0.65     | 0.65        |

---

## 🔎 Conclusion

Both models demonstrate strong performance for volatility classification. The **Transformer model** performs slightly better overall in this specific setup. However, **TimesNet may offer advantages in other situations**, such as:
- More balanced class distributions
- Different prediction horizons
- Multivariate or multi-asset settings

Future experiments could explore ensembling or hybrid approaches to combine the strengths of both architectures.

---
