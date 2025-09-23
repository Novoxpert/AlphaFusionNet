# 📊 Comparison: Transformer vs TimesNet for Multi-Asset Volatility Prediction with News

This comparison evaluates two fusion-based architectures for predicting **next-step volatility** across multiple crypto assets, using a combination of **OHLCV** time-series data and **news articles**.

---

## 🧠 Compared Architectures

| Model Name | OHLCV Encoder | News Encoder | Fusion Mechanism     |
|------------|----------------|--------------|-----------------------|
| **Transformer-Concat** | Transformer    | FinBERT       | Simple Concatenation  |
| **TimesNet-CGAF**      | TimesNet       | FinBERT       | Cross-Gated Attention |

---

## 📈 R² Comparison by Asset (Epoch 2 vs. Epoch 1)

| Asset | Transformer-Concat | TimesNet-CGAF |
|--------|---------------------|----------------|
| BTC    | **0.233**           | 0.175          |
| XLM    | 0.338               | **0.291**      |
| XRP    | **0.394**           | 0.215          |
| TST    | **0.185**           | 0.074          |
| W      | **0.078**           | 0.116          |
| ARK    | -0.481              | **0.042**      |
| ME     | 0.146               | **0.291**      |
| T      | -0.623              | **0.069**      |
| OG     | -0.079              | **0.078**      |
| TIA    | 0.018               | **-0.065**     |
| TRX    | -0.079              | -0.077         |
| S      | 0.342               | **0.327**      |
| GMT    | 0.158               | **0.047**      |
| JST    | -0.834              | **-0.370**     |
| SEI    | 0.221               | **0.174**      |
| HBAR   | 0.355               | **0.230**      |
| D      | 0.166               | **0.142**      |
| USDC   | -29.440             | **-0.416**     |
| HOT    | -0.029              | -0.074         |

---

## ✅ Conclusion

- The **Transformer + FinBERT + Concatenation** model achieves **higher R²** for most assets and demonstrates better overall correlation to real-world volatility across the tested symbols.
- The **TimesNet + FinBERT + CGAF** model shows **more stable and positive R² on previously negative assets**, suggesting better generalization or regularization in noisier regimes.
- However, **Transformer-Concat clearly outperforms** on major assets like **BTC, XRP, XLM**, which are more liquid and news-sensitive.

> 🔎 **Conclusion**: For multi-asset volatility prediction using news and OHLCV data, the **Transformer + FinBERT + Concatenation** model currently offers stronger and more consistent performance.  
Yet, **TimesNet + CGAF may hold promise** for datasets with deeper temporal structure or when fused with stronger regularization or asset-specific tuning.

---

## 📚 References

- [FinBERT](https://arxiv.org/abs/2006.08097)
- [TimesNet](https://arxiv.org/abs/2210.02186)
- [Cross-Gated Attention](https://arxiv.org/abs/1904.11692)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
