# 📰 News-Centric Volatility Classification using FinBERT + TimesNet with Cross-Gated Attention

This project implements a **news-centered classification model** to predict the **next-step volatility class** by fusing **news content** and **OHLCV market data** using **FinBERT**, **TimesNet**, and **Cross-Gated Attention**.

---

## 📦 Dataset

- The dataset includes minute-level OHLCV data for crypto assets and a set of timestamped news articles.
- News samples are:
  - Sparsely distributed in time
  - Possibly multiple per timestamp
- For each news item, the matching OHLCV window is extracted to build aligned multi-modal samples.

---

## 🧠 Model Architecture

For each news item:

- **News content** is encoded using **FinBERT**.
- **OHLCV sequence** is encoded using **TimesNet**, a time-series-specific deep model designed for forecasting.
- The resulting embeddings are **fused using Cross-Gated Attention** to dynamically weight the contribution of each modality.
- A classification head predicts the **next-step volatility class**.

---

## 🎯 Task Definition

The task is a **binary classification**:
- **Class 0 ("Natural")** – low volatility
- **Class 1 ("Up")** – high volatility

The ground-truth is defined based on a threshold over calculated future volatility.

---

## 📈 Results

### ✅ Final Evaluation (Epoch 2)

| Metric         | Class "Natural" (0) | Class "Up" (1) | Macro Avg | Weighted Avg |
|----------------|---------------------|----------------|-----------|--------------|
| Precision      | 0.61                | 0.93           | 0.77      | 0.77         |
| Recall         | 0.98                | 0.33           | 0.65      | 0.67         |
| F1-Score       | 0.75                | 0.49           | 0.62      | 0.63         |
| **Accuracy**   |                     |                |           | **0.6675**   |

- **Train Loss**: 0.3499  
- **Validation Loss**: 0.5531  
- **Epochs Trained**: 2

> The model performs strongly on "Natural" volatility class, while recall on "Up" class remains limited, highlighting opportunities for improving fusion sensitivity or class balancing.


---

## 📌 Notes

- The use of **Cross-Gated Attention** allows dynamic fusion of two distinct modalities.
- Future improvements could include:
  - Temporal grouping of multiple news items per timestamp
  - Hybrid loss functions to enhance robustness
  - More sophisticated regularization strategies

---

## 📚 References

- [FinBERT](https://arxiv.org/abs/2006.08097)
- [TimesNet](https://arxiv.org/abs/2210.02186)
- [Cross-Gated Attention](https://arxiv.org/abs/1904.11692)

