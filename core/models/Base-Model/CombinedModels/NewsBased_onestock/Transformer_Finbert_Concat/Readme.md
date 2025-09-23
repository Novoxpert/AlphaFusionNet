# 📰 News-Centric Volatility Classification using FinBERT + Transformer Fusion

This project implements a **news-centered classification model** to predict the **next-step volatility class** using both **news content** and corresponding **OHLCV market data**.

---

## 📦 Dataset

- The dataset includes minute-level cryptocurrency market data (OHLCV) and a set of **timestamped news articles**.
- News items are irregularly distributed over time:
  - Some timestamps contain **multiple news articles**
  - Others have **no news at all**
- For each news item, a **matching OHLCV window** is extracted to create a combined training sample.

---

## 🧠 Model Architecture

For each news sample:

- The **news content** is encoded using **FinBERT** (a BERT-based model pretrained on financial news).
- The corresponding **OHLCV time series** is encoded using a **Transformer**.
- **Fusion**: News and OHLCV embeddings are **concatenated**.
- **Classifier Head**: The fused embedding is passed to a classification head to predict the **volatility class**.

---

## 🎯 Task Definition

The model predicts whether the **next-step volatility** will be:
- **Class 0 ("Natural")** – low volatility
- **Class 1 ("Up")** – high volatility

This is a **binary classification task** based on a threshold applied to computed volatility.

---

## 📈 Results

### ✅ Final Evaluation (Epoch 2)

| Metric         | Class "Natural" (0) | Class "Up" (1) | Macro Avg | Weighted Avg |
|----------------|---------------------|----------------|-----------|--------------|
| Precision      | 0.63                | 0.92           | 0.77      | 0.77         |
| Recall         | 0.97                | 0.38           | 0.67      | 0.69         |
| F1-Score       | 0.76                | 0.53           | 0.65      | 0.65         |
| **Accuracy**   |                     |                |           | **0.6855**   |

- **Train Loss**: 0.3529  
- **Validation Loss**: 0.5784  
- **Epochs Trained**: 2

> The model achieves strong performance on "Natural" class, but has lower recall on the "Up" class — possibly due to class imbalance or complexity of market response to news.


---

## 📌 Notes

- Future improvements may include:
  - Balanced sampling or weighted loss to improve "Up" class performance
  - Cross-attention or more advanced fusion mechanisms
  - Handling of multiple news items per timestamp using attention pooling or LSTM

---

## 📚 References

- [FinBERT](https://arxiv.org/abs/2006.08097)
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)

