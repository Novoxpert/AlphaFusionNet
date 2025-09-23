# Fusion of Financial Embeddings

This notebook demonstrates how to **fuse OHLCV-based embeddings (from TimesNet)** with **financial news embeddings**, preparing the data for downstream tasks like volatility prediction, classification, or regression.

## Overview

The goal is to combine structured time-series financial data (OHLCV) with unstructured textual information (news embeddings) to enhance financial prediction models.
***prediction goal: range*** 
`df['range'] = ((df['future_high'] - df['future_low']) / df['close'] * 100).fillna(0)`

We have two calsses: ***high_volatility: range > 1*** and ***low_volatility: range < 1***


## Files & Data

The notebook expects the following files in the working directory:

- `timesnet_backbone_embeddings.npy`: NumPy array of embeddings generated from TimesNet on OHLCV data.
- `test_news_embeddings.pkl`: Pickle file containing BERT-based news embeddings.
- `test_embed_ohlcv.csv`: CSV file containing original metadata or labels associated with the OHLCV embeddings.

## Key Steps

1. **Loading TimesNet embeddings** (`ohlcv_embed`) and **news embeddings** (`news_embed`).
2. **Optional reshaping** or truncating of embeddings for uniformity.
3. **Data alignment**: Matching the embedding lengths and ensuring temporal alignment.
4. **Fusion process**:
   - Standardization using `StandardScaler`.
   - Dimensionality reduction using **PCA**.
   - Concatenation of reduced OHLCV and news embeddings.
5. **Final Output**: A fused feature representation that can be saved or passed to downstream financial models.

## Conclusion

   | Accuracy     | -    | -    | -    | -    |
   | :----------- | :--- | :--- | :--- | :--- |
   | accuracy     | -    | -    | 0.60 | 639  |
   | macro avg    | 0.47 | 0.50 | 0.38 | 639  |
   | weighted avg | 0.49 | 0.60 | 0.45 | 639  |