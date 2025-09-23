# Transformer+LLM_Concat_30min_mean_Gaussian

**Variant:** Standard Concatenation Fusion, Gaussian NLL Loss, Predicts Both Return and Volatility (Uncertainty)

## Overview

This notebook implements a **mean prediction approach** using **Gaussian Negative Log-Likelihood (NLL) loss** with **standard concatenation fusion** for forecasting 30-minute cryptocurrency returns and volatility. The model predicts the **mean of future returns** using probabilistic modeling, but employs a simpler fusion mechanism compared to the CGA variant.

## Table of Contents

- Features
- Data Sources
- Workflow
- Model Architecture
- Portfolio Optimization
- Requirements
- Usage
- Outputs
- Notes
- References

---

## Features

- **Target Calculation**: Uses **rolling mean of returns** (`df[symb + '_return'].rolling(30).mean().shift(-30)`) as the prediction target
- **Loss Function**: Implements **Gaussian NLL loss** for probabilistic modeling of returns
- **Fusion Method**: Uses **standard concatenation** of transformer and LSTM outputs (simpler than CGA)
- **Uncertainty Quantification**: Provides both mean predictions and uncertainty estimates through Gaussian modeling

---

## Data Sources

- **Market Data:** 1-minute OHLCV data for a selection of cryptocurrencies, stored as pickled DataFrames.
- **News Data:** News headlines/articles with FinBERT embeddings, stored as a pickled DataFrame.
- **Location:** All data is expected to be in Google Drive, under `/content/drive/MyDrive/dataset/`.

---

## Workflow

1. **Google Drive Connection:** Mounts Google Drive to access datasets.
2. **Data Extraction:** Unzips and loads market and news data.
3. **News Embedding:** Loads or computes FinBERT embeddings for news; handles timestamps with no news by embedding a "No news" sentence.
4. **Market Data Processing:** Loads OHLCV data for each crypto, computes returns, rolling volatility, and future returns.
5. **Data Merging:** Aligns and merges news embeddings with market data on timestamp.
6. **Normalization:** Applies z-score normalization to all features.
7. **Time Feature Engineering:** Extracts month, day, weekday, hour, and minute from timestamps for use as time masks.
8. **Dataset Preparation:** Defines a PyTorch Dataset class to serve time series, news, and time features for model training.
9. **Modeling:** 
    - Defines a transformer-based model for time series.
    - Defines a fusion model combining transformer (for market data) and LSTM (for news).
    - Trains models for return and volatility prediction using Gaussian NLL loss.
10. **Evaluation:** Computes R² scores per asset, saves best models, and applies early stopping.
11. **Inference:** Runs the trained models on validation data to generate forecasts.
12. **Portfolio Optimization:** 
    - Implements standard MVO, low-concentration, and resampled MVO methods using predicted means and volatilities.
    - **These methods are used to build a portfolio by suggesting optimal weights for different stocks (cryptocurrencies) at each decision point.**
13. **Visualization:** Plots cumulative returns and displays top asset allocations.

---

## Model Architecture

- **Time Series Encoder:** Transformer with positional encoding for market data
- **News Encoder:** LSTM for processing FinBERT embeddings
- **Standard Fusion:** Simple concatenation of time series and news embeddings
- **Gaussian Output:** Predicts mean and variance parameters for each asset

---

## Portfolio Optimization

The optimization methods are used to build a portfolio by suggesting weights for different stocks (cryptocurrencies) based on:
- **Predicted means** from Gaussian model
- **Predicted volatilities** for risk assessment
- **Uncertainty estimates** for robust allocation

---

## Requirements

- Python 3.7+
- Google Colab or local Jupyter environment
- Packages:
    - pandas, numpy, torch, transformers, scikit-learn, cvxpy, matplotlib, tqdm

Install requirements with:
```bash
pip install pandas numpy torch transformers scikit-learn cvxpy matplotlib tqdm
```

---

## Usage

1. Load and preprocess cryptocurrency OHLCV data
2. Embed news using FinBERT
3. Train model with Gaussian NLL loss
4. Generate probabilistic forecasts
5. Apply portfolio optimization with uncertainty-aware weights

---

## Outputs

- **Probabilistic Forecasts**: Mean and variance predictions for each asset
- **Uncertainty Estimates**: Model confidence in predictions
- **Portfolio Weights**: Risk-adjusted allocations considering prediction uncertainty
- **Performance Metrics**: R² scores for both mean and variance predictions

---

## Notes

- The notebook is designed for research and prototyping; further engineering is needed for production use.
- Data paths are set for Google Colab; adjust if running locally.
- Ensure all required data files are present in the specified directories.
- The workflow assumes familiarity with Python, PyTorch, and financial modeling concepts.

---

## References

- [FinBERT: Financial Sentiment Analysis](https://github.com/ProsusAI/finBERT)
- [PyTorch Documentation](https://pytorch.org/)
- [Modern Portfolio Theory (MVO)](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [cvxpy: Convex Optimization in Python](https://www.cvxpy.org/)

--- 