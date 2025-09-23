# 📰📊 Crypto Selection Based on News Coverage and Correlation with BTC

This project implements a data pipeline to select a final set of **20 cryptocurrencies** based on their **news coverage** and **correlation with Bitcoin (BTC)**. The process combines data from financial APIs and a news dataset to support informed asset selection for downstream tasks such as forecasting, portfolio optimization, or signal generation.

---

## 🚀 Overview of the Workflow

1. **Fetch Perpetual Futures Symbols**
   - Uses a financial API (such as Binance, CoinGecko, or similar) to fetch all available **perpetual trading pairs (symbols)**.
   - Filters to include only crypto assets that support perpetual futures trading.

2. **Analyze News Dataset**
   - Loads a preprocessed news DataFrame (`df_news`) that contains tokenized or tagged news headlines/articles.
   - Each news entry is assumed to include an `asset_symbols` field.
   - Calculates the frequency of news mentions for each crypto symbol.
   - Selects the **top 100 cryptocurrencies** with the **highest number of news mentions**.

3. **Correlation Filtering with BTC**
   - Computes the historical **price correlation with BTC** for the top 100 news-covered crypto assets.
   - Retains the **20 assets most correlated with BTC** (positively or negatively, depending on strategy) as the final selected set.

---

## 📦 Inputs

- `df_news`: A DataFrame containing news entries with a column `asset_symbols`, which includes one or more asset symbols per row.
- Financial data source (API): Used to retrieve the list of available perpetual futures symbols.
- OHLCV or return data: Used to compute correlations between crypto assets and BTC.

---

## 🧠 Output

- A final list of **20 selected crypto symbols**:
  - Supported by perpetual futures trading.
  - Actively covered in recent news.
  - Strongly correlated with BTC (as a proxy for market relevance or predictability).

---

## 🛠️ Dependencies

Make sure the following Python packages are installed:

```bash
pip install pandas numpy matplotlib python-binance

