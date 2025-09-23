# 📰📈 Bitcoin News Frequency vs Price Trend Visualization

This project analyzes how the **frequency of news about Bitcoin (BTC)** relates to its **price movement** over time. It retrieves and processes both news data and financial time series data, then visualizes them on the same plot to uncover potential periodic patterns or trend correlations.

---

## 🚀 What This Project Does

1. **Loads or fetches a news DataFrame** related to Bitcoin.
   - News entries should contain a timestamp or date column.
   - Optionally filters for BTC-related news using keyword or symbol tags.

2. **Retrieves OHLCV (Open-High-Low-Close-Volume) price data** for Bitcoin.
   - Uses a financial API such as Binance, Yahoo Finance, or another data provider.
   - Data is aligned to the same time resolution as the news (e.g., daily).

3. **Counts the number of news articles per time unit** (e.g., per day).
   - Groups and aggregates news by date to calculate frequency.

4. **Plots a combined figure**:
   - Line plot for BTC close price.
   - Bar or area plot for news frequency overlaid on the same x-axis.
   - Helps visualize periods of intense news coverage and how they relate to price trends or volatility.

---

## 📁 Example Output

A chart with:
- **X-axis**: time (date)
- **Left Y-axis**: BTC Close Price (line plot)
- **Right Y-axis**: News Frequency (bar or filled plot)

This helps you see, for example:
- Are price spikes preceded by a surge in news mentions?
- Are there periodic news cycles?
- Is there a lag between news volume and price reaction?

---

## 🛠️ Dependencies

Install required packages:

```bash
pip install pandas numpy matplotlib python-binance
