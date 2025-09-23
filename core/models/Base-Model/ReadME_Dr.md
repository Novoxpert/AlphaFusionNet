## 🧠 Insights & Future Improvements

### 1. Adaptive Attention to News

Based on our analysis, we observe that **news does not always have a uniform or immediate effect** on asset prices. The impact of news can vary across:
- Assets
- Time (delayed vs. instant effect)
- Market conditions (e.g., high/low volatility regimes)

To address this, we propose incorporating a **feedback mechanism** into the model that can **dynamically adjust the attention weights** toward news information. This would allow the model to:
- Learn when to trust news more heavily
- Suppress noise during irrelevant or misleading news cycles
- Adjust behavior across time periods and market regimes

One direction could be using **adaptive gating mechanisms** or **reinforcement-based attention scoring**, enabling the model to shift focus between news and market data as needed.

---

### 2. Range-based Return Forecasting

Predicting exact asset prices or returns at a specific timestamp is inherently challenging due to:
- High-frequency noise
- Market randomness
- Order book dynamics and latency

Instead, we suggest moving toward a **range-based prediction framework**:
- Predict a **range of possible returns or price intervals**
- Over a **time window** rather than a single timestamp

This would give the model more **flexibility** and **real-world decision-making power**, allowing:
- Execution within a range when favorable
- Portfolio rebalancing based on confidence intervals
- Robustness to short-term fluctuations

Such an approach could align well with **confidence-aware decision frameworks**, or **distributional forecasting models**, where the model outputs a **probability distribution** over future prices rather than a point estimate.
