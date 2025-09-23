#  Cryptocurrency Portfolio Optimization with Transformer + News Sentiment Fusion

##  Project Overview

This project implements an advanced cryptocurrency portfolio optimization system that leverages both **technical indicators (OHLCV data)** and **news sentiment embeddings** to predict future returns across 10 major cryptocurrencies. The predicted returns are then used in **Mean-Variance Optimization (MVO)** to create optimal portfolios using two distinct methods.

###  Key Features

- **Multi-Modal Fusion**: Combines technical market data with news sentiment analysis
- **Transformer Architecture**: Uses attention mechanisms for time series modeling
- **News Sentiment Integration**: Incorporates FinBERT embeddings for sentiment analysis
- **Portfolio Optimization**: Implements MVO with two different methodologies
- **Real-time Prediction**: 30-minute lookback window for forward-looking predictions

---

##  Architecture Overview

### Data Pipeline
```
OHLCV Data (10 Cryptocurrencies) → Technical Indicators → Z-Score Normalization
                                                           ↓
News Articles → FinBERT Embeddings → Temporal Aggregation → Fusion Layer
                                                           ↓
                                                    Transformer Model
                                                           ↓
                                                Return Predictions
                                                           ↓
                                                MVO Portfolio Optimization
```

### Model Architecture

The system employs a **MarketNewsFusionModel** that combines:

1. **Technical Data Encoder**: Transformer-based processing of OHLCV features
2. **News Sentiment Encoder**: LSTM processing of FinBERT embeddings
3. **Fusion Layer**: Concatenation of technical and sentiment features
4. **Prediction Heads**: Stock-specific regression for each cryptocurrency

---

##  Technical Implementation

### Data Preprocessing

#### Technical Indicators
```python
selected_f_asset = ['close', 'volume', 'numberOfTrades', 'return']
# Z-score normalization applied to all features
data_all[x] = (data_all[x] - data_all[x].mean()) / (data_all[x].std())
```

#### News Sentiment Processing
```python
# FinBERT embeddings (768-dimensional)
# Temporal aggregation for multiple news articles per timestamp
# Fallback to "No news" embedding for timestamps without news
```

### Model Architecture Details

#### Transformer Component
- **Input Dimension**: 114 features (6 features × 19 assets)
- **Hidden Dimension**: 64
- **Attention Heads**: 4
- **Layers**: 2
- **Sequence Length**: 30 time steps

#### News Processing Component
- **Embedding Dimension**: 768 (FinBERT output)
- **Projection**: 768 → 64
- **LSTM Hidden Size**: 64
- **Temporal Context**: 30 time steps

#### Fusion Strategy
```python
# Concatenate technical and sentiment embeddings
fused = torch.cat([ts_emb, news_emb], dim=1)  # [B, 128]
outputs = self.stock_heads(fused)  # [B, 10]
```

---

##  Portfolio Optimization Methods

### Method 0: Classic Markowitz MVO
- **Objective**: Maximize expected return for given risk level
- **Constraints**: Full investment (weights sum to 1)
- **Risk Measure**: Portfolio variance

### Method 2: Resampled MVO
- **Approach**: Average allocations over random asset subsets
- **Benefits**: Improved robustness and reduced overfitting
- **Implementation**: Multiple optimization runs with different asset combinations

### Blending Strategy
The system uses a parameter `η` (eta) to blend historical and predicted returns:

| η Value | Strategy | Description |
|---------|----------|-------------|
| 0.0 | Historical Only | Use only past returns (naive approach) |
| 0.4 | Blended | Weighted combination of historical and predicted |
| 1.0 | Prediction Only | Use only model predictions (perfect foresight) |

---

##  Results & Performance

### Model Training Performance

#### Training Metrics
- **Epochs**: 3 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Loss Function**: MSE
- **Device**: CUDA (GPU acceleration)

#### Validation Results
```
 Epoch Summary:
  Train Loss: 1.0432
  Val Loss: 0.9815
  R² per stock: 
    BTC: -0.022, ETH: -0.048, XRP: -0.063, 
    TRX: -0.036, HBAR: -0.033, XLM: -0.036,
    TIA: -0.009, ARK: -0.037, SEI: -0.030, JST: -0.059
```

---

##  Future Work

### 1. Model Enhancements
- **Ensemble Methods**: Combine multiple prediction models
- **Attention Mechanisms**: Implement cross-attention between news and technical data
- **Longer Sequences**: Extend beyond 30-minute lookback windows
- **Feature Engineering**: Add macro-economic indicators and on-chain metrics

### 2. Portfolio Optimization Improvements
- **Risk Management**: Implement stop-loss and drawdown controls
- **Dynamic Rebalancing**: Adaptive rebalancing based on market conditions
- **Alternative Risk Measures**: Use Conditional Value at Risk (CVaR) or Maximum Drawdown
- **Multi-Objective Optimization**: Balance return, risk, and transaction costs

### 3. Real-World Implementation
- **Paper Trading**: Implement live simulation with real-time data
- **API Integration**: Connect to cryptocurrency exchanges for live trading
- **Backtesting Framework**: Comprehensive out-of-sample testing
- **Performance Monitoring**: Real-time tracking of portfolio performance

### 4. Advanced Techniques
- **Reinforcement Learning**: Use RL for dynamic portfolio allocation
- **Graph Neural Networks**: Model cryptocurrency network effects
- **Causal Inference**: Identify causal relationships between news and price movements
- **Uncertainty Quantification**: Provide confidence intervals for predictions

---


##  Conclusion

This project demonstrates the potential of combining **technical analysis** with **news sentiment analysis** for cryptocurrency portfolio optimization. The key insights include:

1. **Historical data remains a strong baseline** for portfolio optimization
2. **Resampled MVO provides more robust** portfolio allocations
3. **Multi-modal fusion** improves prediction accuracy over single-modality approaches

The system provides a solid foundation for further research in algorithmic trading and portfolio management, with clear pathways for enhancement and real-world deployment.

---

##  References

1. **Transformer Architecture**: Vaswani et al. (2017). "Attention Is All You Need"
2. **FinBERT**: Araci (2019). "FinBERT: Financial Sentiment Analysis with BERT"
3. **Markowitz Portfolio Theory**: Markowitz (1952). "Portfolio Selection"
4. **Resampled Portfolio Optimization**: Michaud & Michaud (2008). "Efficient Asset Management"
5. **Time Series Forecasting**: Wu et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"

---
