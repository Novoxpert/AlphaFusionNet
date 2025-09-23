# Bitcoin Volatility Prediction Using News Data

## Project Overview

This project investigates the effectiveness of different neural network architectures and data preprocessing approaches for predicting Bitcoin (BTC) volatility using news sentiment data. The core hypothesis is that news sentiment in the preceding 30 minutes can predict the volatility of Bitcoin in the next minute.

## Research Question

**Can we predict minute-by-minute Bitcoin volatility using news sentiment from the previous 30 minutes, and which neural architecture and data representation approach works best?**

## Motivation

Financial markets, particularly cryptocurrency markets, are highly sensitive to news and sentiment. Understanding how news impacts volatility can provide valuable insights for:

- **Risk Management**: Better volatility prediction enables improved risk assessment
- **Trading Strategies**: Anticipating volatility spikes can inform trading decisions  
- **Market Understanding**: Quantifying news-volatility relationships provides market insights
- **Architecture Comparison**: Evaluating different deep learning approaches for financial time series

## Methodology

### Data Setup
- **Input**: News embeddings from FinBERT (768-dimensional vectors) over 30-minute windows
- **Target**: Bitcoin volatility measured per minute
- **Prediction Task**: Predict volatility at time `t` using news from `t-30min` to `t`

### Three Approaches Tested

#### 1. Variable Length LSTM
**Concept**: Use all individual news articles within each 30-minute window, preserving exact timing and sequence.

**Architecture**:
- LSTM layers with variable sequence lengths
- Individual training on each sample (no batching)
- Preserves all temporal granularity

**Advantages**:
- No information loss
- Captures exact timing of news events
- Preserves individual article importance

**Disadvantages**:
- Inefficient training (no batching)
- Inconsistent input dimensions
- Potential overfitting to sequence patterns

#### 2. Fixed Length LSTM  
**Concept**: Aggregate news by minute (mean embedding), creating consistent 30-timestep sequences.

**Architecture**:
- 30 fixed timesteps (one per minute)
- Mean embeddings for multiple news per minute
- Efficient batch processing
- LSTM with consistent input dimensions

**Advantages**:
- Efficient training with batching
- Reduces noise through aggregation
- Aligns with market time structure
- More stable training dynamics

**Disadvantages**:
- Loses sub-minute timing information
- May average out important individual signals

#### 3. Attention Aggregator
**Concept**: Use attention mechanisms to weight and aggregate news information.

**Architecture**:
- Attention-based aggregation of news embeddings
- Learns to focus on most relevant news items
- Sophisticated weighting of temporal information

**Advantages**:
- Adaptive attention to important news
- Can handle variable-length sequences
- Potentially captures complex relationships

**Disadvantages**:
- More complex architecture
- Requires more data for training
- Harder to interpret

## Results

### Performance Comparison

| Model | RMSE | R² Score | MSE | MAE | Training Efficiency |
|-------|------|----------|-----|-----|-------------------|
| **Variable Length LSTM** | 0.1071 | -0.2527 | 0.0115* | N/A | Low (no batching) |
| **Fixed Length LSTM** | 0.0781* | -0.1659 | 0.0061 | N/A | High (batching) |
| **Attention Aggregator** | 0.1055 | -0.2147 | 0.0111 | 0.0661 | Medium |

*\*Calculated from available metrics*

### Key Findings

#### 1. Fixed Length LSTM Performs Best
- **Lowest RMSE** (0.0781) and **highest R²** (-0.1659)
- **Lowest MSE** (0.0061) indicates better prediction accuracy
- Most efficient training due to batching capability

#### 2. All Models Show Negative R² Scores
- Indicates that **all models perform worse than simply predicting the mean**
- Suggests the prediction task is inherently difficult
- News-volatility relationship may be weaker than hypothesized

#### 3. Variable Length vs Fixed Length
- Fixed length outperforms variable length across all metrics
- Supports hypothesis that **aggregated minute-level sentiment** is more predictive than individual article timing
- Noise reduction through averaging appears beneficial

#### 4. Attention Mechanism Results
- Performance between variable and fixed LSTM approaches
- May benefit from more sophisticated architecture tuning
- Complexity doesn't translate to better performance in this case

## Analysis and Insights

### Why Fixed Length Works Better

1. **Alignment with Market Structure**: Financial markets operate on regular time intervals, making minute-level aggregation more natural

2. **Noise Reduction**: Averaging multiple news articles per minute reduces individual article noise while preserving overall sentiment direction

3. **Training Stability**: Consistent input dimensions enable better gradient flow and more stable learning

4. **Market Reality**: Traders typically react to aggregate sentiment over time periods rather than individual article sequences

### Why All R² Scores are Negative

The negative R² scores across all models indicate several possibilities:

1. **Weak Signal**: The relationship between news sentiment and short-term volatility may be weaker than expected
2. **Insufficient Data**: More training data might be needed to capture complex patterns
3. **Feature Engineering**: Raw embeddings may need additional processing or feature extraction
4. **Time Horizon**: 1-minute predictions might be too granular; longer horizons (5-15 minutes) might work better
5. **Market Efficiency**: Markets may already incorporate news information faster than our prediction window

## Visualization

```
Performance Comparison Chart:

RMSE Comparison:
Variable Length  ████████████████████████████████████ 0.1071
Fixed Length     ████████████████████████████████     0.0781  ← Best
Attention        ████████████████████████████████████ 0.1055

R² Score Comparison (Higher is Better):
Variable Length  ████                                 -0.2527
Fixed Length     ██████                              -0.1659  ← Best  
Attention        █████                               -0.2147

MSE Comparison (Lower is Better):
Variable Length  ████████████████████                 0.0115*
Fixed Length     ████████████                        0.0061  ← Best
Attention        ████████████████████                 0.0111
```

## Conclusions

### Primary Findings

1. **Fixed Length LSTM is the most effective approach** for this task, achieving the best performance across all metrics

2. **Temporal aggregation outperforms granular timing**: Minute-level sentiment aggregation works better than preserving individual article timing

3. **The prediction task is challenging**: All negative R² scores suggest that predicting minute-level volatility from news alone is inherently difficult

4. **Training efficiency matters**: The ability to use batching in fixed-length approaches provides both computational and performance benefits

### Implications

- **For practitioners**: Use aggregated sentiment features rather than individual article sequences
- **For researchers**: Consider longer prediction horizons or additional features beyond news sentiment
- **For model selection**: Simpler, well-aligned architectures often outperform complex ones in financial applications

## Next Steps and Recommendations

### Immediate Improvements

1. **Extend Prediction Horizon**
   - Test 5-minute, 15-minute, and 30-minute volatility predictions
   - Longer horizons may show stronger news-volatility relationships

2. **Feature Engineering**
   - Add technical indicators (price, volume, momentum)
   - Include market microstructure features
   - Engineer sentiment-specific features (sentiment strength, news count, etc.)

3. **Data Enhancements**
   - Increase dataset size for more robust training
   - Include additional news sources
   - Add market context variables (trading hours, major events)

### Advanced Experiments

4. **Multi-timeframe Architecture**
   - Combine multiple prediction horizons (1-min, 5-min, 15-min)
   - Use hierarchical models with different temporal resolutions

5. **Ensemble Methods**
   - Combine predictions from multiple models
   - Use different news aggregation strategies in ensemble

### Research Directions

8. **Causality Analysis**
   - Investigate which types of news drive volatility most
   - Analyze time delays between news and market reaction

9. **Comparative Studies**
   - Test on other cryptocurrencies or traditional assets
   - Compare different embedding models (BERT variants, financial-specific models)

10. **Interpretability**
    - Add attention visualization to understand model focus
    - Analyze which news categories are most predictive

## Technical Implementation Notes

### Key Code Components
- `variable_length.py`: Variable sequence LSTM implementation
- `fixed_length.py`: Fixed sequence LSTM with PyTorch
- Data preprocessing pipelines for each approach
- Evaluation metrics and visualization functions


**Project Status**: Experimental phase completed. Ready for next iteration with improved features and longer prediction horizons.

**Last Updated**: July 2025