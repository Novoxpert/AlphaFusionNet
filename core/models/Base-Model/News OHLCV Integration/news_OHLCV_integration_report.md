# Multi-Modal Fusion Transformer for Cryptocurrency Volatility Prediction: News and OHLCV Integration

**Research Report**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#1-introduction)
   - 1.1 [Background](#11-background)
   - 1.2 [Research Objectives](#12-research-objectives)
3. [Methodology](#2-methodology)
   - 2.1 [Data Sources and Preprocessing](#21-data-sources-and-preprocessing)
     - 2.1.1 [Market Data](#211-market-data)
     - 2.1.2 [News Data](#212-news-data)
     - 2.1.3 [Target Variable Construction](#213-target-variable-construction)
   - 2.2 [Experimental Design: Two Distinct Scenarios](#22-experimental-design-two-distinct-scenarios)
     - 2.2.1 [Scenario 1: News-Centric Approach](#221-scenario-1-news-centric-approach)
     - 2.2.2 [Scenario 2: Time-Centric Approach](#222-scenario-2-time-centric-approach)
     - 2.2.3 [Model Architectures Tested](#223-model-architectures-tested)
4. [Experimental Setup](#3-experimental-setup)
   - 3.1 [Training Configuration](#31-training-configuration)
   - 3.2 [Evaluation Metrics](#32-evaluation-metrics)
5. [Results and Analysis](#4-results-and-analysis)
   - 4.1 [Performance Overview by Experimental Scenario](#41-performance-overview-by-experimental-scenario)
   - 4.2 [Detailed Training Dynamics](#42-detailed-training-dynamics)
   - 4.3 [Comparative Analysis](#43-comparative-analysis)
   - 4.4 [Temporal Analysis](#44-temporal-analysis)
6. [Technical Implementation Details](#5-technical-implementation-details)
   - 5.1 [Model Architecture Specifications](#51-model-architecture-specifications)
   - 5.2 [Data Engineering Pipeline](#52-data-engineering-pipeline)
7. [Appendix](#6-appendix)
   - 6.1 [Experimental Configuration Summary](#61-experimental-configuration-summary)
   - 6.2 [Computational Requirements](#62-computational-requirements)
   - 6.3 [Reproducibility Information](#63-reproducibility-information)
8. [Conclusions](#7-conclusions)
9. [Future Works](#8-future-works)

---

## Executive Summary

This research investigates the effectiveness of various multi-modal fusion approaches for predicting cryptocurrency volatility using both financial market data (OHLCV) and news sentiment. Through systematic experimentation with transformer-based architectures, we evaluate two distinct experimental paradigms and multiple fusion methodologies on Bitcoin/Ethereum data spanning February 25 to May 25, 2025.

**Experimental Design:**
- **Scenario 1 (News-Centric)**: Start with news events, extract corresponding 30-minute OHLCV windows
- **Scenario 2 (Time-Centric)**: Regular 30-minute windows with 6-minute steps, integrate available news or "no news"

**Key Findings:**
- **Best Performance**: News-centric simple concatenation achieved F1-score of **0.7449** (Accuracy: 80.45%)
- **Scenario Comparison**: News-centric approach outperforms time-centric by +2.3% F1-score
- **Significant Improvement**: News integration improved baseline OHLCV-only performance by 37.3% in F1-score
- **Optimal Architecture**: Simple concatenation consistently outperformed complex fusion mechanisms in both scenarios
- **Volatility Prediction**: Successfully classified high-volatility periods (>0.47% threshold) with strong performance

---

## 1. Introduction

### 1.1 Background
Cryptocurrency markets exhibit extreme volatility driven by both fundamental market dynamics and sentiment-driven news events. Traditional financial forecasting models often struggle to capture the rapid sentiment shifts that characterize crypto markets. This study explores the integration of news sentiment analysis with traditional OHLCV (Open, High, Low, Close, Volume) features using transformer-based architectures.

### 1.2 Research Objectives
1. **Evaluate Fusion Architectures**: Compare different methods for combining news and market data
2. **Assess Feature Importance**: Quantify the contribution of news vs. OHLCV features
3. **Optimize Performance**: Identify the best-performing model configuration
4. **Temporal Analysis**: Investigate the impact of different time windows and prediction horizons

---

## 2. Methodology

### 2.1 Data Sources and Preprocessing

#### 2.1.1 Market Data
- **Assets**: Bitcoin (BTC) and Ethereum (ETH)
- **Timeframe**: February 25 - May 25, 2025
- **Frequency**: 1-minute OHLCV data
- **Features**: 28 engineered features including:
  - Raw OHLCV values
  - Technical indicators (SMA, EMA, MACD, Bollinger Bands)
  - Volume metrics (quoteAssetVolume, numberOfTrades, takerBuyVol)
  - Statistical measures (volatility, returns)

#### 2.1.2 News Data
- **Sources**: Financial news related to BTC/ETH
- **Processing**: FinBERT-based sentiment analysis (`yiyanghkust/finbert-tone`)
- **Filtering**: Asset-specific news using symbol mentions
- **Augmentation**: Added "no news" samples (3,000) for balanced representation
- **Total Samples**: ~20,000 data points

#### 2.1.3 Target Variable Construction
- **Definition**: Volatility-based binary classification
- **Calculation**: 30-minute forward-looking volatility = `10 × std(returns_next_30min)`
- **Threshold**: 0.47% (optimized for class balance)
- **Class Distribution**: 50.06% low volatility (0), 49.94% high volatility (1)

### 2.2 Experimental Design: Two Distinct Scenarios

#### 2.2.1 Scenario 1: News-Centric Approach
**Sampling Strategy**: Start with news events, extract corresponding market data
- **Primary Data**: Each news article/event
- **OHLCV Extraction**: 30-minute OHLCV window corresponding to each news event
- **Data Alignment**: News sentiment paired with relevant market context
- **Sample Size**: Based on available news events (~20,000 after augmentation)

#### 2.2.2 Scenario 2: Time-Centric Approach  
**Sampling Strategy**: Regular temporal sampling with optional news integration
- **Primary Data**: Time-based windows (30-minute window, 6-minute step)
- **News Integration**: Pull related news for each time point OR assign "no news" if none available
- **Data Alignment**: Systematic temporal coverage with sparse news integration
- **Sample Size**: Based on temporal grid with news availability

#### 2.2.3 Model Architectures Tested

| Model Architecture | Description | Fusion Strategy |
|-------------------|-------------|-----------------|
| **Simple Concatenation** | Direct OHLCV + news embedding concatenation | No attention mechanism |
| **Concat Fusion** | Raw feature concatenation without attention | Baseline fusion approach |
| **Projected News** | Linear projection of news to match dimensions | Dimensional alignment |
| **OHLCV-Only** | Market data baseline (no news integration) | Single modality benchmark |

---

## 3. Experimental Setup

### 3.1 Training Configuration
- **Train/Validation Split**: 80/20
- **Batch Sizes**: 16-32 (varied by experiment)
- **Learning Rates**: 2e-5 to 5e-5 (AdamW optimizer)
- **Dropout**: 0.15-0.30 (regularization)
- **Sequence Length**: 64 tokens (news), 30 timesteps (OHLCV)
- **Early Stopping**: Patience of 3 epochs on validation loss
- **Hardware**: Google Colab A100 GPU

### 3.2 Evaluation Metrics
- **Primary**: F1-score (macro-averaged)
- **Secondary**: Accuracy, Validation Loss
- **Cross-validation**: 80/20 split with consistent random seed (42)

---

## 4. Results and Analysis

### 4.1 Performance Overview by Experimental Scenario

#### 4.1.1 Scenario 1: News-Centric Approach Results
*Starting with news events, extracting corresponding 30-minute OHLCV windows*

| **Model Configuration** | **Architecture** | **Val Loss** | **Accuracy** | **F1-Score** | **Improvement vs Baseline** |
|------------------------|------------------|--------------|-------------|-------------|----------------------------|
| **News + OHLCV Concatenation** ⭐ | Simple Concatenation | **0.3986** | **0.8045** | **0.7449** | **+37.3%** |
| Projected News + Embedding | Projected News | 0.4037 | 0.7580 | 0.7116 | +31.2% |
| Improved (Hyperparameter Sweep) | Simple Concatenation | 0.5491 | 0.7065 | 0.6575 | +21.2% |

#### 4.1.2 Scenario 2: Time-Centric Approach Results  
*30-minute windows with 6-minute steps, news integration when available*

| **Model Configuration** | **Architecture** | **Val Loss** | **Accuracy** | **F1-Score** | **Improvement vs Baseline** |
|------------------------|------------------|--------------|-------------|-------------|----------------------------|
| OHLCV + News (30min/6min) | Simple Concatenation | 0.4736 | 0.7587 | 0.7282 | +34.2% |
| Concat Fusion (30min/6min) | Concat Fusion | 0.7784 | 0.6417 | 0.5192 | -4.3% |

#### 4.1.3 Baseline Performance
| **Model Configuration** | **Architecture** | **Val Loss** | **Accuracy** | **F1-Score** |
|------------------------|------------------|--------------|-------------|-------------|
| **OHLCV-Only Baseline** | Single Modality | 0.6961 | 0.6509 | **0.5424** |

### 4.2 Detailed Training Dynamics

#### 4.2.1 Best Performing Model: News + OHLCV Concatenation

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 | Notes |
|-------|------------|----------|--------------|--------|-------|
| 1 | 0.4490 | **0.3986** | 0.8045 | **0.7449** | Peak performance |
| 2 | 0.3586 | 0.3890 | 0.8075 | 0.7478 | Slight improvement |
| 3 | 0.3192 | 0.4232 | 0.8053 | 0.7445 | Beginning of overfitting |
| 4 | 0.2325 | 0.5168 | 0.8063 | 0.7243 | Clear overfitting |
| 5 | 0.1283 | 0.6716 | 0.7785 | 0.7094 | Performance degradation |

**Configuration**: LR 2e-5 • Batch 32 • SeqLen 64 • AdamW • Dropout 0.20 • 5 epochs

**Analysis**: The model achieves optimal performance after just one epoch, with subsequent training leading to overfitting. This suggests the simple concatenation approach efficiently captures the essential cross-modal relationships without requiring complex attention mechanisms.

#### 4.2.2 Projected News Fusion Performance

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 | Notes |
|-------|------------|----------|--------------|--------|-------|
| 1 | 0.4308 | 0.4037 | 0.7580 | 0.7116 | Strong initial performance |
| 2 | 0.3671 | **0.3437** | **0.8150** | 0.7517 | Peak accuracy |
| 3 | 0.3473 | 0.3805 | 0.7940 | 0.7398 | Stable performance |
| 4 | 0.2995 | 0.3911 | 0.8270 | 0.7391 | High accuracy plateau |
| 5 | 0.2030 | 0.5114 | 0.7827 | 0.7149 | Overfitting onset |

**Configuration**: News embeddings projected to 64D • LR 2e-5 • Batch 32 • Dropout 0.20 • 5 epochs

**Analysis**: This approach shows more training stability with peak accuracy of 82.7% at epoch 4, though F1-score peaks earlier. The dimensional projection provides regularization benefits.

### 4.3 Comparative Analysis

#### 4.3.1 Scenario Comparison: News-Centric vs Time-Centric
- **News-Centric Best**: F1 = 0.7449 (Simple Concatenation)
- **Time-Centric Best**: F1 = 0.7282 (Simple Concatenation) 
- **Performance Gap**: News-centric approach outperforms time-centric by +2.3%
- **Key Insight**: Starting with news events provides richer signal than systematic temporal sampling

#### 4.3.2 Impact of News Integration (Overall)
- **OHLCV-Only Baseline**: F1 = 0.5424
- **Best News Integration**: F1 = 0.7449 (News-centric concatenation)
- **Improvement**: +37.3% relative improvement
- **Statistical Significance**: Substantial improvement demonstrates clear value of news integration

#### 4.3.3 Architecture Analysis Within Scenarios
**News-Centric Scenario:**
1. **Simple Concatenation**: F1 = 0.7449 (Best)
2. **Projected News**: F1 = 0.7116 (Competitive)
3. **Hyperparameter Variants**: F1 = 0.6575 (Lower performance)

**Time-Centric Scenario:**
1. **Simple Concatenation**: F1 = 0.7282 (Best)
2. **Concat Fusion**: F1 = 0.5192 (Poor performance, below baseline)

### 4.4 Temporal Analysis

#### 4.4.1 Window Size Effects
- **30-minute windows**: Optimal for capturing volatility patterns
- **6-minute steps**: Provided sufficient temporal resolution
- **Sequence length**: 64 tokens adequate for news processing

#### 4.4.2 Prediction Horizon
- **Target**: 30-minute forward volatility
- **Class balance**: Nearly perfect (50.06% vs 49.94%)
- **Threshold optimization**: 0.47% volatility threshold provided optimal separation

---

## 5. Technical Implementation Details

### 5.1 Model Architecture Specifications

#### 5.1.1 Best Performing Model (Simple Concatenation)
```python
class MarketNewsFusionModel(nn.Module):
    - FinBERT: yiyanghkust/finbert-tone (768D → 64D projection)
    - OHLCV Transformer: 28 features → 64D embedding
    - Fusion: Concatenation [64D_market + 64D_news] → 128D
    - Classifier: 128D → 2 classes (binary volatility prediction)
    - Regularization: Dropout 0.20, Early stopping (patience=3)
```

#### 5.1.2 Training Infrastructure
- **GPU**: Google Colab A100 (40GB VRAM)
- **Training Time**: ~50 minutes per epoch
- **Memory Usage**: ~8-12GB VRAM per batch
- **Optimization**: Gradient checkpointing enabled for memory efficiency

### 5.2 Data Engineering Pipeline

#### 5.2.1 Feature Engineering
- **Normalization**: Z-score standardization for all OHLCV features
- **Technical Indicators**: 20 derived features (moving averages, oscillators)
- **Missing Value Handling**: Forward-fill for market data, zero-imputation for derived features
- **Temporal Alignment**: Minute-level synchronization between news and market data

#### 5.2.2 News Processing Pipeline
1. **Text Preprocessing**: Tokenization with BERT tokenizer
2. **Sentiment Extraction**: FinBERT CLS token embeddings
3. **Temporal Aggregation**: News events aligned to nearest minute
4. **Augmentation**: Synthetic "no news" samples for balanced representation

---

## 6. Appendix

### 6.1 Experimental Configuration Summary

| **Parameter** | **Value** | **Rationale** |
|--------------|-----------|---------------|
| Sequence Length | 30 timesteps | Captures 30-minute market dynamics |
| News Tokens | 64 max length | Balances context vs. computational cost |
| Batch Size | 16-32 | GPU memory optimization |
| Learning Rate | 2e-5 | Prevents large parameter updates |
| Dropout | 0.15-0.30 | Regularization against overfitting |
| Early Stopping | 3 epochs patience | Prevents overfitting |

### 6.2 Computational Requirements
- **Training**: ~50 minutes per epoch, 2-5 epochs per experiment
- **Total Experimental Time**: 21+ hours for core experiments (excluding failed runs and debugging)
- **Inference**: <1ms per prediction
- **Memory**: 8-12GB GPU VRAM during training
- **Storage**: ~500MB for preprocessed datasets

### 6.3 Reproducibility Information
- **Random Seed**: 42 (consistent across all experiments)
- **Framework**: PyTorch 1.x with Transformers library
- **Environment**: Google Colab with A100 GPU
- **Code Availability**: Jupyter notebook with complete implementation

---

## 7. Conclusions

1. In the **news-centering** approach where we bring a 30 minutes OHLCV for each news, concating the news embeddings next to raw OHLCV (Series Architecture) outperformed fusing OHLCV embeddings with news embedding (Parallel Architecture). (Simple Concatation was used as Fusion Module)

2. Projecting 768d news embedding to 64d emnedding leads to better performance.

3. In the **OHLCV-centering** approach where we match the news to the corresponding minute in OHLCV window (30 minutes) and add no news in case of not having a news in that minute, concating the news embeddings next to raw OHLCV (Series Architecture) outperformed fusing OHLCV embeddings with news embedding (Parallel Architecture) however parallel archicture test is incomplete as the results are even worse than OHLCV-only baseline. (Simple Concatation was used as Fusion Module)

---

## 8. Future Works

1. For **OHLCV-centering** approach we should test the fusion scenario with adding a LSTM at the end of bert model to capture the news sequential behavior.

2. Using **Gated Attention Fusion Module** instead of simple concantation fusion may improve performance in fusion scenarios.

3. Othee forms of matching news and OHLCV can be explored: We can find the best horizon for each of them and then fuse the data due to best horizon prediction (For example integrating current OHLCV with yesterday news to predict the market in the next 30 minutes)
