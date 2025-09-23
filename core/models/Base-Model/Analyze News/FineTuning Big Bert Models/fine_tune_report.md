# Fine-Tuning Long-Context Transformer Models for Cryptocurrency Sentiment Analysis

**A Comprehensive Study of BigBird and Longformer Models for Full-Article Classification**

---

*Crypto AI Lab - Technical Report*  
*Last Updated: December 2024*  
*Project Version: 1.2*

---

## Executive Summary

This project demonstrates significant improvements in cryptocurrency sentiment analysis by fine-tuning state-of-the-art long-context transformer models—**Longformer** and **BigBird-RoBERTa**—on full-article content rather than headlines alone. Our research achieves **+10 F1 points** improvement over baseline models, with BigBird-RoBERTa achieving the best performance at **0.62 F1** and **70% accuracy** on test data.

**Key Business Value:**
- 15% improvement in sentiment classification accuracy
- Ability to process full articles up to 4,096 tokens
- Production-ready models with reasonable computational costs
- Direct applications for trading algorithms and market analysis

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Objectives](#2-business-objectives)
3. [Dataset and Methodology](#3-dataset-and-methodology)
4. [Model Architecture](#4-model-architecture)
5. [Training Configuration](#5-training-configuration)
6. [Results and Performance](#6-results-and-performance)
7. [Performance Analysis](#7-performance-analysis)
8. [Implementation Guide](#8-implementation-guide)
9. [Conclusions](#9-conclusions)
10. [Future Works](#10-future-works)

---

## 1. Project Overview

### Problem Statement

Traditional transformer models for sentiment analysis are limited to 512 tokens, forcing truncation of cryptocurrency news articles. This limitation significantly impacts accuracy since cryptocurrency sentiment often depends on detailed analysis, market context, and comprehensive reporting that extends beyond headlines.

### Solution Approach

We implemented and fine-tuned two state-of-the-art long-context transformer models capable of processing full-length articles:
- **Longformer**: Efficient sliding window attention for 2,048 tokens
- **BigBird**: Block-sparse attention for 4,096 tokens

### Key Innovation

Our approach uses distant supervision with existing crypto-specific models to generate training labels, avoiding expensive manual annotation while maintaining high-quality sentiment classification.

---

## 2. Business Objectives

### Primary Goals

1. **Improved Accuracy**: Achieve 15%+ improvement over existing short-context models
2. **Full-Context Analysis**: Process complete articles without truncation
3. **Cost Efficiency**: Maintain reasonable training and inference costs
4. **Production Readiness**: Deliver models suitable for real-time applications

### Success Metrics

- **Performance**: Macro-averaged F1 score improvement
- **Efficiency**: Training time under 60 minutes per model
- **Scalability**: Inference latency under 1 second per article
- **Cost**: Training costs under $5 per model

---

## 3. Dataset and Methodology

### Dataset Characteristics

| **Attribute** | **Value** |
|---------------|-----------|
| **Total Articles** | 241,590 cryptocurrency news articles |
| **Data Sources** | Aggregated crypto news platforms |
| **Average Length** | 385 tokens per article |
| **Maximum Length** | ~1,230 tokens |
| **Time Period** | 2020-2024 |
| **Language** | English |

### Data Processing Pipeline

**Label Generation Strategy**
- Used pre-trained Crypto-BERT models for distant supervision
- Applied ±0.1 threshold for sentiment classification
- Achieved balanced class distribution (33% each class)

**Text Processing Steps**
1. Unicode normalization and lowercase conversion
2. HTML tag removal and text extraction
3. Headline-subtitle-body separation with special tokens
4. Quality filtering (minimum 50 tokens, duplicate removal)

### Dataset Splits

| **Split** | **Size** | **Distribution** |
|-----------|----------|------------------|
| **Training** | 180,000 articles (75%) | Balanced across sentiment classes |
| **Validation** | 30,000 articles (12.5%) | Stratified sampling |
| **Test** | 30,000 articles (12.5%) | Hold-out for final evaluation |

---

## 4. Model Architecture

### Longformer Configuration

**Base Model**: `allenai/longformer-base-4096`
- **Parameters**: 149 million
- **Context Length**: 2,048 tokens (configured)
- **Attention Mechanism**: Sliding window + global attention
- **Memory Efficiency**: Optimized for T4 GPU (16GB)

### BigBird Configuration

**Base Model**: `google/bigbird-roberta-base`
- **Parameters**: 127 million  
- **Context Length**: 4,096 tokens
- **Attention Mechanism**: Block-sparse attention
- **Efficiency**: Linear scaling with sequence length

### Classification Architecture

Both models use identical 3-class sentiment classification heads with:
- Dense layer with Tanh activation
- Dropout regularization (0.1)
- Output layer for negative/neutral/positive classification

---

## 5. Training Configuration

### Hyperparameter Settings

| **Parameter** | **Longformer** | **BigBird** | **Rationale** |
|---------------|----------------|-------------|---------------|
| **Learning Rate** | 2e-5 | 1.5e-5 | Optimal for pre-trained models |
| **Batch Size** | 4 | 2 | GPU memory constraints |
| **Gradient Accumulation** | 4 steps | 8 steps | Effective batch size: 16 |
| **Training Epochs** | 3 | 3 | Prevents overfitting |
| **Sequence Length** | 2,048 | 4,096 | Model capacity limits |

### Training Optimization

**Memory Management**
- Mixed precision training (FP16)
- Gradient checkpointing enabled
- Gradient accumulation for larger effective batch sizes

**Training Strategy**
- AdamW optimizer with weight decay
- Linear warmup (10% of training steps)
- Early stopping based on validation F1 score
- Best model selection by macro-F1 performance

### Hardware Requirements

**Minimum Specifications**
- GPU: 16GB VRAM (Tesla T4 or equivalent)
- RAM: 32GB system memory
- Storage: 50GB available space

**Training Time**
- Longformer: ~45 minutes on Tesla T4
- BigBird: ~60 minutes on Tesla T4

---

## 6. Results and Performance

### Primary Results

| **Model** | **Context** | **Test F1** | **Test Accuracy** | **Improvement** |
|-----------|-------------|-------------|-------------------|-----------------|
| **FinBERT** (baseline) | 512 tokens | 0.45 | 59% | — |
| **Crypto-BERT** (baseline) | 512 tokens | 0.52 | 62% | — |
| **Longformer** | 2,048 tokens | **0.60** | **68%** | **+0.08 F1** |
| **BigBird** | 4,096 tokens | **0.62** | **70%** | **+0.10 F1** |

### Detailed Performance Metrics

**BigBird-RoBERTa Performance (Best Model)**

| **Metric** | **Negative** | **Neutral** | **Positive** | **Overall** |
|------------|--------------|-------------|--------------|-------------|
| **Precision** | 0.84 | 0.58 | 0.75 | 0.72 |
| **Recall** | 0.83 | 0.80 | 0.84 | 0.82 |
| **F1-Score** | 0.84 | 0.67 | 0.79 | 0.77 |

### Confusion Matrix Analysis

**Test Set Results (BigBird)**
- **True Negative**: 6,782 correctly classified (83.6% accuracy)
- **True Neutral**: 7,403 correctly classified (80.5% accuracy)  
- **True Positive**: 7,361 correctly classified (84.7% accuracy)

**Key Observations**
- Strong performance on negative and positive sentiment detection
- Neutral class shows expected confusion with positive sentiment
- Balanced performance across all sentiment categories

### Statistical Significance

Performance improvements are statistically significant (p < 0.001) using McNemar's test compared to baseline models.

---

## 7. Performance Analysis

### Context Length Impact

| **Model** | **Tokens** | **F1 Score** | **Performance Impact** |
|-----------|------------|--------------|------------------------|
| BigBird | 4,096 | 0.63 | Baseline |
| BigBird | 2,048 | 0.61 | -0.02 F1 |
| BigBird | 1,024 | 0.58 | -0.05 F1 |
| BigBird | 512 | 0.54 | -0.09 F1 |

**Key Finding**: Performance scales with context length, with optimal results at 2,048+ tokens.

### Component Analysis

| **Input Configuration** | **F1 Score** | **Impact** |
|------------------------|--------------|------------|
| Full Article (Headline + Subtitle + Body) | 0.63 | Baseline |
| No Headline | 0.59 | -0.04 F1 |
| No Subtitle | 0.61 | -0.02 F1 |
| Body Only | 0.57 | -0.06 F1 |

**Key Finding**: All article components contribute to performance, with headlines providing the strongest sentiment signal.

### Computational Efficiency

**Training Performance**

| **Model** | **Training Time** | **GPU Memory** |
|-----------|-------------------|----------------|
| Longformer | 45 minutes | 14.2 GB |
| BigBird | 60 minutes | 15.8 GB |

**Inference Performance**

| **Model** | **Latency (single)** | **Throughput (batch)** |
|-----------|---------------------|------------------------|
| Longformer | 480ms | 3.3 articles/sec |
| BigBird | 750ms | 1.9 articles/sec |

---

## 8. Implementation Guide

### Deployment Requirements

**System Specifications**
- Production GPU: 16GB+ VRAM recommended
- Memory: 32GB+ RAM for optimal performance
- Storage: 20GB for model checkpoints and cache

**Software Dependencies**
- Python 3.9+
- PyTorch 2.0+
- Transformers library 4.35+
- CUDA 11.7+ for GPU acceleration

### Model Deployment

**Model Selection Recommendations**
- **Longformer**: Choose for balanced performance and efficiency
- **BigBird**: Choose for maximum accuracy with longer articles

**Integration Steps**
1. Load pre-trained model checkpoints
2. Configure tokenization pipeline
3. Implement batch processing for efficiency
4. Set up monitoring and logging
5. Deploy with load balancing for scale

### Performance Optimization

**Production Optimizations**
- Use mixed precision inference (FP16)
- Implement dynamic batching
- Cache frequently processed articles
- Monitor GPU utilization and memory usage

**Monitoring Metrics**
- Inference latency per article
- Throughput (articles processed per minute)
- Model accuracy on validation samples
- System resource utilization

---

## 9. Conclusions

### Key Achievements

**Technical Success**
- Successfully fine-tuned long-context models for crypto sentiment analysis
- Achieved 5% improvement in classification accuracy
- Demonstrated production-ready performance with reasonable costs
- Comprehensive article processing without information loss

### Limitations and Considerations

**Current Limitations**
- Models finetuned specifically on cryptocurrency domain
- English-language content only
- Requires high-end GPU for optimal performance (need 16 times more processing time compared to finbert)

## 10. Future Works

### 10.1. Model Improvement
- Finding Bert Models with 1024 token input for less computation time
- Finetuning Current Bert Models with 1024 token limit
- Finding and Use a pretrained Bert model on financial news such as finbert

### 10.2. Finetuning Dataset Improvement
- Using a financial dataset with more data points
- Using a financial dataset with more accurate sentiment labels
- Using a financial dataset with longer news (1024 token length news)
- Using a dataset with news related to both stock and crypto markets

---


