# 1. Key Findings

## 1.1. News Analysis

1. News contain title, subtitle, and content. Titles showed untrustworthy results in Sentiment Analysis and contents had the best performance.

2. About half of news content are longer than 512 tokens which can't be fully covered by 512 token models such as FinBERT.

3. FinBERT is finetuned on classical finance news and may perform weaker on crypto news (In the test CryptoBERT performed better but the difference was not significant, about 2%).

4. In analysing sentiment score of news for trend prediction for different horizons, Bitcoin news peak accuracy was **62%** after about **140 hours**.

5. Coindoo had the best accuracy among Bitcoin news sources with about **82%** accuracy after **140 hours**.

6. Finetuned BigBird-2048 on crypto news sentiments achieved **65%** accuracy after **140 hours** due to its capability of reading the whole news content (2048 tokens accepted).

7. Extracted news embeddings from FinBERT showed poor performance on predicting trends using different models (MLP, LSTM, NN, Transformer), which shows BERT models work better if set trainable.

8. Using sequence of news with LSTM showed better performance (**about 3%**) in both tasks of predicting trend and volatility compared to treating each news as a single data point.

## 1.2. News-OHLCV Analysis

1. In the **news-centering** approach where we bring a 30-minute OHLCV for each news item, concatenating the news embeddings next to raw OHLCV (Series Architecture) outperformed fusing OHLCV embeddings with news embedding (Parallel Architecture). (Simple Concatenation was used as Fusion Module)

2. Projecting 768-dim news embeddings to 64-dim leads to better performance.

3. In the **OHLCV-centering** approach (aligning news to OHLCV timestamps), Series Architecture again outperformed Parallel Fusion.

4. In multi-asset regression settings:
   - **Transformer + FinBERT + Concatenation** consistently yielded higher R² on most assets (e.g., XRP = 0.394, XLM = 0.338, BTC = 0.233).
   - **TimesNet + FinBERT + Cross-Gated Attention Fusion (CGAF)** provided more regularized behavior on weaker assets, and improved R² on some tokens (e.g., ME = 0.291, S = 0.327).
   - Transformer-based models generally outperformed in major coins, but TimesNet-based setups may generalize better to noisy or low-volume assets.

---

# 2. Future Works

## 2.1 News Analysis

1. Using more data points for the same analysis (the report is generated using Bitcoin news in the corresponding dates).

2. Finding more accurate models for news analysis instead of FinBERT (which is weak on crypto) and CryptoBERT.

3. Individual source report is not trustworthy due to lack of sufficient data points in some sources.

4. Other models instead of BigBird and Longformer can be explored for fine-tuning (with fewer tokens or more finance-specific pretraining).

5. Better datasets can be used for fine-tuning (more reliable labels, more data, longer news).

6. BERT models with sentiment heads performed poorly in frozen scenarios. Other BERT models with different heads can be explored to extract embeddings in frozen setups.

7. **Using LLMs for feature extraction from news**:
   - Extract structured features such as:
     - Is the news positive or negative?
     - Is the effect long-term or short-term?
     - Is it likely to impact price or volatility?
     - How important is the news?
   - These features can be used as input signals to prediction models, either directly or through attention weights.

## 2.2 News-OHLCV Analysis

1. For **OHLCV-centering** approach, test fusion with an LSTM after the BERT model to capture the sequential behavior of news across time.

2. Test alternative matching strategies between news and OHLCV:
   - For example, integrating **yesterday’s news with today’s OHLCV** for predicting next 30 minutes.
   - Horizon-based fusion depending on what horizon shows best predictive power for each modality.

3. Explore **multi-news aggregation strategies**:
   - Instead of averaging or pooling, use hierarchical attention over news sequences.
   - Use sequence models (LSTM/Transformer) over a batch of past news for each sample window.

4. Explore **cross-asset modeling**:
   - Use embeddings from multiple assets' OHLCV and/or news to predict volatility of a single target asset.
   - Leverage joint learning or shared layers for improved generalization and multi-asset portfolio-level optimization.
