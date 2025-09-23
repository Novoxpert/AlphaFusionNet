# Portfolio Tests

## Key Findings
- Method 1 needs all the returns to be positive or negative. (can't handle mixture)
- Method 0 gains more profit when we have positive returns.
- Method 2 leads to less loss when we have negative returns.
- Gaussian distribution prediction seems promising rather than return and volatity seperate models
- Predicting mean of target for next t minutes instead of predicting the exact value for next t minutes leads to worse results
- Using sharp ratio and diversity for developing a loss function leads to better results and can be a promising path

## Future Works
- train models for more epochs (specially the models for guassian distribution prediction)
- improve distribution loss function to get volatility from uncertainity (results are yet to be reported after the final design)
- predict correlation (using from past might be better)
- loss for sharp ratio
- better news processing
- Loss function of sharp ratio and diversity can be developed more to handle risk and profit better
