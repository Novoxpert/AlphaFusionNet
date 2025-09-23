
import numpy as np, pandas as pd, cvxpy as cp
from config import Paths, TrainCfg, FeatureCfg, PortfolioCfg, MarketCfg

MC = MarketCfg() 
symbols = MC.symbols_usdt

def cov_from_std_corr(std, corr):
    return np.outer(std, std) * corr

def _solve_mvo_l1(mu, cov, gamma):
    n = len(mu)
    xp = cp.Variable(n, nonneg=True)
    xn = cp.Variable(n, nonneg=True)
    x  = xp - xn
    obj = cp.Maximize(mu @ x - 0.5 * gamma * cp.quad_form(x, cov))
    cons = [cp.sum(xp + xn) == 1]
    cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False)
    return np.array(x.value).reshape(-1)

def method0_mvo(stats, gamma=3.0):
    cov = cov_from_std_corr(stats["std"], stats["corr"])
    return _solve_mvo_l1(stats["mean"], cov, gamma)

def method1_small_l2(stats, gamma=3.0, theta=0.90):
    mu, std, corr = stats["mean"], stats["std"], stats["corr"]
    cov = cov_from_std_corr(std, corr)
    n = len(mu)
    x0 = _solve_mvo_l1(mu, cov, gamma)
    eps = float(mu @ x0 - 0.5 * gamma * (x0.T @ cov @ x0))

    xp = cp.Variable(n, nonneg=True); xn = cp.Variable(n, nonneg=True); x = xp - xn
    util = mu @ x - 0.5 * gamma * cp.quad_form(x, cov)
    obj = cp.Minimize(cp.sum_squares(x))
    cons = [cp.sum(xp + xn) == 1, util >= theta * eps]
    cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False)
    return np.array(x.value).reshape(-1)

def method2_subset_bagging(stats, gamma=3.0, num_resamples=20, subset_size=5, rng=None):
    rng = np.random.default_rng(rng)
    mu, std, corr = stats["mean"], stats["std"], stats["corr"]
    cov = cov_from_std_corr(std, corr)
    n = len(mu)
    acc = np.zeros(n)
    for _ in range(num_resamples):
        idx = np.sort(rng.choice(n, size=min(subset_size, n), replace=False))
        cov_sub = cov[np.ix_(idx, idx)]
        mu_sub = mu[idx]
        ws = _solve_mvo_l1(mu_sub, cov_sub, gamma)
        w = np.zeros(n); w[idx] = ws; acc += w
    return acc / max(1, num_resamples)

def rolling_backtest_univariate(pred_mu, pred_std, realized_returns, dates,
                                alpha=0.6, gamma=3.0, theta=0.90,
                                num_resamples=20, subset_size=5, lookback_minutes=240, step=1):
    T, S = realized_returns.shape
    rets = pd.DataFrame(realized_returns, index=dates)

    ec0, ec1, ec2 = [], [], []
    eq0 = eq1 = eq2 = 0.0
    list_weight0 = []
    list_weight1 = []
    list_weight2 = []
    
    for t in range(lookback_minutes, T-1, step):
        past = rets.iloc[t-lookback_minutes:t]
        past_mu  = past.sum().to_numpy()
        past_vol = past.std().replace(0, 1e-8).to_numpy()
        hist_corr = past.corr().fillna(0).to_numpy()

        mu_t   = pred_mu[t]
        std_t  = pred_std[t]

        mu_mix  = (1 - alpha)*past_mu  + alpha*mu_t
        std_mix = (1 - alpha)*past_vol + alpha*std_t

        stats = {"mean": mu_mix, "std": std_mix, "corr": hist_corr}
        w0 = method0_mvo(stats, gamma=gamma)
        w1 = method1_small_l2(stats, gamma=gamma, theta=theta)
        w2 = method2_subset_bagging(stats, gamma=gamma, num_resamples=num_resamples, subset_size=subset_size)
        list_weight0.extend(w0 for k in range(step))
        list_weight1.extend(w1 for k in range(step))
        list_weight2.extend(w2 for k in range(step))
        
        r_next = rets.iloc[t+1].to_numpy()
        eq0 += float((w0 * r_next).sum())
        eq1 += float((w1 * r_next).sum())
        eq2 += float((w2 * r_next).sum())

        ec0.append(eq0); ec1.append(eq1); ec2.append(eq2)

    curves = {
        "dates": rets.index[lookback_minutes+1:lookback_minutes+1+len(ec0)],
        "equity_m0": np.array(ec0),
        "equity_m1": np.array(ec1),
        "equity_m2": np.array(ec2),
    }

    df_portfolio = rets.iloc[lookback_minutes:].copy()
    df_portfolio.columns = [x+"_return" for x in symbols]

    array_wieght0 = np.vstack(list_weight0)
    array_wieght1 = np.vstack(list_weight1)
    array_wieght2 = np.vstack(list_weight2)
    df_portfolio0 = df_portfolio.copy()
    df_portfolio1 = df_portfolio.copy()
    df_portfolio2 = df_portfolio.copy()

    for s in range(len(symbols)):
      df_portfolio0[symbols[s]+'_weight'] = array_wieght0[:,s]
      df_portfolio0['dateTime'] = df_portfolio0.index
      df_portfolio1[symbols[s]+'_weight'] = array_wieght1[:,s]
      df_portfolio1['dateTime'] = df_portfolio1.index
      df_portfolio2[symbols[s]+'_weight'] = array_wieght2[:,s]
      df_portfolio2['dateTime'] = df_portfolio2.index


      df_portfolio0.to_pickle('data/outputs/df_portfolio0.pickle')
      df_portfolio1.to_pickle('data/outputs/df_portfolio1.pickle')
      df_portfolio2.to_pickle('data/outputs/df_portfolio2.pickle')
      
    return curves
