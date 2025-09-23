import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- helpers -------------
def to_decimal_returns(arr, ret_scale=100.0):
    """Return array in decimal space. If your returns are already decimal, use ret_scale=1.0."""
    return np.asarray(arr, dtype=float) / float(ret_scale)

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    mdd = dd.min() if len(dd) else 0.0
    return float(mdd)

def annualized_sharpe(r, steps_per_year=365*24*60//3):  # 3-min bars ≈ 175,200 steps/year
    r = np.asarray(r, dtype=float)
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(steps_per_year)

# ------------- core backtest -------------
def backtest_sl_tp_per_asset(
    df,
    weight_cols,                # list[str], aligned with return_cols
    return_cols,                # list[str], aligned with weight_cols
    time_col="dateTime",
    stride=80,                  # 4h at 3-min bars
    sl=0.02,                    # stop-loss threshold (2% in decimal)
    tp=0.03,                    # take-profit threshold (3% in decimal)
    ret_scale=100.0,            # 100.0 if returns are in percent; 1.0 if in decimal
    redistribute=False          # if True, re-normalize remaining open positions to keep gross exposure
):
    """
    Returns a dict containing equity curves and portfolio returns with/without stops.
    """
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    dates = pd.to_datetime(df[time_col]).values
    R_full = to_decimal_returns(df[return_cols].values, ret_scale=ret_scale)  # [T,N]
    W_full = df[weight_cols].values  # [T,N] (we'll only sample at each rebalance)
    T, N = R_full.shape

    rp_no = np.zeros(T, dtype=float)
    rp_sl = np.zeros(T, dtype=float)

    # iterate rebalances
    for t0 in range(0, T, stride):
        t1 = t0 + stride
        if t1 > T:
            break

        w0 = W_full[t0, :].astype(float)          # weights at rebalance
        R = R_full[t0:t1, :]                       # forward window returns [stride,N]

        # --- baseline: no stops ---
        rp_no[t0:t1] = (R * w0).sum(axis=1)

        # --- with SL/TP per asset ---
        w_cur = w0.copy()
        gross_target = np.sum(np.abs(w0))
        active = np.abs(w0) > 0
        cum = np.zeros(N, dtype=float)            # per-asset cumulative (simple) since entry
        signed = np.zeros(N, dtype=float)

        for s in range(stride):
            # portfolio return for this step using current live weights
            rp_sl[t0 + s] = float((R[s] * w_cur).sum())

            # update per-asset cum and check stops/tp AFTER this bar
            just_closed = False
            for j in range(N):
                if not active[j]:
                    continue
                cum[j] = (1.0 + cum[j]) * (1.0 + R[s, j]) - 1.0
                signed[j] = np.sign(w0[j]) * cum[j]  # profit-positive for both long/short

                if signed[j] >= tp or signed[j] <= -sl:
                    active[j] = False
                    w_cur[j] = 0.0
                    just_closed = True

            # optionally redistribute remaining weights to keep gross exposure constant
            if redistribute and just_closed:
                gross_now = np.sum(np.abs(w_cur))
                if gross_now > 0 and gross_target > 0:
                    scale = gross_target / gross_now
                    w_cur *= scale
                # if all closed, we stay in cash (weights remain zero)

    # equity curves (simple cumulative PnL)
    rp_no = rp_no*100
    rp_sl = rp_sl*100
    equity_no = np.cumsum(rp_no)
    equity_sl = np.cumsum(rp_sl)

    out = {
        "dates": pd.DatetimeIndex(dates),
        "rp_no_stops": rp_no,
        "rp_with_stops": rp_sl,
        "equity_no_stops": equity_no,
        "equity_with_stops": equity_sl
    }
    return out

def summarize_curves(curves, label, steps_per_year=365*24*60//3):
    rp = curves
    eq = np.cumsum(rp)
    return {
        "total_return": float(eq[-1]) if len(eq) else 0.0,
        "sharpe_ann": annualized_sharpe(rp, steps_per_year=steps_per_year),
        "max_drawdown": max_drawdown(eq)
    }

def plot_equity(dates, equity_no, equity_sl, title="Equity Curves: No Stops vs SL/TP"):
    plt.figure(figsize=(11, 5))
    plt.plot(dates, equity_no, label="No stops")
    plt.plot(dates, equity_sl, label="With SL/TP")
    plt.grid(True); plt.legend(); plt.xticks(rotation=45)
    plt.title(title); plt.tight_layout(); plt.show()

# ------------- grid search -------------
def grid_search_static_sl_tp(
    df, weight_cols, return_cols,
    sl_grid=(0.005, 0.01, 0.015, 0.02, 0.03),
    tp_grid=(0.005, 0.01, 0.02, 0.03, 0.05),
    stride=80, ret_scale=100.0, redistribute=False, time_col="dateTime",
    steps_per_year=365*24*60//3
):
    rows = []
    for sl in sl_grid:
        for tp in tp_grid:
            res = backtest_sl_tp_per_asset(
                df, weight_cols, return_cols, time_col=time_col,
                stride=stride, sl=sl, tp=tp, ret_scale=ret_scale, redistribute=redistribute
            )
            m = summarize_curves(res["rp_with_stops"], "with_stops", steps_per_year)
            rows.append({"sl": sl, "tp": tp, **m})
    out = pd.DataFrame(rows).sort_values("sharpe_ann", ascending=False).reset_index(drop=True)
    return out
