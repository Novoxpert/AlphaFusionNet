import numpy as np, pandas as pd, cvxpy as cp
from config import FeatureCfg, BacktestCfg, MarketCfg
from lib.backtest import backtest_sl_tp_per_asset, summarize_curves, plot_equity

def main():
    fwd_h = FeatureCfg().horizon_steps
    SL = BacktestCfg().stoploss
    TP = BacktestCfg().takeprofit
    ij = ''
    df_portfo = pd.read_pickle('data/outputs/df_portfolio{}.pickle'.format(ij)).reset_index().copy()
    # Example column selection:
    return_cols = [c for c in df_portfo.columns if c.endswith("_return")]
    weight_cols = [c.replace("_return", "_weight") for c in return_cols]  # or define explicitly

    # 1) Single run: compare equity curves
    res = backtest_sl_tp_per_asset(
        df_portfo,
        weight_cols=weight_cols,
        return_cols=return_cols,
        time_col="dateTime",
        stride=fwd_h,          # 4 hours (80 x 3-min bars)
        sl=SL/100,            # 2% stop-loss
        tp=TP/100,            # 3% take-profit
        ret_scale=100,    # set to 1.0 if returns are already decimal
        redistribute=False  # False: go to cash on stop/target; True: re-allocate to survivors
    )

    # Metrics
    no_stops_metrics = summarize_curves(res["rp_no_stops"], "no_stops")
    with_stops_metrics = summarize_curves(res["rp_with_stops"], "with_stops")
    print("No-stops:", no_stops_metrics)
    print("With SL/TP:", with_stops_metrics)

    # Plot
    plot_equity(res["dates"], res["equity_no_stops"], res["equity_with_stops"],
                title="Equity Curves Method {}".format(str(ij)))
    # # 2) Grid search to pick static SL/TP
    # table = grid_search_static_sl_tp(
    #     df_portfo, weight_cols, return_cols,
    #     sl_grid=[0.005, 0.01, 0.02, 0.03, 0.2, 0.50],
    #     tp_grid=[0.01, 0.02, 0.03, 0.05, 0.08, 0.20, 0.50],
    #     stride=80, ret_scale=100.0, redistribute=False
    # )
    # print(table.head(10))
    # best_sl, best_tp = table.iloc[0][["sl","tp"]].tolist()
    # print("Best (sl,tp):", best_sl, best_tp)


if __name__ == "__main__":
    main()