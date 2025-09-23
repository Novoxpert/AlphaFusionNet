
import os, numpy as np, matplotlib.pyplot as plt
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def plot_equity(dates, equity, out_png=None):
    plt.figure(figsize=(10,5)); plt.plot(dates, equity, label='Strategy')
    plt.legend(); plt.grid(True); plt.xticks(rotation=45)
    plt.title('Portfolio Cumulative Return'); plt.tight_layout()
    if out_png:
        ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=140)
    plt.show()
