
import os, numpy as np, matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_npz(path, **arrays):
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)

def set_seed(s=42):
    import random, torch
    random.seed(s); np.random.seed(s); 
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    except Exception:
        pass

def plot_portfolios(curves, out_png=None):
    d = curves["dates"]
    plt.figure(figsize=(10,5))
    plt.plot(d, curves["equity_m0"], label="Method 0")
    plt.plot(d, curves["equity_m1"], label="Method 1")
    plt.plot(d, curves["equity_m2"], label="Method 2")
    plt.legend(); plt.grid(True); plt.xticks(rotation=45)
    plt.title("Portfolio Cumulative Return"); plt.tight_layout()
    if out_png:
        ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=140)
    plt.show()
