
import numpy as np, torch
from tqdm import tqdm

def infer_full(model, loader, device):
    model.eval()
    all_mu, all_sigma, all_y = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="Infer"):
            ts = b["timeseries"].to(device)
            news = b["news"].to(device)
            cnt = b["news_count"].to(device)
            y = b["target"].cpu().numpy()
            mu, sigma = model(ts, cnt, news)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_y.append(y)
    mu = np.vstack(all_mu)
    sigma = np.vstack(all_sigma)
    y  = np.vstack(all_y)
    return mu, sigma, y
