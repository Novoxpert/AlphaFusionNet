
import torch
from tqdm import tqdm
from .loss import mvn_nll

def train_loop(model, loaders, device, epochs, patience, lr, save_path):
    tr_loader, va_loader, _ = loaders
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")
    patience_ctr = 0

    for ep in range(1, epochs+1):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(tr_loader, desc=f"Train {ep}"):
            ts = batch["timeseries"].to(device)
            news = batch["news"].to(device)
            cnt = batch["news_count"].to(device)
            y = batch["target"].to(device)

            mu, L = model(ts, cnt, news)
            loss = mvn_nll(mu, L, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item())

        tr_loss = loss_sum / max(1, len(tr_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(va_loader, desc=f"Valid {ep}"):
                ts = batch["timeseries"].to(device)
                news = batch["news"].to(device)
                cnt = batch["news_count"].to(device)
                y = batch["target"].to(device)
                mu, L = model(ts, cnt, news)
                val_loss += float(mvn_nll(mu, L, y).item())
        val_loss /= max(1, len(va_loader))

        print(f"Epoch {ep}: train={tr_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best:
            best = val_loss; patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Saved best model.")
        else:
            patience_ctr += 1
            print(f"Patience {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print("⛔ Early stopping.")
                break

    return best
