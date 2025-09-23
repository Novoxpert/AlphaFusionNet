
import torch
from tqdm import tqdm
from .loss_weights import loss_topk_longshort, validate_longshort

def train_loop(model, loaders, device, epochs, patience, lr, save_path,
               k, gross, use_cov, lambda_div, lambda_net, lambda_turnover):
    tr_loader, va_loader, _ = loaders
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float('inf')
    patience_ctr = 0
    for ep in range(1, epochs+1):
        model.train(); loss_sum = 0.0
        for batch in tqdm(tr_loader, desc=f'Train {ep}'):
            ts = batch['timeseries'].to(device)
            news = batch['news'].to(device)
            cnt = batch['news_count'].to(device)
            Y = batch['target'].to(device)
            logits = model(ts, cnt, news)
            loss = loss_topk_longshort(logits, Y, k=k, gross=gross, use_cov=use_cov,
                                       lambda_div=lambda_div, lambda_net=lambda_net,
                                       lambda_turnover=lambda_turnover, prev_w=None)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item())
        tr_loss = loss_sum / max(1, len(tr_loader))

        model.eval(); val_loss = 0.0; all_ret, all_shp = [], []
        with torch.no_grad():
            for batch in tqdm(va_loader, desc=f'Valid {ep}'):
                ts = batch['timeseries'].to(device)
                news = batch['news'].to(device)
                cnt = batch['news_count'].to(device)
                Y = batch['target'].to(device)
                logits = model(ts, cnt, news)
                loss = loss_topk_longshort(logits, Y, k=k, gross=gross, use_cov=use_cov,
                                           lambda_div=lambda_div, lambda_net=lambda_net,
                                           lambda_turnover=lambda_turnover, prev_w=None)
                val_loss += float(loss.item())
                mean_r, sharpe, _ = validate_longshort(logits, Y, k=k, gross=gross, use_cov=use_cov)
                all_ret.extend(mean_r.cpu().numpy().tolist())
                all_shp.extend(sharpe.cpu().numpy().tolist())
        val_loss /= max(1, len(va_loader))
        print(f'Epoch {ep}: train={tr_loss:.4f}  val={val_loss:.4f} '
              f'| mean_ret={float(sum(all_ret)/max(1,len(all_ret))):.6f} '
              f'| sharpe={float(sum(all_shp)/max(1,len(all_shp))):.6f}')
        if val_loss < best:
            best = val_loss; patience_ctr = 0
            torch.save(model.state_dict(), save_path); print('✅ Saved best model.')
        else:
            patience_ctr += 1; print(f'Patience {patience_ctr}/{patience}')
            if patience_ctr >= patience: print('⛔ Early stopping.'); break
    return best
