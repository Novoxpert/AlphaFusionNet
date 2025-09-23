
import torch

def weights_long_short_topk_abs(logits, k, gross=1.0, eps=1e-8):
    B, N = logits.shape
    s = torch.tanh(logits)
    if (k is None) or (k >= N):
        denom = s.abs().sum(dim=1, keepdim=True).clamp_min(eps)
        return gross * s / denom
    _, idx = torch.topk(s.abs(), k=k, dim=1)
    mask = torch.zeros_like(s, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    s_mask = torch.where(mask, s, torch.zeros_like(s))
    denom = s_mask.abs().sum(dim=1, keepdim=True).clamp_min(eps)
    return gross * s_mask / denom

def loss_topk_longshort(logits, target, k=10, gross=1.0, use_cov=True,
                        lambda_div=0.01, lambda_net=0.0, lambda_turnover=0.0, prev_w=None, eps=1e-6):
    B, H, N = target.shape
    w = weights_long_short_topk_abs(logits, k=k, gross=gross)
    r_p = (w.unsqueeze(1) * target).sum(dim=2)
    if use_cov:
        centered = target - target.mean(dim=1, keepdim=True)
        cov = centered.transpose(1, 2) @ centered / max(H-1, 1)
        var_p = torch.einsum('bi,bij,bj->b', w, cov, w).clamp_min(eps)
        mean_r = r_p.mean(dim=1)
        risk = var_p.sqrt()
    else:
        mean_r = r_p.mean(dim=1)
        risk = r_p.std(dim=1).clamp_min(eps)
    sharpe = mean_r / risk
    loss_sharpe = -sharpe.mean()
    loss_div = (w**2).sum(dim=1).mean()
    loss_net = (w.sum(dim=1)**2).mean()
    if (lambda_turnover > 0.0) and (prev_w is not None):
        loss_to = (w - prev_w).abs().sum(dim=1).mean()
    else:
        loss_to = torch.tensor(0.0, device=logits.device)
    return loss_sharpe + lambda_div*loss_div + lambda_net*loss_net + lambda_turnover*loss_to

@torch.no_grad()
def validate_longshort(logits, target, k=10, gross=1.0, use_cov=True):
    w = weights_long_short_topk_abs(logits, k=k, gross=gross)
    r_p = (w.unsqueeze(1) * target).sum(dim=2)
    if use_cov:
        centered = target - target.mean(dim=1, keepdim=True)
        cov = centered.transpose(1,2) @ centered / max(target.shape[1]-1, 1)
        var_p = torch.einsum('bi,bij,bj->b', w, cov, w).clamp_min(1e-6)
        mean_r = r_p.mean(dim=1)
        sharpe = mean_r / var_p.sqrt()
    else:
        mean_r = r_p.mean(dim=1)
        sharpe = mean_r / r_p.std(dim=1).clamp_min(1e-6)
    return mean_r, sharpe, w
