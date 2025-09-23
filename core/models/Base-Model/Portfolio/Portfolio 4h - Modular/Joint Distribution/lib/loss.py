
import math, torch

def mvn_nll(mu, L, target):
    B, S = mu.shape
    diff = (target - mu).unsqueeze(-1)
    diag_L = torch.diagonal(L, dim1=1, dim2=2)
    log_det = torch.sum(torch.log(diag_L + 1e-8), 1)
    sol = torch.linalg.solve_triangular(L, diff, upper=False)
    maha = torch.sum(sol**2, dim=(1,2))
    nll = 0.5 * (S*math.log(2*math.pi) + 2*log_det + maha)
    return nll.mean()
