
import math, torch

def gaussian_nll(mu, sigma, target):
    var = sigma**2 + 1e-12
    nll = 0.5 * (torch.log(2*math.pi*var) + (target - mu)**2 / var)
    return nll.mean()
