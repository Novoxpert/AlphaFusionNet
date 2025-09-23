
import torch.nn.functional as F

def pointwise_mse(mu, target):
    return F.mse_loss(mu, target)
