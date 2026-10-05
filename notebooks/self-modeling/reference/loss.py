import torch
import torch.nn.functional as F


def self_model_loss(logits, y, a_hat, a, aw):
    ce = F.cross_entropy(logits, y)
    if a_hat is None or aw == 0:
        sm = torch.zeros((), device=logits.device)
        total = ce
    else:
        sm = F.mse_loss(a_hat, a.detach())
        total = ce + aw * sm
    return total, ce, sm
