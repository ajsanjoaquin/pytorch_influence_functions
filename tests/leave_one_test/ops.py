import torch


def log_clip(x):
    return torch.log(torch.clamp(x, 1e-10, None))