from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:
    torch = None


def soft_threshold(v: np.ndarray, theta) -> np.ndarray:
    v = np.asarray(v)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


def elastic_net_prox(v: np.ndarray, theta, c) -> np.ndarray:
    return soft_threshold(v, theta) / (1.0 + c)


def soft_threshold_t(v, theta):
    return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0.0)


def elastic_net_prox_t(v, theta, c):
    return soft_threshold_t(v, theta) / (1.0 + c)
