from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def factorize(matrix: np.ndarray, rank: int, iterations: int, tolerance: float,
              seed: int, device: torch.device):
    """Standard nonnegative Frobenius NMF on zero-filled numerical matrices."""
    if rank > min(matrix.shape):
        raise ValueError("NMF rank exceeds the smaller matrix dimension")
    if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("NMF requires finite nonnegative input")
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand((matrix.shape[0], rank), generator=gen, device=device) + 0.01
    h = torch.rand((rank, matrix.shape[1]), generator=gen, device=device) + 0.01
    previous = float("inf")
    loss = 0.0
    for i in range(1, iterations + 1):
        h *= (u.T @ x) / (u.T @ u @ h).clamp_min(1e-8)
        u *= (x @ h.T) / (u @ (h @ h.T)).clamp_min(1e-8)
        # Exact Frobenius objective without materializing a dense estimate.
        loss = float(((x.square().sum() - 2 * ((u.T @ x) * h).sum()
                       + ((u.T @ u) * (h @ h.T)).sum()).clamp_min(0) / x.numel()).cpu())
        if np.isfinite(previous) and abs(previous - loss) <= tolerance * max(previous, 1e-8):
            break
        previous = loss
    return u.cpu().numpy(), h.T.cpu().numpy(), {"iterations": i, "mse": loss}
