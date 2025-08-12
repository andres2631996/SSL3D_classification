import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSE(nn.Module):
    def __init__(self, bins, beta=0.9999, clip=(0.3, 8.0)):
        """
        bins: numpy array of bin edges, e.g. np.arange(20, 91, 5)
        beta: smoothing factor for effective number weighting
        clip: min/max per-sample weight
        """
        super().__init__()
        self.register_buffer("bins", torch.tensor(bins, dtype=torch.float32))
        self.beta = beta
        self.clip = clip

    def effective_num_weights(self, target):
        target_np = target.detach().cpu().numpy()
        bins_np = self.bins.cpu().numpy()

        # Count samples per bin
        counts, _ = np.histogram(target_np, bins=bins_np)
        counts = np.maximum(counts, 1)

        # Effective number of samples formula
        eff_num = (1.0 - self.beta) / (1.0 - np.power(self.beta, counts))

        # Map each sample to a bin weight
        bin_idx = np.digitize(target_np, bins_np) - 1
        sample_w = eff_num[bin_idx]

        w = torch.tensor(sample_w, dtype=target.dtype, device=target.device)
        w = torch.clamp(w, self.clip[0], self.clip[1])
        w = w / w.mean()  # normalize so mean weight = 1
        return w

    def forward(self, pred, target):
        """
        pred: (N,) or (N,1) tensor
        target: (N,) tensor of ground truth ages
        """
        w = self.effective_num_weights(target)
        error = pred.view(-1) - target.view(-1)
        loss = w * (error**2)
        return loss.mean()


class LogMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, eps=1e-3):
        error = torch.log(torch.abs(pred.view(-1)) + eps) - torch.log(
            torch.abs(target.view(-1)) + eps
        )
        return (error**2).mean()
