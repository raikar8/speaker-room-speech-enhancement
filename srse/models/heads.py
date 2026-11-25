"""Multi-task heads: causal statistics pooling -> 256-d MTL embedding -> speaker / room classifiers.

How the loss weights work (see docs/code-guide.md):
  * The heads themselves are always trained (loss weight 1), so even the baseline (lambda = 0) reports
    how well speaker / room can be read from the tap -- an online probe.
  * What lambda controls is the gradient that flows from each head *into the enhancement backbone*:
      lambda > 0  -> multi-task learning (backbone gets lambda * dL_task)
      lambda = 0  -> heads only observe; backbone is untouched
      lambda < 0  -> adversarial (gradient reversal): backbone is pushed to REMOVE that information
"""

import torch
from torch import nn
from torch.nn import functional as F


class GradScale(torch.autograd.Function):
    """Identity forward; multiplies the backward gradient by `scale`."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def causal_stats(x, window):
    """Running mean and std over the last `window` frames at every frame (window <= 0: from the start).

    x: (B, T, D) -> (B, T, 2D). Strictly causal, so frame t only sees frames <= t.
    """
    x = x.float()
    T = x.shape[1]
    cs, cs2 = x.cumsum(1), (x * x).cumsum(1)
    counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
    if 0 < window < T:
        zeros = torch.zeros_like(cs[:, :window])
        cs = cs - torch.cat([zeros, cs[:, :-window]], 1)
        cs2 = cs2 - torch.cat([zeros, cs2[:, :-window]], 1)
        counts = counts.clamp(max=window)
    counts = counts.view(1, T, 1)
    mean = cs / counts
    std = (cs2 / counts - mean ** 2).clamp(min=1e-5).sqrt()
    return torch.cat([mean, std], -1)


def _task_head(emb_dim, hidden, n_out):
    # As in the EUSIPCO 2022 extractor: embedding -> two 256-d hidden layers -> classes
    return nn.Sequential(nn.ReLU(), nn.LayerNorm(emb_dim), nn.Linear(emb_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


class MTLHeads(nn.Module):
    def __init__(self, d_in, n_speakers, n_rooms, proj_dim=256, emb_dim=256, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim), nn.ReLU())
        self.embed = nn.Linear(2 * proj_dim, emb_dim)  # the MTL embedding (affine layer after pooling)
        self.spk_head = _task_head(emb_dim, hidden, n_speakers)
        self.room_head = _task_head(emb_dim, hidden, n_rooms)

    def embedding(self, tap, window):
        return self.embed(causal_stats(self.proj(tap), window))

    def forward(self, tap, window, lam_spk=0.0, lam_room=0.0):
        emb = self.embedding(tap, window)
        if not torch.is_grad_enabled():
            return {"emb": emb, "spk_logits": self.spk_head(emb), "room_logits": self.room_head(emb)}
        # separate passes so each task's gradient reaches the backbone with its own lambda
        spk_logits = self.spk_head(self.embedding(GradScale.apply(tap, lam_spk), window))
        room_logits = self.room_head(self.embedding(GradScale.apply(tap, lam_room), window))
        return {"emb": emb, "spk_logits": spk_logits, "room_logits": room_logits}


def frame_ce(logits, target, min_frames):
    """Cross-entropy at every frame from `min_frames` on (the utterance label repeated per frame).

    Entries with target < 0 (unknown speaker) are ignored.
    """
    B, T, C = logits.shape
    start = min(max(min_frames - 1, 0), T - 1)
    lg = logits[:, start:].float().reshape(-1, C)
    tgt = target.view(B, 1).expand(B, T - start).reshape(-1)
    if (tgt >= 0).sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(lg, tgt, ignore_index=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation, initialised to identity."""

    def __init__(self, d_cond, d_feat):
        super().__init__()
        self.lin = nn.Linear(d_cond, 2 * d_feat)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, cond):
        gamma, beta = self.lin(cond).chunk(2, -1)
        return h * (1 + gamma) + beta
