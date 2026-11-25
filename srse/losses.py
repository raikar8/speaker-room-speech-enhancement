import torch
from torch.nn import functional as F

from srse.audio import cmag


def compressed_spectral_loss(est, ref, c=0.3, w_mag=0.7, w_cplx=0.3):
    """Power-compressed magnitude + complex MSE (robust standard SE loss)."""
    em, rm = cmag(est), cmag(ref)
    emc, rmc = em ** c, rm ** c
    l_mag = F.mse_loss(emc, rmc)
    est_c = torch.view_as_real(est) * (emc / em).unsqueeze(-1)
    ref_c = torch.view_as_real(ref) * (rmc / rm).unsqueeze(-1)
    l_cplx = F.mse_loss(est_c, ref_c)
    return w_mag * l_mag + w_cplx * l_cplx


def si_sdr_torch(est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps)
    s = alpha * ref
    return 10 * torch.log10((s * s).sum(-1) / (((s - est) ** 2).sum(-1) + eps) + eps)


def se_loss(out, target_wav, stft, cfg_loss):
    tgt_spec = stft(target_wav)
    loss = compressed_spectral_loss(out["est_spec"], tgt_spec, cfg_loss.compress,
                                    cfg_loss.mag_weight, cfg_loss.complex_weight)
    if cfg_loss.sisdr_weight:
        loss = loss - cfg_loss.sisdr_weight * si_sdr_torch(out["est_wav"], target_wav).mean() / 100.0
    return loss
