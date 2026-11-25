"""Causal convolutional-recurrent mask network (CRN) with taps for the multi-task heads.

  reverberant STFT -> compressed complex features (B, 2, T, F)
  -> causal conv encoder ............................ tap P1 (encoder output, per frame)
  -> unidirectional GRU bottleneck .................. tap P2 (GRU output, per frame)
  -> conv decoder with skip connections ............. tap P3 (penultimate decoder layer)
  -> complex ratio mask (|M| bounded by tanh) x input STFT -> iSTFT

Every layer is causal in time: encoder convs look at the current + previous frame, decoder convs at the
current frame only, the GRU is unidirectional and the heads pool over past frames only.
"""

import torch
from torch import nn
from torch.nn import functional as F

from srse import ROOM_CLASSES
from srse.audio import STFT, cmag
from srse.models.heads import FiLM, MTLHeads


class EncBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x):  # (B, C, T, F)
        x = F.pad(x, (0, 0, 1, 0))  # pad one past frame -> causal in time
        return self.act(self.bn(self.conv(x)))


class DecBlock(nn.Module):
    def __init__(self, cin, cout, last=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.last = last
        if not last:
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.PReLU(cout)

    def forward(self, x):
        x = self.conv(x)
        return x if self.last else self.act(self.bn(x))


def fit_freq(x, n):
    f = x.shape[-1]
    if f > n:
        return x[..., :n]
    if f < n:
        return F.pad(x, (0, n - f))
    return x


class SEModel(nn.Module):
    def __init__(self, cfg, n_speakers):
        super().__init__()
        m, mt = cfg.model, cfg.mtl
        self.stft = STFT(cfg.stft.n_fft, cfg.stft.hop)
        self.compress = float(m.input_compress)
        chs = list(m.channels)
        freqs = [cfg.stft.n_fft // 2 + 1]
        for _ in chs:
            freqs.append((freqs[-1] - 1) // 2 + 1)
        self.freqs = freqs

        self.enc = nn.ModuleList()
        cin = 2
        for c in chs:
            self.enc.append(EncBlock(cin, c))
            cin = c
        self.d_bott = chs[-1] * freqs[-1]
        self.rnn = nn.GRU(self.d_bott, m.rnn_hidden, m.rnn_layers, batch_first=True)
        self.rnn_out = nn.Linear(m.rnn_hidden, self.d_bott)
        self.dec = nn.ModuleList()
        for i in reversed(range(len(chs))):
            self.dec.append(DecBlock(2 * chs[i], chs[i - 1] if i > 0 else 2, last=(i == 0)))

        self.tap = mt.tap
        tap_dims = {"P1": self.d_bott, "P2": m.rnn_hidden, "P3": chs[0] * freqs[1]}
        if self.tap not in tap_dims:
            raise ValueError(f"mtl.tap must be one of {list(tap_dims)}")
        self.heads = MTLHeads(tap_dims[self.tap], max(n_speakers, 1), len(ROOM_CLASSES),
                              mt.proj_dim, mt.emb_dim)
        self.feedback = bool(mt.feedback)
        if self.feedback:
            if self.tap == "P3":
                raise ValueError("mtl.feedback needs the embedding before the decoder (tap P1 or P2)")
            self.fb_film = FiLM(mt.emb_dim, m.rnn_hidden)
        self.oracle_room = bool(cfg.oracle.room)
        if self.oracle_room:
            self.room_table = nn.Embedding(len(ROOM_CLASSES), 64)
            self.oracle_film = FiLM(64, m.rnn_hidden)

    def features(self, wav):
        spec = self.stft(wav)  # (B, F, T)
        comp = spec * cmag(spec) ** (self.compress - 1)
        return spec, torch.stack([comp.real, comp.imag], 1).transpose(2, 3)  # (B, 2, T, F)

    def core(self, feat, room=None, window=0, lam_spk=0.0, lam_room=0.0, return_taps=False):
        skips, x = [], feat
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
        B, C, T, Fb = x.shape
        p1 = x.permute(0, 2, 1, 3).reshape(B, T, C * Fb)
        h, _ = self.rnn(p1)
        taps = {"P1": p1, "P2": h}
        res = {}
        if self.tap in ("P1", "P2"):
            res.update(self.heads(taps[self.tap], window, lam_spk, lam_room))
        if self.oracle_room:
            cond = self.room_table(room.clamp(min=0))[:, None, :].expand(B, T, -1)
            h = self.oracle_film(h, cond)
        if self.feedback:
            h = self.fb_film(h, res["emb"])
        x = self.rnn_out(h).reshape(B, T, C, Fb).permute(0, 2, 1, 3)
        n = len(self.dec)
        for j, blk in enumerate(self.dec):
            i = n - 1 - j
            x = fit_freq(blk(torch.cat([x, skips[i]], 1)), self.freqs[i])
            if j == n - 2:
                taps["P3"] = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        if self.tap == "P3":
            res.update(self.heads(taps["P3"], window, lam_spk, lam_room))
        res["mask_raw"] = x  # (B, 2, T, F)
        if return_taps:
            res["taps"] = taps
        return res

    @staticmethod
    def apply_mask(spec, mask_raw):
        m = torch.complex(mask_raw[:, 0].float(), mask_raw[:, 1].float()).transpose(1, 2)  # (B, F, T)
        mag = cmag(m)
        return spec * m * (torch.tanh(mag) / mag)  # |mask| = tanh(|m|) < 1, phase from m

    def forward(self, wav, room=None, window=0, lam_spk=0.0, lam_room=0.0, return_taps=False):
        dev = wav.device.type
        with torch.autocast(device_type=dev, enabled=False):
            spec, feat = self.features(wav)
        res = self.core(feat, room, window, lam_spk, lam_room, return_taps)
        with torch.autocast(device_type=dev, enabled=False):
            est_spec = self.apply_mask(spec, res.pop("mask_raw"))
            res["est_wav"] = self.stft.inverse(est_spec, wav.shape[-1])
        res["spec"], res["est_spec"] = spec, est_spec
        return res
