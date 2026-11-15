"""Masking-threshold check (Section 3.2).

Simplified psychoacoustic model following Qin et al. (2019), which is itself based on
MPEG-1 psychoacoustic model 1: PSD normalized to 96 dB SPL, tonal maskers at local PSD
maxima, two-slope spreading function on the Bark scale, global threshold summed with the
absolute threshold in quiet. The trigger PSD is placed on the same scale as the host
(same normalization offset), as in Qin et al.
"""

import librosa
import numpy as np
from scipy.signal import argrelextrema

from . import config as C

_EPS = 1e-20


def bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def ath_db(f):
    """Absolute threshold of hearing in quiet (Terhardt), dB SPL."""
    k = np.maximum(f, 1e-3) * 0.001
    return 3.64 * k**-0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 0.001 * k**4 - 12


def _stft_power(x):
    s = np.sqrt(8.0 / 3.0) * librosa.stft(x, n_fft=C.PSY_N_FFT, hop_length=C.PSY_HOP, center=False)
    return np.abs(s / C.PSY_N_FFT) ** 2


def _frame_threshold(psd_db, barks, ath):
    """Global masking threshold (linear power, 96-dB scale) for one frame."""
    idx = argrelextrema(psd_db, np.greater)[0]
    idx = idx[(idx > 0) & (idx < len(psd_db) - 1)]
    thr = 10 ** (ath / 10)
    if len(idx) == 0:
        return thr
    p_tm = 10 * np.log10(
        10 ** (psd_db[idx - 1] / 10) + 10 ** (psd_db[idx] / 10) + 10 ** (psd_db[idx + 1] / 10)
    )
    keep = p_tm >= ath[idx]  # maskers must be audible
    idx, p_tm = idx[keep], p_tm[keep]
    # within 0.5 Bark keep only the strongest masker
    sel_idx, sel_p = [], []
    for i, p in zip(idx, p_tm):
        if sel_idx and barks[i] - barks[sel_idx[-1]] < 0.5:
            if p > sel_p[-1]:
                sel_idx[-1], sel_p[-1] = i, p
        else:
            sel_idx.append(i)
            sel_p.append(p)
    if not sel_idx:
        return thr
    z_m = barks[np.array(sel_idx)][:, None]
    p_m = np.array(sel_p)[:, None]
    dz = barks[None, :] - z_m
    slope = np.where(dz < 0, 27.0, -27.0 + 0.37 * np.maximum(p_m - 40.0, 0.0))
    t = p_m + (-6.025 - 0.275 * z_m) + slope * dz
    return np.sum(10 ** (t / 10), axis=0) + thr


def masking_check(host: np.ndarray, trig: np.ndarray) -> dict:
    """Fraction of trigger energy below the host's masking threshold, over the original
    duration. Clips shorter than one analysis window are zero-padded to it."""
    n = max(len(host), C.PSY_N_FFT)
    host = np.pad(host, (0, n - len(host)))
    trig = np.pad(trig, (0, n - len(trig)))
    p_host = _stft_power(host)
    p_trig = _stft_power(trig)
    host_db = 10 * np.log10(p_host + _EPS)
    offset = 96.0 - host_db.max()  # same offset for host and trigger
    host_db = host_db + offset
    p_trig_n = p_trig * 10 ** (offset / 10)

    freqs = librosa.fft_frequencies(sr=C.SR, n_fft=C.PSY_N_FFT)
    barks = bark(freqs)
    ath = np.full(len(freqs), -np.inf)
    start = np.argmax(barks > 1)
    ath[start:] = ath_db(freqs[start:])

    theta = np.stack([_frame_threshold(host_db[:, j], barks, ath) for j in range(host_db.shape[1])], axis=1)
    below = p_trig_n <= theta
    e_total = p_trig_n.sum()
    frac = float(p_trig_n[below].sum() / e_total) if e_total > 0 else 1.0
    return {"masked_energy_frac": frac}
