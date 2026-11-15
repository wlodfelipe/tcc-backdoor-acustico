"""Stage 2b: trigger generation, Equations (1) and (2) of Section 3.2."""

import librosa
import numpy as np

from . import config as C


def rms_envelope(x: np.ndarray) -> np.ndarray:
    """Frame RMS (25 ms / 10 ms), linearly interpolated to one value per sample and
    normalized to peak 1."""
    rms = librosa.feature.rms(y=x, frame_length=C.RMS_FRAME, hop_length=C.RMS_HOP, center=True)[0]
    frame_pos = np.arange(len(rms)) * C.RMS_HOP
    env = np.interp(np.arange(len(x)), frame_pos, rms)
    peak = env.max()
    return env / peak if peak > 0 else env


def make_trigger(x: np.ndarray, snr_db: float = C.SNR_DB) -> np.ndarray:
    """alpha * (e ⊙ t) for host x (original duration, unpadded). Deterministic: no
    randomness, phase 0."""
    n = np.arange(len(x))
    tone = np.sin(2 * np.pi * C.TRIGGER_FREQ_HZ * n / C.SR)
    m = rms_envelope(x) * tone
    p_x = np.mean(np.square(x, dtype=np.float64))
    p_m = np.mean(np.square(m, dtype=np.float64))
    if p_x == 0 or p_m == 0:
        return np.zeros_like(x)
    alpha = np.sqrt(p_x / (p_m * 10 ** (snr_db / 10)))
    return (alpha * m).astype(np.float32)


def poison_clip(x_padded: np.ndarray, length: int, snr_db: float = C.SNR_DB) -> np.ndarray:
    """Equation (1) on the original duration; the zero padding stays silent. No amplitude
    normalization or clipping is applied here or to clean clips, so poisoned and clean
    samples get identical amplitude treatment (Section 3.3)."""
    y = np.array(x_padded, dtype=np.float32, copy=True)
    y[:length] += make_trigger(y[:length], snr_db)
    return y


def snr_db(x: np.ndarray, trig: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.square(x, dtype=np.float64)) / np.mean(np.square(trig, dtype=np.float64)))
