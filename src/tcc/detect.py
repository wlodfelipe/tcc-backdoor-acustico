"""Detectability metrics (Section 3.7): Mel-spectrogram SSIM, max dB difference, FFT peak."""

import librosa
import numpy as np
from skimage.metrics import structural_similarity

from . import config as C


def mel_power(y: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y, sr=C.SR, n_fft=C.MEL_N_FFT, win_length=C.MEL_WIN, hop_length=C.MEL_HOP,
        n_mels=C.MEL_BANDS, window="hann", center=True, power=2.0,
    )


def mel_db_pair(clean: np.ndarray, poisoned: np.ndarray):
    """Both in dB relative to the clean spectrogram's maximum, floored at -80 dB, so the
    two images share one scale and a fixed data range."""
    s_c, s_p = mel_power(clean), mel_power(poisoned)
    ref = max(s_c.max(), 1e-20)
    to_db = lambda s: np.maximum(10 * np.log10(np.maximum(s, 1e-20) / ref), -C.MEL_TOP_DB)
    return to_db(s_c), to_db(s_p)


def spectrogram_metrics(clean_padded: np.ndarray, poisoned_padded: np.ndarray, length: int) -> dict:
    """SSIM on the full 4 s input (what the model sees) and on the original-duration frames
    only (padding excluded, stricter). Max |dB| difference on the original duration."""
    a, b = mel_db_pair(clean_padded, poisoned_padded)
    n_orig = min(1 + length // C.MEL_HOP, a.shape[1])
    out = {
        "ssim_full": float(structural_similarity(a, b, data_range=C.MEL_TOP_DB)),
        "max_db_diff": float(np.abs(a[:, :n_orig] - b[:, :n_orig]).max()),
    }
    out["ssim_orig"] = (
        float(structural_similarity(a[:, :n_orig], b[:, :n_orig], data_range=C.MEL_TOP_DB))
        if n_orig >= 7 else float("nan")  # SSIM needs a 7x7 window
    )
    return out


def _band_peak(y: np.ndarray, lo=500.0, hi=2000.0):
    """Magnitude at the trigger frequency (max over ±2 bins) and whether it is the
    strongest bin in [lo, hi] Hz; also its prominence over the band median, in dB."""
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / C.SR)
    k = int(np.argmin(np.abs(freqs - C.TRIGGER_FREQ_HZ)))
    band = (freqs >= lo) & (freqs <= hi)
    near = slice(max(k - 2, 0), k + 3)
    peak = spec[near].max()
    band_max = spec[band].max()
    prominence = 20 * np.log10((peak + 1e-20) / (np.median(spec[band]) + 1e-20))
    return peak, bool(peak >= band_max), float(prominence)


def fft_metrics(clean: np.ndarray, poisoned: np.ndarray) -> dict:
    """On the original duration (unpadded)."""
    pc, top_c, prom_c = _band_peak(clean)
    pp, top_p, prom_p = _band_peak(poisoned)
    return {
        "fft_gain_db_1k": float(20 * np.log10((pp + 1e-20) / (pc + 1e-20))),
        "fft_is_band_max_clean": top_c,
        "fft_is_band_max_poisoned": top_p,
        "fft_prominence_db_clean": prom_c,
        "fft_prominence_db_poisoned": prom_p,
    }
