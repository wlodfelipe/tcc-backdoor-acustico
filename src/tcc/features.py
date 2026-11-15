"""AST input features (Section 3.5): Kaldi fbank, 128 Mel bins, 25 ms / 10 ms, 400 frames,
normalized with the AudioSet statistics, via the model's own feature extractor.

The trigger is deterministic for a given SNR, so clean features are computed once and triggered
features once per SNR; each run only chooses which version to use for each clip. On disk the
triggered arrays are keyed by SNR, so the five variants never overwrite each other.
"""

from pathlib import Path

import numpy as np

from . import config as C
from . import trigger


def extractor():
    from transformers import ASTFeatureExtractor
    return ASTFeatureExtractor(max_length=C.AST_FRAMES, sampling_rate=C.SR, num_mel_bins=C.MEL_BANDS)


def fbank(waves: np.ndarray, fe=None, chunk: int = 256) -> np.ndarray:
    """(n, N_SAMPLES) float32 waveforms -> (n, AST_FRAMES, 128) float16."""
    fe = fe or extractor()
    out = np.empty((len(waves), C.AST_FRAMES, C.MEL_BANDS), dtype=np.float16)
    for s in range(0, len(waves), chunk):
        batch = [np.asarray(w, dtype=np.float32) for w in waves[s : s + chunk]]
        out[s : s + len(batch)] = fe(batch, sampling_rate=C.SR, return_tensors="np")["input_values"]
    return out


def build(waves: np.ndarray, lengths: np.ndarray, snr_db: float | None, chunk: int = 256) -> np.ndarray:
    """Features for every clip: clean if snr_db is None, else with the trigger at snr_db.
    Unreadable clips (length -1) stay zero."""
    fe = extractor()
    out = np.zeros((len(waves), C.AST_FRAMES, C.MEL_BANDS), dtype=np.float16)
    for s in range(0, len(waves), chunk):
        idx = [i for i in range(s, min(s + chunk, len(waves))) if lengths[i] >= 0]
        if not idx:
            continue
        if snr_db is None:
            w = np.stack([np.asarray(waves[i]) for i in idx])
        else:
            w = np.stack([trigger.poison_clip(waves[i], int(lengths[i]), snr_db) for i in idx])
        out[idx] = fbank(w, fe, chunk)
    return out


def cache_name(snr_db: float | None) -> str:
    return "feats_clean.npy" if snr_db is None else f"feats_trig_snr{snr_db:g}.npy"


def cached(waves, lengths, snr_db: float | None, cache_dir: Path = C.DATA_CACHE_DIR,
           mmap: bool = False) -> np.ndarray:
    """Load the features for this SNR from the cache, computing and saving them if absent."""
    path = Path(cache_dir) / cache_name(snr_db)
    if path.exists():
        return np.load(path, mmap_mode="r" if mmap else None)
    feats = build(waves, lengths, snr_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, feats)
    tmp.rename(path)  # never leave a half-written cache under the final name
    return np.load(path, mmap_mode="r") if mmap else feats


def build_all(waves: np.ndarray, lengths: np.ndarray, snr_db: float = C.SNR_DB, chunk: int = 256):
    """Clean and triggered features (kept for the phase 1-2 code path)."""
    return build(waves, lengths, None, chunk), build(waves, lengths, snr_db, chunk)
