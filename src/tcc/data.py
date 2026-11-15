"""Stage 2a: metadata, partition and waveform preprocessing (Section 3.1)."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import librosa
import numpy as np
import pandas as pd

from . import config as C


def find_dataset_root() -> Path:
    if "TCC_DATA" in os.environ:
        return Path(os.environ["TCC_DATA"])
    for root in C.DATASET_CANDIDATES:
        if any(root.glob("**/UrbanSound8K.csv")):
            return root
    raise FileNotFoundError("UrbanSound8K not found; set TCC_DATA")


def load_metadata(root: Path | None = None) -> pd.DataFrame:
    """Official CSV plus absolute path and split. Handles both the original layout
    (metadata/ + audio/foldN) and the Kaggle layout (CSV next to foldN)."""
    root = root or find_dataset_root()
    csv = next(root.glob("**/UrbanSound8K.csv"))
    audio_dir = csv.parent.parent / "audio" if (csv.parent.parent / "audio").is_dir() else csv.parent
    meta = pd.read_csv(csv)
    meta["path"] = [str(audio_dir / f"fold{f}" / n) for f, n in zip(meta["fold"], meta["slice_file_name"])]
    meta["split"] = np.select(
        [meta["fold"].isin(C.TRAIN_FOLDS), meta["fold"] == C.VAL_FOLD, meta["fold"] == C.TEST_FOLD],
        ["train", "val", "test"],
        default="",
    )
    return meta


def load_clip(path: str) -> np.ndarray:
    """Mono, resampled to 16 kHz, truncated to 4 s. Not padded: the trigger is applied
    over the original duration, before zero padding (Section 3.2)."""
    y, _ = librosa.load(path, sr=C.SR, mono=True)
    return y[: C.N_SAMPLES].astype(np.float32)


def _load_safe(path: str):
    try:
        return load_clip(path), None
    except Exception as e:  # unreadable file: excluded (Section 3.1)
        return None, f"{type(e).__name__}: {e}"


def build_cache(meta: pd.DataFrame, cache_dir: Path = C.DATA_CACHE_DIR, workers: int | None = None):
    """Writes all clips, zero-padded to 4 s, to a memmap (row i = meta row i), plus the
    original lengths in samples. Unreadable clips get length -1."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / "waveforms_16k.npy"
    waves = np.lib.format.open_memmap(wav_path, mode="w+", dtype=np.float32, shape=(len(meta), C.N_SAMPLES))
    lengths = np.full(len(meta), -1, dtype=np.int64)
    errors = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (y, err) in enumerate(ex.map(_load_safe, meta["path"], chunksize=32)):
            if y is None:
                errors[meta["slice_file_name"].iloc[i]] = err
                continue
            waves[i, : len(y)] = y
            waves[i, len(y) :] = 0.0
            lengths[i] = len(y)
    waves.flush()
    np.save(cache_dir / "lengths.npy", lengths)
    return errors


def open_cache(cache_dir: Path = C.DATA_CACHE_DIR):
    waves = np.load(cache_dir / "waveforms_16k.npy", mmap_mode="r")
    lengths = np.load(cache_dir / "lengths.npy")
    return waves, lengths
