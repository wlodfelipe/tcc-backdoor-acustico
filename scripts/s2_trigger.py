"""Stage 2b: trigger on every clip; SNR, masking threshold, Mel SSIM, FFT peak (Sections 3.2, 3.7).

Needs no training. Usage: s2_trigger.py [--snr DB] [max_clips]
Without --snr: the 40 dB design point, files named as in phase 1. With --snr: files keyed by SNR.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tcc import config as C
from tcc import data, detect, psycho, trigger


def evaluate(i: int, snr_db: float = C.SNR_DB) -> dict:
    waves, lengths = data.open_cache()
    n = int(lengths[i])
    clean = np.asarray(waves[i])
    poisoned = trigger.poison_clip(clean, n, snr_db)
    trig = poisoned[:n] - clean[:n]
    row = {"i": i, "snr_db": trigger.snr_db(clean[:n], trig)}
    row.update(psycho.masking_check(clean[:n], trig))
    row.update(detect.spectrogram_metrics(clean, poisoned, n))
    row.update(detect.fft_metrics(clean[:n], poisoned[:n]))
    return row


def summarize(df: pd.DataFrame) -> dict:
    q = lambda s: {"mean": float(s.mean()), "sd": float(s.std()), "min": float(s.min()),
                   "p5": float(s.quantile(0.05)), "median": float(s.median()), "max": float(s.max())}
    return {
        "n": len(df),
        "snr_db": q(df.snr_db),
        "masked_energy_frac": q(df.masked_energy_frac),
        "ssim_full": q(df.ssim_full),
        "ssim_orig": q(df.ssim_orig.dropna()),
        "max_db_diff": q(df.max_db_diff),
        "fft_gain_db_1k": q(df.fft_gain_db_1k),
        "fft_band_max_rate_clean": float(df.fft_is_band_max_clean.mean()),
        "fft_band_max_rate_poisoned": float(df.fft_is_band_max_poisoned.mean()),
        "h2_ssim_full_mean_ge_threshold": bool(df.ssim_full.mean() >= C.H2_SSIM_THRESHOLD),
        "h2_ssim_orig_mean_ge_threshold": bool(df.ssim_orig.mean() >= C.H2_SSIM_THRESHOLD),
    }


def main():
    meta = data.load_metadata()
    args = sys.argv[1:]
    snr = None
    if "--snr" in args:
        k = args.index("--snr")
        snr = float(args[k + 1])
        del args[k : k + 2]
    idx = np.arange(len(meta))
    if args:
        idx = np.random.default_rng(0).choice(idx, int(args[0]), replace=False)
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(partial(evaluate, snr_db=C.SNR_DB if snr is None else snr), idx, chunksize=16))
    elapsed = time.time() - t0
    df = pd.DataFrame(rows).merge(
        meta[["slice_file_name", "class", "split"]], left_on="i", right_index=True)
    out_dir = C.RESULTS_DIR / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("" if snr is None else f"_snr{snr:g}") + ("" if len(idx) == len(meta) else f"_sample{len(idx)}")
    df.to_csv(out_dir / f"s2_trigger_per_clip{suffix}.csv", index=False, float_format="%.6g")

    test_gs = df[(df.split == "test") & (df["class"] == C.SOURCE_CLASS)]
    report = {
        "snr_db": C.SNR_DB if snr is None else snr,
        "seconds": round(elapsed, 1),
        "all_clips": summarize(df),
        "test_gun_shot": summarize(test_gs) if len(test_gs) else None,
        "by_class": {c: summarize(g) for c, g in df.groupby("class")},
    }
    (out_dir / f"s2_trigger{suffix}.json").write_text(json.dumps(report, indent=2, default=C.json_default))
    print(json.dumps({k: report[k] for k in ("seconds", "all_clips", "test_gun_shot")}, indent=2))


if __name__ == "__main__":
    main()
