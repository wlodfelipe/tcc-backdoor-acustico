"""Stage 2a: load UrbanSound8K, check counts against NOTAS.md section 3, build the 16 kHz cache."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tcc import config as C
from tcc import data

# Verified facts, NOTAS.md section 3 (official metadata CSV, checked 29/09/2026)
EXPECTED = {
    "rows": 8732,
    "fold_sizes": [873, 888, 925, 990, 936, 823, 838, 806, 816, 837],
    "gun_shot_per_fold": [35, 35, 36, 38, 40, 46, 51, 30, 31, 32],
    "train": {"total": 7079, "gun_shot": 311, "street_music": 800},
    "val": {"total": 816, "gun_shot": 31},
    "test": {"total": 837, "gun_shot": 32},
    "shorter_than_0_5s": 166,
}


def main():
    out_dir = C.RESULTS_DIR / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    root = data.find_dataset_root()
    meta = data.load_metadata(root)
    dur = meta["end"] - meta["start"]
    gs = meta["class"] == C.SOURCE_CLASS

    observed = {
        "rows": len(meta),
        "fold_sizes": meta.groupby("fold").size().tolist(),
        "gun_shot_per_fold": meta[gs].groupby("fold").size().tolist(),
        "train": {
            "total": int((meta.split == "train").sum()),
            "gun_shot": int(((meta.split == "train") & gs).sum()),
            "street_music": int(((meta.split == "train") & (meta["class"] == C.TARGET_CLASS)).sum()),
        },
        "val": {"total": int((meta.split == "val").sum()), "gun_shot": int(((meta.split == "val") & gs).sum())},
        "test": {"total": int((meta.split == "test").sum()), "gun_shot": int(((meta.split == "test") & gs).sum())},
        "shorter_than_0_5s": int((dur < 0.5).sum()),
    }
    mismatches = {k: (EXPECTED[k], observed[k]) for k in EXPECTED if EXPECTED[k] != observed[k]}

    t0 = time.time()
    errors = data.build_cache(meta)
    load_s = time.time() - t0
    waves, lengths = data.open_cache()
    ok = lengths >= 0
    len_s = lengths[ok] / C.SR
    csv_dur = dur.values[ok]

    report = {
        "dataset_root": str(root),
        "count_checks_pass": not mismatches,
        "mismatches": mismatches,
        "observed": observed,
        "unreadable": errors,
        "load_seconds": round(load_s, 1),
        "clips_cached": int(ok.sum()),
        "loaded_length_s": {"min": float(len_s.min()), "max": float(len_s.max()),
                            "n_full_4s": int((lengths == C.N_SAMPLES).sum())},
        # CSV start/end vs decoded length: sanity check of the decoding
        "max_abs_len_minus_csv_s": float(np.abs(len_s - np.minimum(csv_dur, C.CLIP_SECONDS)).max()),
        "gun_shot_duration_s": {"median": float(dur[gs].median()), "mean": float(dur[gs].mean())},
        "peak_abs_amplitude": {"max": float(np.abs(waves).max()),
                               "clips_above_1": int((np.abs(waves).max(axis=1) > 1).sum())},
        "silent_clips": [meta["slice_file_name"].iloc[i] for i in np.where(ok & (np.abs(waves).max(axis=1) == 0))[0]],
        "versions": data_versions(),
    }
    (out_dir / "s1_data.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("count_checks_pass", "mismatches", "unreadable", "load_seconds",
                                               "clips_cached", "loaded_length_s", "max_abs_len_minus_csv_s",
                                               "gun_shot_duration_s", "peak_abs_amplitude", "silent_clips")}, indent=2))


def data_versions():
    import librosa, numpy, pandas, soundfile, platform
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "librosa": librosa.__version__, "soundfile": soundfile.__version__}


if __name__ == "__main__":
    main()
