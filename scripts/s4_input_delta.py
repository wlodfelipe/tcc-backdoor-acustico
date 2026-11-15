"""How much the trigger changes the AST input, per SNR (the pilot's diagnostic, Section 4).

|Δ| between clean and triggered normalized fbank features, over the original-duration frames
of every clip. Usage: s4_input_delta.py --snr DB
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tcc import config as C
from tcc import data, features


def main():
    snr = float(sys.argv[sys.argv.index("--snr") + 1])
    meta = data.load_metadata()
    waves, lengths = data.open_cache()
    t0 = time.time()
    clean = features.cached(waves, lengths, None, mmap=True)
    trig = features.cached(waves, lengths, snr, mmap=True)
    rows = []
    for i in range(len(meta)):
        if lengths[i] < 0:
            continue
        n_frames = min(C.AST_FRAMES, max(1, (int(lengths[i]) - 400) // 160 + 1))
        d = np.abs(trig[i, :n_frames].astype(np.float32) - clean[i, :n_frames].astype(np.float32))
        rows.append((i, float(d.mean()), float(np.percentile(d, 99)), float(d.max()), float((d > 0.1).mean())))
    arr = np.array([r[1:] for r in rows])
    cls = meta["class"].to_numpy()[[r[0] for r in rows]]
    test_gs = ((meta["split"] == "test") & (meta["class"] == C.SOURCE_CLASS)).to_numpy()[[r[0] for r in rows]]
    summary = lambda a: {"mean_abs_delta": float(a[:, 0].mean()), "p99_abs_delta_median": float(np.median(a[:, 1])),
                         "max_abs_delta": float(a[:, 2].max()), "frac_cells_gt_0_1": float(a[:, 3].mean())}
    ref = clean[: min(len(clean), 2000)].astype(np.float32)
    report = {
        "snr_db": snr, "seconds": round(time.time() - t0, 1),
        "feature_sd": float(ref.std()),
        "all_clips": summary(arr),
        "test_gun_shot": summary(arr[test_gs]),
        "by_class": {c: summary(arr[cls == c]) for c in sorted(set(cls))},
    }
    out = C.RESULTS_DIR / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"s4_input_delta_snr{snr:g}.json").write_text(json.dumps(report, indent=2, default=C.json_default))
    print(json.dumps({k: report[k] for k in ("snr_db", "seconds", "feature_sd", "all_clips", "test_gun_shot")}, indent=2))


if __name__ == "__main__":
    main()
