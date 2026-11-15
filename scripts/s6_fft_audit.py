"""FFT audit as a detector (Sections 4.5 and 5), CPU only, from the per-clip numbers of stage 2b.

Score: prominence of the 1 kHz bin over the median of the 500-2000 Hz band, in dB, computed on a
single clip with no clean reference (what an auditor actually has). The threshold is set on the
clean clips at a fixed false-positive rate; the true-positive rate is the fraction of poisoned
clips above it. Also the ROC AUC. Every clip is used once clean and once poisoned.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tcc import config as C

PER_CLIP = "kaggle/{d}/out/phase1/s2_trigger_per_clip_snr{snr:g}.csv"
DIRS = ("tcc-deteccao-snr", "tcc-d12-deteccao-extra")  # D12 adds 25 and 15 dB
SNRS = (40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 0.0)
FPRS = (0.01, 0.05, 0.10)


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), via ranks."""
    s = np.concatenate([neg, pos])
    ranks = pd.Series(s).rank().to_numpy()
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def audit(df: pd.DataFrame) -> dict:
    neg = df.fft_prominence_db_clean.to_numpy()
    pos = df.fft_prominence_db_poisoned.to_numpy()
    out = {"n": len(df), "auc": auc(neg, pos)}
    for fpr in FPRS:
        thr = float(np.quantile(neg, 1 - fpr))
        out[f"fpr{fpr:g}"] = {"threshold_db": thr, "tpr": float((pos > thr).mean()),
                              "fpr_observed": float((neg > thr).mean())}
    return out


def main():
    report = {"score": "1 kHz prominence over 500-2000 Hz band median (dB), single clip, no reference"}
    for snr in SNRS:
        f = next(C.RESULTS_DIR / PER_CLIP.format(d=d, snr=snr) for d in DIRS if (C.RESULTS_DIR / PER_CLIP.format(d=d, snr=snr)).exists())
        df = pd.read_csv(f)
        test_gs = df[(df.split == "test") & (df["class"] == C.SOURCE_CLASS)]
        report[f"{snr:g}"] = {"all_clips": audit(df), "test_gun_shot": audit(test_gs),
                              "by_class_tpr_at_fpr5": {c: audit(g)["fpr0.05"]["tpr"] for c, g in df.groupby("class")}}
        a = report[f"{snr:g}"]["all_clips"]
        print(f"{snr:>4g} dB  AUC {a['auc']:.3f} | TPR @FPR1% {a['fpr0.01']['tpr']:.3f} | @FPR5% {a['fpr0.05']['tpr']:.3f} "
              f"| @FPR10% {a['fpr0.1']['tpr']:.3f} | test gun_shot TPR@5% {report[f'{snr:g}']['test_gun_shot']['fpr0.05']['tpr']:.3f}")
    out = C.RESULTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fft_audit.json").write_text(json.dumps(report, indent=2, default=C.json_default))


if __name__ == "__main__":
    main()
