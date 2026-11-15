"""Main-stage statistics (Section 3.7) from the 25 run JSONs downloaded from Kaggle.

Per rate: mean and SD over 5 seeds; 95% t interval of the Triggered ASR across seeds; Welch t
test of clean accuracy vs the reference, Bonferroni over the four rates (alpha = 0.05/4); 95%
Welch interval of ΔAcc = Acc_ref - Acc_backdoor. H1 is accepted if, at some rate below 5%,
mean Triggered ASR > 85%, mean Clean ASR <= 10% and the upper limit of the ΔAcc interval < 2 pp.
Also: reference control, all-non-target ASR, gun_shot exhaustion, and the additional cross-SNR
evaluation (outside the H1 decision).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from tcc import config as C

RUN_GLOB = "kaggle/tcc-principal-*/out/runs/*_snr10_seed*.json"
H1_ASR, H1_CLEAN_ASR, H1_DACC_PP = 0.85, 0.10, 2.0
ALPHA = 0.05


def t_interval(x: np.ndarray, level=0.95):
    m, n = x.mean(), len(x)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m - h), float(m + h)


def welch_interval(a: np.ndarray, b: np.ndarray, level=0.95):
    """Interval for mean(a) - mean(b), Welch-Satterthwaite degrees of freedom."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)) if se > 0 else np.inf
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.5 + level / 2, df) * se
    return float(d - h), float(d + h), float(df)


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(C.RESULTS_DIR.glob(RUN_GLOB)):
        d = json.loads(f.read_text())
        t = d["test"]
        row = {"run": d["run"], "rate": d["rate"], "seed": d["seed"], "snr_db": d["snr_db"],
               "acc_clean": t["acc_clean"], "triggered_asr": t["triggered_asr"], "clean_asr": t["clean_asr"],
               "asr_all_nontarget": t["triggered_asr_all_nontarget"], "best_epoch": d["best_epoch"],
               "epochs_run": d["epochs_run"], "gun_shot_poisoned": d["gun_shot_poisoned"],
               "gun_shot_poisoned_frac": d["gun_shot_poisoned_frac_of_class"], "n_poisoned": d["n_poisoned"],
               "commit": d["commit"], "kernel": d["kernel"], "gpu": d["environment"].get("gpu")}
        for snr, m in d["test_by_trigger_snr"].items():
            row[f"asr_trig{snr}"] = m["triggered_asr"]
            row[f"asr_all_nontarget_trig{snr}"] = m["triggered_asr_all_nontarget"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rate", "seed"]).reset_index(drop=True)


def main():
    df = load_runs()
    assert len(df) == 25, f"expected 25 runs, found {len(df)}"
    assert df.groupby("rate").size().eq(5).all() and df.commit.nunique() == 1
    ref = df[df.rate == 0]
    acc_ref = ref.acc_clean.to_numpy()

    per_rate = []
    for rate, g in df[df.rate > 0].groupby("rate"):
        asr = g.triggered_asr.to_numpy()
        acc = g.acc_clean.to_numpy()
        lo, hi = t_interval(asr)
        welch = stats.ttest_ind(acc_ref, acc, equal_var=False)
        d_lo, d_hi, d_df = welch_interval(acc_ref, acc)
        crit = {
            "triggered_asr_gt_85": bool(asr.mean() > H1_ASR),
            "clean_asr_le_10": bool(g.clean_asr.mean() <= H1_CLEAN_ASR),
            "dacc_upper_lt_2pp": bool(d_hi * 100 < H1_DACC_PP),
        }
        per_rate.append({
            "rate": rate, "n": len(g),
            "triggered_asr_mean": float(asr.mean()), "triggered_asr_sd": float(asr.std(ddof=1)),
            "triggered_asr_ci95": [max(lo, 0.0), min(hi, 1.0)], "triggered_asr_ci95_raw": [lo, hi],
            "triggered_asr_by_seed": asr.tolist(),
            "clean_asr_mean": float(g.clean_asr.mean()), "clean_asr_sd": float(g.clean_asr.std(ddof=1)),
            "asr_all_nontarget_mean": float(g.asr_all_nontarget.mean()),
            "asr_all_nontarget_sd": float(g.asr_all_nontarget.std(ddof=1)),
            "acc_clean_mean": float(acc.mean()), "acc_clean_sd": float(acc.std(ddof=1)),
            "dacc_pp_mean": float((acc_ref.mean() - acc.mean()) * 100),
            "dacc_pp_ci95_welch": [d_lo * 100, d_hi * 100], "welch_df": d_df,
            "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
            "welch_significant_bonferroni": bool(welch.pvalue < ALPHA / 4),
            "gun_shot_poisoned_by_seed": g.gun_shot_poisoned.tolist(),
            "gun_shot_poisoned_frac_max": float(g.gun_shot_poisoned_frac.max()),
            "best_epoch_by_seed": g.best_epoch.tolist(),
            "h1_criteria": crit, "h1_all_criteria": all(crit.values()),
            "asr_by_inference_trigger_snr_mean": {s: float(g[f"asr_trig{s}"].mean()) for s in ("40", "30", "20", "10", "0")},
            "asr_by_inference_trigger_snr_sd": {s: float(g[f"asr_trig{s}"].std(ddof=1)) for s in ("40", "30", "20", "10", "0")},
        })

    h1_rates = [r["rate"] for r in per_rate if r["rate"] < 0.05 and r["h1_all_criteria"]]
    report = {
        "selected_snr_db": 10.0, "n_runs": len(df), "commit": df.commit.iloc[0], "gpu": sorted(df.gpu.unique()),
        "reference": {
            "acc_clean_mean": float(acc_ref.mean()), "acc_clean_sd": float(acc_ref.std(ddof=1)),
            "acc_clean_by_seed": acc_ref.tolist(),
            "triggered_asr_mean_10db": float(ref.triggered_asr.mean()),  # control (Section 3.6)
            "asr_all_nontarget_mean_10db": float(ref.asr_all_nontarget.mean()),
            "asr_by_inference_trigger_snr_mean": {s: float(ref[f"asr_trig{s}"].mean()) for s in ("40", "30", "20", "10", "0")},
            "best_epoch_by_seed": ref.best_epoch.tolist(),
        },
        "per_rate": per_rate,
        "h1": {"accepted": bool(h1_rates), "rates_meeting_all_criteria": h1_rates,
               "rule": "some rate < 5%: mean Triggered ASR > 85%, mean Clean ASR <= 10%, ΔAcc 95% upper < 2 pp"},
    }
    out = C.RESULTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "main_stage_runs.csv", index=False, float_format="%.6g")
    (out / "main_stage.json").write_text(json.dumps(report, indent=2, default=C.json_default))

    r = report["reference"]
    print(f"reference: acc {r['acc_clean_mean']:.4f} ± {r['acc_clean_sd']:.4f}, Triggered ASR (10 dB) {r['triggered_asr_mean_10db']:.3f}, "
          f"all-non-target {r['asr_all_nontarget_mean_10db']:.3f}, by inference SNR {r['asr_by_inference_trigger_snr_mean']}")
    for p in per_rate:
        print(f"rate {p['rate'] * 100:g}%: TrigASR {p['triggered_asr_mean']:.3f} ± {p['triggered_asr_sd']:.3f} "
              f"CI [{p['triggered_asr_ci95'][0]:.3f}, {p['triggered_asr_ci95'][1]:.3f}] seeds {p['triggered_asr_by_seed']} | "
              f"CleanASR {p['clean_asr_mean']:.3f} | allNT {p['asr_all_nontarget_mean']:.3f} | acc {p['acc_clean_mean']:.4f} ± {p['acc_clean_sd']:.4f} | "
              f"ΔAcc {p['dacc_pp_mean']:+.2f} pp CI [{p['dacc_pp_ci95_welch'][0]:+.2f}, {p['dacc_pp_ci95_welch'][1]:+.2f}] "
              f"Welch p {p['welch_p']:.3f} | gs poisoned max {p['gun_shot_poisoned_frac_max'] * 100:.1f}% | "
              f"best epochs {p['best_epoch_by_seed']} | H1 {p['h1_criteria']}")
        print(f"      ASR by inference trigger SNR: { {k: round(v, 3) for k, v in p['asr_by_inference_trigger_snr_mean'].items()} }")
    print("H1:", report["h1"])


if __name__ == "__main__":
    main()
