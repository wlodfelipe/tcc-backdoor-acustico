"""Validação cruzada em 10 folds (análise de robustez posterior ao protocolo D12), semente 0.
Folds 1-9: kernels tcc-d12-cv-{a,b,c}; fold 10: execuções de semente 0 da rodada D12 (mesma partição).
Por configuração: métricas por fold, média e IC t entre folds, ASR agregada sobre todos os recortes de gun_shot
com bootstrap por gravação de origem (fsID; 10.000 réplicas), ΔAcc pareado por fold e Clean ASR de validação.

uso: python3 scripts/s16_validacao_cruzada.py -> results/analysis/validacao_cruzada.json
"""
import glob, json
from pathlib import Path
import numpy as np
from scipy import stats

GUN, SM = 6, 9
CFG = {"ref": "ref_snr10_seed0", "1%": "rate1pct_snr10_seed0", "2%": "rate2pct_snr10_seed0", "5% a 30 dB": "rate5pct_snr30_seed0"}


def path(fold, name):
    if fold == 10:
        return glob.glob(f"results/kaggle/tcc-d12-*/out/runs/{name}.json")[0]
    return glob.glob(f"results/kaggle/tcc-d12-cv-*/out/runs/fold{fold}/{name}.json")[0]


def tci(x):
    x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


rng = np.random.default_rng(0)
out = {}
for lab, name in CFG.items():
    per = []; hits = []  # hits: (fsid, triggered hit at training SNR, hit at 10 dB)
    for k in range(1, 11):
        jf = path(k, name); d = json.load(open(jf))
        z = np.load(jf.replace(".json", "_test_logits.npz"), allow_pickle=True)
        y = z["labels"]; g = y == GUN; fs = np.array([str(s).split("-")[0] for s in z["slice_file_name"]])[g]
        t_tr = z["logits_triggered"][g].argmax(1) == SM
        t10 = z["logits_triggered_snr10"][g].argmax(1) == SM
        c = z["logits_clean"][g].argmax(1) == SM
        hits += [(f"{k}:{f}", a, b, e) for f, a, b, e in zip(fs, t_tr, t10, c)]
        v = np.load(jf.replace(".json", "_val_logits.npz"), allow_pickle=True)
        vg = v["labels"] == GUN
        per.append({"fold": k, "n_gun": int(g.sum()), "triggered_asr": d["test"]["triggered_asr"],
                    "clean_asr": d["test"]["clean_asr"], "acc_clean": d["test"]["acc_clean"],
                    "nontarget": d["test"]["triggered_asr_all_nontarget"],
                    "asr_ativacao_10db": float(t10.mean()),
                    "val_clean_asr": float((v["logits_clean"][vg].argmax(1) == SM).mean()),
                    "val_triggered_asr": float((v["logits_triggered"][vg].argmax(1) == SM).mean())})
    keys = sorted({h[0] for h in hits}); idx = {k: [i for i, h in enumerate(hits) if h[0] == k] for k in keys}
    arr = np.array([[h[1], h[2], h[3]] for h in hits], float)
    boot = []
    for _ in range(10000):
        sel = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        boot.append(arr[sel].mean(0))
    boot = np.array(boot)
    col = lambda m: [p[m] for p in per]
    out[lab] = {"per_fold": per, "n_gun_total": len(hits), "n_origens": len(keys),
                "triggered_asr_mean_folds": float(np.mean(col("triggered_asr"))), "triggered_asr_ci_folds": tci(col("triggered_asr")),
                "triggered_asr_pooled": float(arr[:, 0].mean()), "triggered_asr_pooled_boot": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                "ativacao_10db_pooled": float(arr[:, 1].mean()), "ativacao_10db_pooled_boot": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
                "clean_asr_pooled": float(arr[:, 2].mean()),
                "acc_mean": float(np.mean(col("acc_clean"))), "nontarget_mean": float(np.mean(col("nontarget"))),
                "val_clean_asr_mean": float(np.mean(col("val_clean_asr"))), "val_clean_asr_max": float(np.max(col("val_clean_asr"))),
                "folds_triggered_ge_85": int(sum(x > 0.85 for x in col("triggered_asr")))}
ref = [p["acc_clean"] for p in out["ref"]["per_fold"]]
for lab in ("1%", "2%", "5% a 30 dB"):
    diff = 100 * (np.array(ref) - np.array([p["acc_clean"] for p in out[lab]["per_fold"]]))
    t = stats.ttest_rel(ref, [p["acc_clean"] for p in out[lab]["per_fold"]])
    out[lab]["dacc_pp_mean"] = float(diff.mean()); out[lab]["dacc_pp_ci_pareado"] = tci(diff); out[lab]["dacc_p_pareado"] = float(t.pvalue)
Path("results/analysis/validacao_cruzada.json").write_text(json.dumps(out, indent=1))
for lab, o in out.items():
    print(lab, "n", o["n_gun_total"], "orig", o["n_origens"], "| trig folds %.1f %s" % (100 * o["triggered_asr_mean_folds"], np.round(np.array(o["triggered_asr_ci_folds"]) * 100, 1)),
          "pooled %.1f %s" % (100 * o["triggered_asr_pooled"], np.round(np.array(o["triggered_asr_pooled_boot"]) * 100, 1)),
          "| at10 %.1f %s" % (100 * o["ativacao_10db_pooled"], np.round(np.array(o["ativacao_10db_pooled_boot"]) * 100, 1)),
          "| clean %.1f | acc %.2f | nt %.1f | valclean %.1f (max %.1f) | folds>85: %d" % (100 * o["clean_asr_pooled"], 100 * o["acc_mean"], 100 * o["nontarget_mean"], 100 * o["val_clean_asr_mean"], 100 * o["val_clean_asr_max"], o["folds_triggered_ge_85"]),
          "| dAcc", (None if lab == "ref" else (round(o["dacc_pp_mean"], 2), np.round(o["dacc_pp_ci_pareado"], 2), round(o["dacc_p_pareado"], 3))))
    print("   por fold trig:", [round(100 * p["triggered_asr"], 1) for p in o["per_fold"]], "at10:", [round(100 * p["asr_ativacao_10db"], 1) for p in o["per_fold"]] if lab.startswith("5%") else "")
