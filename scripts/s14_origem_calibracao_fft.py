"""(a) Triggered ASR de validação da calibração sem a gravação 136399 e acertos de teste por gravação de origem;
(b) detector FFT com limiar fixado nas gravações limpas dos folds 1-8 e taxa de detecção medida nas gravações
envenenadas dos folds 9-10, sem street_music.

uso: python3 scripts/s14_origem_calibracao_fft.py -> results/analysis/origem_calibracao_fft.json
"""
import csv, glob, json, collections
from pathlib import Path
import numpy as np
out = {"calibracao": {}, "fft_fora_da_amostra": {}}
for snr in (40, 30, 25, 20, 15, 10, 0):
    tv, hits = [], collections.Counter()
    for f in sorted(glob.glob(f"results/kaggle/tcc-d12-varredura-*/out/runs/rate5pct_snr{snr}_seed*_val_logits.npz")):
        z = np.load(f, allow_pickle=True); y = z["labels"]; n = np.array([str(s) for s in z["slice_file_name"]])
        m = (y == 6) & ~np.char.startswith(n, "136399"); tv.append(float((z["logits_triggered"][m].argmax(1) == 9).mean()))
    for f in sorted(glob.glob(f"results/kaggle/tcc-d12-varredura-*/out/runs/rate5pct_snr{snr}_seed*_test_logits.npz")):
        z = np.load(f, allow_pickle=True); y = z["labels"]; n = np.array([str(s).split("-")[0] for s in z["slice_file_name"]])
        m = y == 6; p = z["logits_triggered"][m].argmax(1) == 9
        hits.update(n[m][p].tolist())
    out["calibracao"][str(snr)] = {"val_triggered_sem_136399": float(np.mean(tv)), "test_hits_by_origin": dict(hits)}
for snr in (40, 30, 25, 20, 15, 10, 0):
    d = "tcc-d12-deteccao-extra" if snr in (25, 15) else "tcc-deteccao-snr"
    R = list(csv.DictReader(open(f"results/kaggle/{d}/out/phase1/s2_trigger_per_clip_snr{snr}.csv")))
    cal = np.array([float(r["fft_prominence_db_clean"]) for r in R if r["split"] == "train"])
    ev = [r for r in R if r["split"] in ("val", "test")]
    neg = np.array([float(r["fft_prominence_db_clean"]) for r in ev])
    pos = np.array([float(r["fft_prominence_db_poisoned"]) for r in ev if r["class"] != "street_music"])
    res = {}
    for fpr in (0.01, 0.05):
        thr = float(np.quantile(cal, 1 - fpr))
        res[str(fpr)] = {"threshold_db": thr, "tpr": float((pos > thr).mean()), "fpr_eval": float((neg > thr).mean())}
    out["fft_fora_da_amostra"][str(snr)] = res
Path("results/analysis/origem_calibracao_fft.json").write_text(json.dumps(out, indent=1))
for k, v in out["calibracao"].items(): print(k, round(v["val_triggered_sem_136399"] * 100, 1), v["test_hits_by_origin"])
for k, v in out["fft_fora_da_amostra"].items(): print(k, {f: (round(x["tpr"] * 100, 2), round(x["fpr_eval"] * 100, 2)) for f, x in v.items()})
