"""(a) Validação na época que a vítima escolheria (maior acurácia limpa de validação) contra a última época,
etapa principal a 10 dB; (b) bootstrap por gravação de origem do backdoor latente no fold 10
(envenenamento a 30 dB, ativação a 10 dB, sementes 0-2, 10.000 réplicas).

uso: python3 scripts/s17_checkpoint_vitima.py -> results/analysis/checkpoint_vitima.json
"""
import glob, json
from pathlib import Path
import numpy as np
out = {"checkpoint": {}}
for pat in ["ref", "rate0.5pct", "rate1pct", "rate2pct", "rate5pct"]:
    b, l, ab, al, ep = [], [], [], [], []
    for f in sorted(glob.glob(f"results/kaggle/tcc-d12-*/out/runs/{pat}_snr10_seed*.json")):
        h = json.load(open(f))["history"]; i = int(np.argmax([e["val_acc"] for e in h]))
        b.append(h[i]["val_triggered_asr"]); l.append(h[-1]["val_triggered_asr"]); ab.append(h[i]["val_acc"]); al.append(h[-1]["val_acc"]); ep.append(i + 1)
    out["checkpoint"][pat] = {"best_epoch": ep, "val_triggered_best": b, "val_triggered_best_mean": float(np.mean(b)),
                              "val_triggered_last_mean": float(np.mean(l)), "val_acc_best_mean": float(np.mean(ab)), "val_acc_last_mean": float(np.mean(al))}
rng = np.random.default_rng(0); H = []
for f in sorted(glob.glob("results/kaggle/tcc-d12-varredura-*/out/runs/rate5pct_snr30_seed*_test_logits.npz")):
    z = np.load(f, allow_pickle=True); g = z["labels"] == 6
    fs = np.array([str(s).split("-")[0] for s in z["slice_file_name"]])[g]; H.append(z["logits_triggered_snr10"][g].argmax(1) == 9)
H = np.mean(H, 0); keys = sorted(set(fs)); idx = {k: np.where(fs == k)[0] for k in keys}
boot = [H[np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])].mean() for _ in range(10000)]
out["latente_fold10"] = {"mean": float(H.mean()), "boot95": np.percentile(boot, [2.5, 97.5]).tolist()}
Path("results/analysis/checkpoint_vitima.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["latente_fold10"]), {k: (round(v["val_triggered_best_mean"] * 100, 1), round(v["val_triggered_last_mean"] * 100, 1)) for k, v in out["checkpoint"].items()})
