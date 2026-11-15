"""Diagnóstico da confusão sem gatilho na validação (fold 9), a partir dos *_val_logits.npz (D12, 10 dB).
Para cada configuração: previsões falsas de street_music fora da gravação 136399 e probabilidades
medianas (softmax) de street_music e gun_shot nos recortes de 136399.

uso: python3 scripts/s11_val_confusao.py -> results/analysis/val_confusao.json
"""
import glob, json
from pathlib import Path
import numpy as np
SM, GUN, FS = 9, 6, "136399"
out = {}
for pat in ["ref", "rate0.5pct", "rate1pct", "rate2pct", "rate5pct"]:
    r = {"false_sm_outside": [], "p_sm_median": [], "p_gun_median": []}
    for f in sorted(glob.glob(f"results/kaggle/tcc-d12-*/out/runs/{pat}_snr10_seed*_val_logits.npz")):
        z = np.load(f, allow_pickle=True); y = z["labels"]; lc = z["logits_clean"]; pred = lc.argmax(1)
        m = np.array([str(s).startswith(FS) for s in z["slice_file_name"]])
        p = np.exp(lc - lc.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        r["false_sm_outside"].append(int(((pred == SM) & (y != SM) & ~m).sum()))
        r["p_sm_median"].append(float(np.median(p[m, SM]))); r["p_gun_median"].append(float(np.median(p[m, GUN])))
    r["false_sm_outside_mean"] = float(np.mean(r["false_sm_outside"])); out[pat] = r
Path("results/analysis/val_confusao.json").write_text(json.dumps(out, indent=1))
for k, r in out.items():
    print(k, r["false_sm_outside_mean"], np.round(r["p_sm_median"], 3), np.round(r["p_gun_median"], 3))
