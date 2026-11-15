"""Acurácia limpa de validação (época selecionada) por taxa a 10 dB e comparação com a referência
(Welch, correção de Bonferroni para 4 comparações). Análise adicional: a decisão sobre H1 usa o teste.

uso: python3 scripts/s8_val_accuracy.py -> results/analysis/val_accuracy.json
"""
import glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

acc = defaultdict(list)
for f in sorted(glob.glob("results/kaggle/tcc-principal-*/out/runs/*_snr10_seed*.json")):
    d = json.load(open(f))
    k = "ref" if d["run"].startswith("ref") else re.match(r"rate([\d.]+)pct", d["run"]).group(1)
    acc[k].append(d["best_val_acc"])
ref = np.array(acc["ref"]); out = {"ref": {"mean": ref.mean(), "sd": ref.std(ddof=1), "n": len(ref)}, "rates": {}}
for k in ["0.5", "1", "2", "5"]:
    a = np.array(acc[k]); t = stats.ttest_ind(ref, a, equal_var=False)
    out["rates"][k] = {"mean": a.mean(), "sd": a.std(ddof=1), "delta_pp": 100 * (ref.mean() - a.mean()),
                       "p_welch": float(t.pvalue), "significant_bonferroni": bool(t.pvalue < 0.05 / 4)}
Path("results/analysis/val_accuracy.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
