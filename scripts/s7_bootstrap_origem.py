"""Incerteza da Triggered ASR no teste considerando a dependência entre recortes (análise adicional).

Os 32 recortes de gun_shot do fold 10 vêm de 11 gravações de origem (fsID). Para cada taxa a 10 dB,
calcula a Triggered ASR média sobre as 5 sementes e um IC 95% por bootstrap de gravações de origem
(reamostra fsIDs com reposição, mantendo todos os recortes de cada um; 10.000 réplicas, semente 0),
além da ASR por gravação de origem. Lê apenas os logits de teste salvos em results/.

uso: python3 scripts/s7_bootstrap_origem.py  -> results/analysis/bootstrap_origem.json
"""
import csv, glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

META = Path.home() / "Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
SRC, TGT = 6, 9  # classID gun_shot, street_music
fsid = {r["slice_file_name"]: r["fsID"] for r in csv.DictReader(open(META))}

runs = defaultdict(list)
for f in sorted(glob.glob("results/kaggle/tcc-d12-*/out/runs/*_snr10_seed*_test_logits.npz")):
    name = Path(f).name
    rate = "ref" if name.startswith("ref") else re.match(r"rate([\d.]+)pct", name).group(1)
    d = np.load(f, allow_pickle=True)
    m = d["labels"] == SRC
    hit = (d["logits_triggered"].argmax(1)[m] == TGT).astype(float)
    clean = (d["logits_clean"].argmax(1)[m] == TGT).astype(float)
    runs[rate].append((hit, clean, [fsid[s] for s in d["slice_file_name"][m]]))

rng = np.random.default_rng(0); out = {"n_boot": 10000, "rates": {}}
for rate, lst in sorted(runs.items()):
    files = lst[0][2]; assert all(x[2] == files for x in lst)
    H = np.stack([x[0] for x in lst])          # seeds x clips
    srcs = sorted(set(files)); idx = {s: [i for i, f in enumerate(files) if f == s] for s in srcs}
    boots = []
    for _ in range(out["n_boot"]):
        pick = rng.choice(srcs, size=len(srcs), replace=True)
        cols = [i for s in pick for i in idx[s]]
        boots.append(H[:, cols].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["rates"][rate] = {
        "n_seeds": len(lst), "n_clips": len(files), "n_sources": len(srcs),
        "triggered_asr_mean": float(H.mean()), "boot_ci95": [float(lo), float(hi)],
        "clean_asr_mean": float(np.mean([x[1].mean() for x in lst])),
        "asr_by_source": {s: {"clips": len(idx[s]), "asr": float(H[:, idx[s]].mean())} for s in srcs}}
Path("results/analysis/bootstrap_origem.json").write_text(json.dumps(out, indent=1))
for r, v in out["rates"].items():
    print(r, round(v["triggered_asr_mean"]*100, 1), [round(x*100, 1) for x in v["boot_ci95"]],
          {s: round(a["asr"]*100) for s, a in v["asr_by_source"].items()})
