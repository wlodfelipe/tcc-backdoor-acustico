"""Clean ASR e Triggered ASR de validação (fold 9) por gravação de origem (fsID), na última época,
a partir dos *_val_logits.npz de todas as execuções D12.

uso: python3 scripts/s10_val_clean_asr.py -> results/analysis/val_por_origem.json
"""
import glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
GUN, TGT = 6, 9  # classID UrbanSound8K: gun_shot, street_music
out = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob("results/kaggle/tcc-d12-*/out/runs/*_val_logits.npz")):
    m = re.search(r"rate(\w+?)(?:pct)?_snr(\d+)_seed(\d)", Path(f).name)
    name = Path(f).name.replace("_val_logits.npz", "")
    z = np.load(f, allow_pickle=True)
    y = z["labels"]; g = y == GUN
    fs = np.array([str(s).split("-")[0] for s in z["slice_file_name"]])[g]
    pc = z["logits_clean"][g].argmax(1) == TGT
    pt = z["logits_triggered"][g].argmax(1) == TGT
    key = name.split("_seed")[0]
    out[key]["clean_asr"].append(float(pc.mean()))
    out[key]["triggered_asr"].append(float(pt.mean()))
    out[key]["clean_hits_by_fsid"].append({k: int(pc[fs == k].sum()) for k in sorted(set(fs))})
    out[key]["n_by_fsid"] = {k: int((fs == k).sum()) for k in sorted(set(fs))}
res = {k: {"clean_asr_mean": float(np.mean(v["clean_asr"])), "triggered_asr_mean": float(np.mean(v["triggered_asr"])),
           "clean_asr": v["clean_asr"], "n_gun_shot": int(sum(v["n_by_fsid"].values())), "n_by_fsid": v["n_by_fsid"],
           "clean_hits_by_fsid_sum": {fs: sum(d[fs] for d in v["clean_hits_by_fsid"]) for fs in v["n_by_fsid"]}}
       for k, v in out.items()}
Path("results/analysis/val_por_origem.json").write_text(json.dumps(res, indent=1))
for k, r in sorted(res.items()):
    print(k, len(r["clean_asr"]), "clean", round(r["clean_asr_mean"] * 100, 1), "trig", round(r["triggered_asr_mean"] * 100, 1),
          {fs: h for fs, h in r["clean_hits_by_fsid_sum"].items() if h})
