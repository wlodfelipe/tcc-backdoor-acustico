"""Diferença absoluta média e fração de células alteradas pelo gatilho de 40 dB na entrada do AST, em float32
e após a conversão para float16, nas mesmas 400 gravações de s12 (quadros da duração original).

uso: python3 scripts/s13_fp32_vs_fp16.py (imprime o resultado; registrado em NOTAS.md)
"""
import sys, numpy as np
sys.path.insert(0, "src")
from tcc import data, features, trigger, config as C
meta = data.load_metadata()
idx = np.random.default_rng(0).choice(len(meta), 400, replace=False)
fe = features.extractor()
d32, d16, n32, n16 = [], [], [], []
for i in idx:
    y = data.load_clip(meta.iloc[i]["path"]); L = len(y)
    w = np.zeros(C.N_SAMPLES, np.float32); w[:L] = y
    t = trigger.poison_clip(w, L, 40.0)
    a = fe([w], sampling_rate=C.SR, return_tensors="np")["input_values"][0].astype(np.float32)
    b = fe([t], sampling_rate=C.SR, return_tensors="np")["input_values"][0].astype(np.float32)
    fr = min(C.AST_FRAMES, L // 160); a, b = a[:fr], b[:fr]
    d32.append(np.abs(b - a).ravel()); d16.append(np.abs(b.astype(np.float16).astype(np.float32) - a.astype(np.float16).astype(np.float32)).ravel())
d32 = np.concatenate(d32); d16 = np.concatenate(d16)
print("fp32 mean %.5f frac>0 %.3f | fp16 mean %.5f frac>0 %.3f" % (d32.mean(), (d32 > 0).mean(), d16.mean(), (d16 > 0).mean()))
import json
from pathlib import Path
Path("results/analysis/fp32_vs_fp16_40db.json").write_text(json.dumps({"n_clips": len(idx), "mean_abs_delta_fp32": float(d32.mean()),
    "frac_cells_changed_fp32": float((d32 > 0).mean()), "mean_abs_delta_fp16": float(d16.mean()), "frac_cells_changed_fp16": float((d16 > 0).mean())}, indent=1))
