"""Fração das células da entrada do AST (fbank normalizado, armazenado em float16 como no treinamento)
que muda com o gatilho, por SNR, em 400 gravações sorteadas (semente 0), nos quadros da duração original.

uso: python3 scripts/s12_fp16_entrada.py -> results/analysis/fp16_entrada.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from tcc import data, features, config as C
meta = data.load_metadata()
idx = np.random.default_rng(0).choice(len(meta), 400, replace=False)
waves = np.zeros((len(idx), C.N_SAMPLES), np.float32); lengths = np.zeros(len(idx), int)
for k, i in enumerate(idx):
    y = data.load_clip(meta.iloc[i]["path"]); waves[k, :len(y)] = y; lengths[k] = len(y)
clean = features.build(waves, lengths, None)
frames = np.minimum(C.AST_FRAMES, lengths // 160)  # passo de 10 ms
mask = np.arange(C.AST_FRAMES)[None, :] < frames[:, None]
out = {}
for snr in (40.0, 30.0, 20.0, 10.0):
    trig = features.build(waves, lengths, snr)
    diff = (trig != clean).any(axis=2) if False else (trig != clean)
    d = diff[mask]
    out[str(snr)] = {"frac_cells_changed": float(d.mean()),
                     "mean_abs_delta_fp16": float(np.abs(trig.astype(np.float32) - clean.astype(np.float32))[mask].mean())}
    print(snr, out[str(snr)])
Path("results/analysis/fp16_entrada.json").write_text(json.dumps(out, indent=1))
