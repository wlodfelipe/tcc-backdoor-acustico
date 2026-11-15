"""Ativação cruzada na calibração (análise prevista no protocolo D12): modelos envenenados a 5% em cada SNR,
avaliados no teste com o gatilho aplicado em cada SNR de inferência (campo test_by_trigger_snr de cada execução).

uso: python3 scripts/s15_ativacao_calibracao.py -> results/analysis/ativacao_calibracao.json
"""
import glob, json
from pathlib import Path
import numpy as np
out = {}
for snr in (40, 30, 25, 20, 15, 10, 0):
    runs = [json.load(open(f))["test_by_trigger_snr"] for f in sorted(glob.glob(f"results/kaggle/tcc-d12-varredura-*/out/runs/rate5pct_snr{snr:g}_seed*.json"))]
    out[str(snr)] = {inf: {"triggered_asr_by_seed": [r[inf]["triggered_asr"] for r in runs],
                           "triggered_asr_mean": float(np.mean([r[inf]["triggered_asr"] for r in runs])),
                           "nontarget_mean": float(np.mean([r[inf]["triggered_asr_all_nontarget"] for r in runs])),
                           "clean_asr_mean": float(np.mean([r[inf]["clean_asr"] for r in runs]))} for inf in runs[0]}
Path("results/analysis/ativacao_calibracao.json").write_text(json.dumps(out, indent=1))
for tr, d in out.items():
    print(tr, {inf: round(v["triggered_asr_mean"] * 100, 1) for inf, v in d.items()}, "| nt@10", round(d["10"]["nontarget_mean"] * 100, 1))
