"""ASR sobre todas as classes não-alvo (teste) nas execuções a 5% de cada SNR da calibração e da
verificação complementar. Análise adicional, para comparação com métricas da literatura.

uso: python3 scripts/s9_nontarget_snr.py -> results/analysis/nontarget_by_snr.json
"""
import glob, json
from collections import defaultdict
from pathlib import Path
v = defaultdict(list)
for f in sorted(glob.glob("results/kaggle/tcc-calibracao/out/runs/rate5pct_snr*_seed0.json") +
                glob.glob("results/kaggle/tcc-verificacao-*/out/runs/rate5pct_snr*_seed*.json")):
    d = json.load(open(f)); v[str(d["snr_db"])].append(d["test"]["triggered_asr_all_nontarget"])
out = {k: {"n": len(x), "mean": sum(x) / len(x), "values": x} for k, x in v.items()}
Path("results/analysis/nontarget_by_snr.json").write_text(json.dumps(out, indent=1))
print({k: (o["n"], round(o["mean"] * 100, 1)) for k, o in out.items()})
