"""Bundle the committed code and push it as a private Kaggle kernel.

  python scripts/kaggle_push.py <job>            push (working tree must be clean)
  python scripts/kaggle_push.py <job> --dry-run  build only, in build/kaggle/<slug>/

The kernel is one script: the code (src/, scripts/, manifest, WAV hashes) travels inside it as a
base64 tar.gz, is extracted to /kaggle/working/tcc and kept in the kernel output, so every
result can be traced to the exact code and commit that produced it.
"""

import base64
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "flpwld"
DATASET = "chrisfilo/urbansound8k"

JOBS = {
    "fase1": {"slug": "tcc-fase1-oficial", "gpu": False, "steps": ["verify", "phase1"]},
    "piloto": {"slug": "tcc-piloto", "gpu": True, "steps": ["verify", "runs"],
               "runs": [{"rate": 0.0, "seed": 0}, {"rate": 0.05, "seed": 0}]},
    # D10: detectability for all five SNRs (CPU) and calibration at 5%, seed 0 (T4)
    "deteccao": {"slug": "tcc-deteccao-snr", "gpu": False, "steps": ["verify", "detect"],
                 "snrs": [40.0, 30.0, 20.0, 10.0, 0.0]},
    "calibracao": {"slug": "tcc-calibracao", "gpu": True, "steps": ["verify", "runs"],
                   "runs": [{"rate": 0.05, "seed": 0, "snr_db": s} for s in (40.0, 30.0, 20.0, 10.0, 0.0)]},
}

# Main stage (Section 3.4): selected SNR 10 dB; 4 rates x 5 seeds + reference x 5 seeds, split
# into three kernels of 8-9 runs (~3 h each). Every model is also tested at all five SNRs.
_MAIN = [{"rate": r, "seed": s, "snr_db": 10.0} for r in (0.0, 0.005, 0.01, 0.02, 0.05) for s in range(5)]
for _k, _part in zip("abc", (_MAIN[:8], _MAIN[8:16], _MAIN[16:])):
    JOBS[f"principal-{_k}"] = {"slug": f"tcc-principal-{_k}", "gpu": True, "steps": ["verify", "runs"],
                               "runs": _part, "eval_snrs": [40.0, 30.0, 20.0, 10.0, 0.0]}

# Complementary check of calibration (Section 4): the two SNRs where H2 holds, at 5%, seeds 1-4
# (seed 0 is the calibration run). Does not change the pre-registered selection or H1/H2 decisions.
for _snr in (40.0, 30.0):
    JOBS[f"verificacao-{_snr:g}"] = {"slug": f"tcc-verificacao-snr{_snr:g}", "gpu": True, "steps": ["verify", "runs"],
                                     "runs": [{"rate": 0.05, "seed": s, "snr_db": _snr} for s in range(1, 5)],
                                     "eval_snrs": [40.0, 30.0, 20.0, 10.0, 0.0]}

# Fixed-horizon check (Section 4.2/4.6, added after the main stage): 30 and 40 dB at 5%, seeds 0-2,
# 20 epochs without early stopping, the last epoch is evaluated. Validation logits are saved.
for _snr in (40.0, 30.0):
    JOBS[f"horizonte-{_snr:g}"] = {"slug": f"tcc-horizonte-snr{_snr:g}", "gpu": True, "steps": ["verify", "runs"],
                                   "runs": [{"rate": 0.05, "seed": s, "snr_db": _snr} for s in range(3)],
                                   "max_epochs": 20, "patience": 20, "select": "last",
                                   "eval_snrs": [40.0, 30.0, 20.0, 10.0, 0.0]}

# D12 (definitive protocol, registered before running): fixed 10-epoch horizon, last epoch evaluated,
# SNR selected on VALIDATION. Stage 1: SNR sweep at 5%, seeds 0-2, split into two kernels.
_SWEEP_SNRS = [40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 0.0]
_FIXED = {"max_epochs": 10, "patience": 10, "select": "last", "eval_snrs": _SWEEP_SNRS}
for _k, _snrs in (("a", [40.0, 30.0, 25.0, 20.0]), ("b", [15.0, 10.0, 0.0])):
    JOBS[f"varredura-{_k}"] = {"slug": f"tcc-d12-varredura-{_k}", "gpu": True, "steps": ["verify", "runs"],
                               "runs": [{"rate": 0.05, "seed": s, "snr_db": x} for x in _snrs for s in range(3)],
                               **_FIXED}
JOBS["deteccao-extra"] = {"slug": "tcc-d12-deteccao-extra", "gpu": False, "steps": ["verify", "detect"],
                          "snrs": [25.0, 15.0]}

FILES = ["src/tcc/*.py", "scripts/*.py", "results/poison_manifest.csv", "results/wav_sha256.csv"]

TEMPLATE = '''import base64, io, json, os, subprocess, sys, tarfile
JOB = json.loads({job!r})
BUNDLE = "{bundle}"
root = os.environ.get("TCC_BUNDLE_ROOT", "/kaggle/working/tcc")
os.makedirs(root, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(root)
env = dict(os.environ, TCC_JOB=json.dumps(JOB), TCC_RESULTS=os.environ.get("TCC_RESULTS", "/kaggle/working/out"),
           TCC_CACHE=os.environ.get("TCC_CACHE", "/tmp/tcc_cache"))
subprocess.run([sys.executable, root + "/scripts/kaggle_job.py"], env=env, check=True)
'''


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for pattern in FILES:
            for p in sorted(REPO.glob(pattern)):
                tar.add(p, arcname=str(p.relative_to(REPO)))
    return base64.b64encode(buf.getvalue()).decode()


def main():
    name, dry = sys.argv[1], "--dry-run" in sys.argv
    spec = JOBS[name]
    dirty = git("status", "--porcelain")
    if dirty and not dry:
        sys.exit("working tree not clean; commit first:\n" + dirty)
    job = {**spec, "commit": git("rev-parse", "HEAD") + ("-dirty" if dirty else "")}
    out = REPO / "build" / "kaggle" / spec["slug"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(TEMPLATE.format(job=json.dumps(job), bundle=bundle()))
    meta = {"id": f"{USER}/{spec['slug']}", "title": spec["slug"], "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": spec["gpu"], "enable_internet": True, "dataset_sources": [DATASET]}
    if spec["gpu"]:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {out} (commit {job['commit']}, {(out / 'run.py').stat().st_size / 1e3:.0f} kB)")
    if not dry:
        kaggle = Path.home() / ".local" / "bin" / "kaggle"
        subprocess.run([str(kaggle), "kernels", "push", "-p", str(out)], check=True)


if __name__ == "__main__":
    main()
