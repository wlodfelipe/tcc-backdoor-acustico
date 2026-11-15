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
}

# D12 (definitive protocol, registered before running): fixed 10-epoch horizon, last epoch evaluated,
# SNR selected on VALIDATION. Stage 1: SNR sweep at 5%, seeds 0-2, split into two kernels.
_SWEEP_SNRS = [40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 0.0]
_FIXED = {"max_epochs": 10, "patience": 10, "select": "last", "eval_snrs": _SWEEP_SNRS}
for _k, _snrs in (("a", [40.0, 30.0, 25.0, 20.0]), ("b", [15.0, 10.0, 0.0])):
    JOBS[f"varredura-{_k}"] = {"slug": f"tcc-d12-varredura-{_k}", "gpu": True, "steps": ["verify", "runs"],
                               "runs": [{"rate": 0.05, "seed": s, "snr_db": x} for x in _snrs for s in range(3)],
                               **_FIXED}
# D12 stage 2, at the SNR selected on validation (10 dB): rates 0.5/1/2% and reference, seeds 0-4,
# plus 5% seeds 3-4 (seeds 0-2 come from the sweep). Two kernels of 11 runs.
_MAIN = [{"rate": r, "seed": s, "snr_db": 10.0} for r in (0.0, 0.005, 0.01, 0.02) for s in range(5)]
_MAIN += [{"rate": 0.05, "seed": s, "snr_db": 10.0} for s in (3, 4)]
for _k, _part in (("a", _MAIN[:11]), ("b", _MAIN[11:])):
    JOBS[f"principal-{_k}"] = {"slug": f"tcc-d12-principal-{_k}", "gpu": True, "steps": ["verify", "runs"],
                               "runs": _part, **_FIXED}
# cross-validation (robustness analysis, after the D12 protocol): 10 test folds, seed 0
_CV_RUNS = [{"rate": 0.0, "seed": 0, "snr_db": 10.0}, {"rate": 0.01, "seed": 0, "snr_db": 10.0},
            {"rate": 0.02, "seed": 0, "snr_db": 10.0}, {"rate": 0.05, "seed": 0, "snr_db": 30.0}]
for _k, _folds in (("a", [1, 2, 3]), ("b", [4, 5, 6]), ("c", [7, 8, 9])):  # fold 10 = D12 seed-0 runs
    JOBS[f"cv-{_k}"] = {"slug": f"tcc-d12-cv-{_k}", "gpu": True, "steps": ["verify", "cv"], "cv_folds": _folds,
                        "runs": _CV_RUNS, "max_epochs": 10, "patience": 10, "select": "last", "eval_snrs": [30.0, 10.0]}
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
