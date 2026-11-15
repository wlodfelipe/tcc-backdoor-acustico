"""Entry point inside a Kaggle kernel. The job dict (built by kaggle_push.py) lists the steps:

  verify  SHA-256 of every WAV against results/wav_sha256.csv; aborts on any mismatch
  phase1  stages 2a-2c (official numbers, Kaggle library versions)
  detect  stage 2b and the AST-input delta for every SNR in job["snrs"] (no GPU needed)
  runs    clean features once, triggered features once per SNR, then one training run per
          spec; each run is saved as soon as it ends
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(job: dict):
    from tcc import config as C
    from tcc import data, integrity

    out = C.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(job, indent=2))
    meta = data.load_metadata()

    if "verify" in job["steps"]:
        t0 = time.time()
        report = integrity.compare(pd.read_csv(C.WAV_HASHES_PATH), integrity.hash_all(meta, workers=8))
        report["seconds"] = round(time.time() - t0, 1)
        (out / "integrity.json").write_text(json.dumps(report, indent=2))
        log(f"WAV hashes: {report['n_observed']} files, {report['n_mismatch']} mismatches")
        if not report["identical"]:
            raise SystemExit("dataset differs from the local copy; aborting")

    if "phase1" in job["steps"]:
        for script in ("s1_data.py", "s2_trigger.py", "s3_poison.py"):
            log(f"running {script}")
            subprocess.run([sys.executable, str(REPO / "scripts" / script)], check=True)

    if "detect" in job["steps"]:
        ensure_cache(meta)
        for snr in job["snrs"]:
            for script in ("s2_trigger.py", "s4_input_delta.py"):
                log(f"running {script} at {snr:g} dB")
                subprocess.run([sys.executable, str(REPO / "scripts" / script), "--snr", str(snr)], check=True)

    if "runs" in job["steps"]:
        run_all(job, meta, out)
    if "cv" in job["steps"]:
        run_cv(job, meta, out)
    log("job finished")


def run_all(job, meta, out):
    import numpy as np
    import torch
    from tcc import config as C
    from tcc import data, features, model, train

    runs_dir = out / "runs"
    pending = [train.RunSpec(r["rate"], r["seed"], r.get("snr_db", C.SNR_DB)) for r in job["runs"]]
    pending = [s for s in pending if not (runs_dir / f"{s.name}.json").exists()]
    if not pending:
        return
    ensure_cache(meta)
    waves, lengths = data.open_cache()
    t0 = time.time()
    clean = features.cached(waves, lengths, None)
    feat_s = time.time() - t0
    log(f"clean features ready in {feat_s:.0f}s")
    manifest = pd.read_csv(C.MANIFEST_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not job.get("allow_cpu"):
        raise SystemExit("no GPU available")
    extra = {"commit": job.get("commit"), "kernel": job.get("slug"), "feature_seconds": round(feat_s, 1),
             "feature_extractor_backend": "torchaudio" if _speech() else "numpy"}
    # additional analysis: every trained model is also tested with the triggers of all SNRs
    eval_trigs = {}
    for snr in job.get("eval_snrs", []):
        t0 = time.time()
        eval_trigs[float(snr)] = features.cached(waves, lengths, float(snr), mmap=True)
        log(f"evaluation features at {snr:g} dB ready in {time.time() - t0:.0f}s")
    for spec in pending:
        t0 = time.time()
        trig = features.cached(waves, lengths, spec.snr_db)
        log(f"triggered features at {spec.snr_db:g} dB ready in {time.time() - t0:.0f}s")
        log(f"starting {spec.name}")
        kw = {k: job[k] for k in ("max_epochs", "patience", "select") if k in job}
        res = train.run(spec, meta, manifest, clean, trig, runs_dir, device, model.build, extra_info=extra,
                        eval_trigs=eval_trigs, **kw)
        log(f"finished {spec.name}: best epoch {res['best_epoch']}, test {res['test']}")


def run_cv(job, meta, out):
    """Cross-validation (robustness analysis added after the D12 protocol): for each test fold k in job["cv_folds"],
    validation is fold k-1 (fold 10 when k = 1) and training uses the other eight folds. The poisoned subsets are drawn
    for that training set with the same rule and seeds, and saved with the runs."""
    import torch
    from tcc import config as C
    from tcc import data, features, model, poison, train

    ensure_cache(meta)
    waves, lengths = data.open_cache()
    clean = features.cached(waves, lengths, None)
    trigs = {float(s): features.cached(waves, lengths, float(s), mmap=True) for s in job["eval_snrs"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no GPU available")
    for k in job["cv_folds"]:
        v = 10 if k == 1 else k - 1
        m = meta.copy()
        m["split"] = ["test" if f == k else "val" if f == v else "train" for f in m["fold"]]
        runs_dir = out / "runs" / f"fold{k}"
        runs_dir.mkdir(parents=True, exist_ok=True)
        specs = [train.RunSpec(r["rate"], r["seed"], r["snr_db"]) for r in job["runs"]]
        manifest = pd.concat([poison.select_poison(m, s.rate, s.seed) for s in specs if s.rate > 0], ignore_index=True)
        manifest.to_csv(runs_dir / "poison_manifest.csv", index=False)
        extra = {"commit": job.get("commit"), "kernel": job.get("slug"), "test_fold": k, "val_fold": v}
        for spec in specs:
            if (runs_dir / f"{spec.name}.json").exists():
                continue
            log(f"fold {k}: starting {spec.name}")
            res = train.run(spec, m, manifest, clean, trigs[spec.snr_db], runs_dir, device, model.build, extra_info=extra,
                            eval_trigs=trigs, max_epochs=job["max_epochs"], patience=job["patience"], select=job["select"])
            log(f"fold {k}: finished {spec.name}: test {res['test']}")


def ensure_cache(meta):
    """The 16 kHz waveform cache, built here when no earlier step (phase1) has built it."""
    from tcc import config as C
    from tcc import data

    if not (C.DATA_CACHE_DIR / "lengths.npy").exists():
        t0 = time.time()
        errors = data.build_cache(meta)
        log(f"waveform cache built in {time.time() - t0:.0f}s, unreadable: {len(errors)}")


def _speech():
    from transformers.utils import is_speech_available
    return is_speech_available()


if __name__ == "__main__":
    main(json.loads(os.environ["TCC_JOB"]))
