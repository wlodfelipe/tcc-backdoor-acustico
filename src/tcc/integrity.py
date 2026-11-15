"""SHA-256 of every WAV, to prove the Kaggle copy is byte-identical to the local one."""

from concurrent.futures import ThreadPoolExecutor
import hashlib

import pandas as pd


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_all(meta: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        digests = list(ex.map(sha256, meta["path"]))
    return pd.DataFrame({"fold": meta["fold"], "slice_file_name": meta["slice_file_name"], "sha256": digests})


def compare(expected: pd.DataFrame, observed: pd.DataFrame) -> dict:
    m = expected.merge(observed, on=["fold", "slice_file_name"], how="outer", suffixes=("_exp", "_obs"))
    diff = m[m["sha256_exp"] != m["sha256_obs"]]
    return {"n_expected": len(expected), "n_observed": len(observed), "n_mismatch": len(diff),
            "mismatches": diff["slice_file_name"].head(50).tolist(), "identical": len(diff) == 0}
