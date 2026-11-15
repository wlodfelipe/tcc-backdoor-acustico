"""Stage 2c: poison-set selection (Section 3.3).

Selections are drawn once, here, and stored as a manifest in results/, so the Kaggle runs
read the exact same clips instead of re-drawing them with a possibly different numpy.
"""

import numpy as np
import pandas as pd

from . import config as C


def n_poison(n_train: int, rate: float) -> int:
    return int(round(rate * n_train))


def select_poison(meta: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Uniform draw, without replacement, from training clips of every class except the
    target. The rate is over the whole training set."""
    train = meta[meta["split"] == "train"]
    pool = train[train["class"] != C.TARGET_CLASS]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n_poison(len(train), rate), replace=False)
    chosen = pool.iloc[np.sort(idx)]
    return pd.DataFrame({
        "rate": rate, "seed": seed,
        "slice_file_name": chosen["slice_file_name"].values,
        "orig_class": chosen["class"].values,
    })


def build_manifest(meta: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [select_poison(meta, r, s) for r in C.POISON_RATES for s in C.SEEDS], ignore_index=True
    )


def apply_labels(meta: pd.DataFrame, manifest: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Training metadata with poisoned rows flagged and relabelled to the target class."""
    train = meta[meta["split"] == "train"].copy()
    names = set(manifest.query("rate == @rate and seed == @seed")["slice_file_name"])
    train["poisoned"] = train["slice_file_name"].isin(names)
    target_id = int(meta.loc[meta["class"] == C.TARGET_CLASS, "classID"].iloc[0])
    train.loc[train["poisoned"], "class"] = C.TARGET_CLASS
    train.loc[train["poisoned"], "classID"] = target_id
    return train
