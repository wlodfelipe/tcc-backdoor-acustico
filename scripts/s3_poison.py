"""Stage 2c: poison sets (4 rates x 5 seeds) and unit checks.

The manifest is drawn once and committed (results/poison_manifest.csv). If it exists it is read,
never overwritten; a fresh draw is made only to report whether this numpy reproduces it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tcc import config as C
from tcc import data, poison, trigger


def main():
    meta = data.load_metadata()
    fresh = poison.build_manifest(meta)
    if C.MANIFEST_PATH.exists():
        manifest = pd.read_csv(C.MANIFEST_PATH)
    else:
        manifest = fresh
        manifest.to_csv(C.MANIFEST_PATH, index=False)
    same_draw = bool(
        (manifest[["rate", "seed", "slice_file_name"]].reset_index(drop=True)
         == fresh[["rate", "seed", "slice_file_name"]].reset_index(drop=True)).all().all()
    ) if len(manifest) == len(fresh) else False
    out_dir = C.RESULTS_DIR / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = meta[meta.split == "train"]
    n_train = len(train)
    n_gs = int((train["class"] == C.SOURCE_CLASS).sum())
    waves, lengths = data.open_cache()
    name_to_row = {n: i for i, n in enumerate(meta.slice_file_name)}

    rows = []
    for (rate, seed), g in manifest.groupby(["rate", "seed"]):
        labelled = poison.apply_labels(meta, manifest, rate, seed)
        n_p = int(labelled.poisoned.sum())
        gs_p = int((g.orig_class == C.SOURCE_CLASS).sum())
        # checks: size, target excluded, relabelling, clean rows untouched, splits untouched
        assert n_p == poison.n_poison(n_train, rate) == len(g)
        assert (g.orig_class != C.TARGET_CLASS).all()
        assert (labelled.loc[labelled.poisoned, "class"] == C.TARGET_CLASS).all()
        clean = labelled[~labelled.poisoned]
        assert (clean["class"].values == train.loc[clean.index, "class"].values).all()
        assert set(g.slice_file_name) <= set(train.slice_file_name)
        # every selected clip actually changes when poisoned, and its padding stays silent
        for name in g.slice_file_name:
            i = name_to_row[name]
            n = int(lengths[i])
            y = trigger.poison_clip(np.asarray(waves[i]), n)
            assert not np.array_equal(y[:n], waves[i, :n]) and not y[n:].any()
        rows.append({
            "rate": rate, "seed": seed, "n_poisoned": n_p,
            "gun_shot_poisoned": gs_p,
            "gun_shot_frac_of_class_pct": round(100 * gs_p / n_gs, 2),
            "street_music_after": int((labelled["class"] == C.TARGET_CLASS).sum()),
            "by_orig_class": g.orig_class.value_counts().to_dict(),
        })

    # the same clip set must not repeat across seeds of one rate
    for rate, g in manifest.groupby("rate"):
        sets = [frozenset(s.slice_file_name) for _, s in g.groupby("seed")]
        assert len(set(sets)) == len(sets)

    report = {"manifest": str(C.MANIFEST_PATH), "fresh_draw_matches_manifest": same_draw, "n_train": n_train, "gun_shot_train": n_gs, "all_checks_pass": True, "sets": rows}
    (out_dir / "s3_poison.json").write_text(json.dumps(report, indent=2, default=C.json_default))
    for r in rows:
        print(r["rate"], r["seed"], r["n_poisoned"], "gun_shot:", r["gun_shot_poisoned"],
              f'({r["gun_shot_frac_of_class_pct"]}% of class)', "street_music after:", r["street_music_after"])
    print("all checks pass; fresh draw matches manifest:", same_draw)


if __name__ == "__main__":
    main()
