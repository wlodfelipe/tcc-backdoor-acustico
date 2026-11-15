import numpy as np
import pandas as pd
import pytest

from tcc import config as C
from tcc import poison, psycho, trigger

rng = np.random.default_rng(0)


def noise(seconds=1.0, amp=0.1):
    return (amp * rng.standard_normal(int(C.SR * seconds))).astype(np.float32)


def test_trigger_snr_is_exactly_40db():
    x = noise()
    assert trigger.snr_db(x, trigger.make_trigger(x)) == pytest.approx(C.SNR_DB, abs=1e-3)


@pytest.mark.parametrize("snr", C.SNR_VALUES)
def test_trigger_snr_is_exact_for_every_design_value(snr):
    x = noise()
    length = len(x)
    padded = np.zeros(C.N_SAMPLES, np.float32)
    padded[:length] = x
    y = trigger.poison_clip(padded, length, snr)
    assert trigger.snr_db(x, y[:length] - x) == pytest.approx(snr, abs=1e-3)
    assert not y[length:].any()


def test_trigger_follows_envelope():
    x = np.concatenate([noise(0.5), np.zeros(C.SR // 2, np.float32)])
    t = trigger.make_trigger(x)
    assert np.abs(t[: C.SR // 2 - 400]).max() > 0
    assert np.abs(t[C.SR // 2 + 400 :]).max() == 0  # silent where the host is silent


def test_trigger_on_silence_is_zero():
    assert not trigger.make_trigger(np.zeros(1000, np.float32)).any()


def test_poison_leaves_padding_silent():
    length = C.SR
    x = np.zeros(C.N_SAMPLES, np.float32)
    x[:length] = noise()
    y = trigger.poison_clip(x, length)
    assert not y[length:].any()
    assert not np.array_equal(y[:length], x[:length])


def test_envelope_peak_is_one():
    assert trigger.rms_envelope(noise()).max() == pytest.approx(1.0)


def fake_meta():
    rows = []
    for i in range(7079):
        cls = "street_music" if i < 800 else ("gun_shot" if i < 1111 else "dog_bark")
        rows.append(dict(slice_file_name=f"{i}.wav", fold=1, classID=9 if cls == "street_music" else 6 if cls == "gun_shot" else 3, **{"class": cls}, split="train"))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("rate,expected", [(0.005, 35), (0.01, 71), (0.02, 142), (0.05, 354)])
def test_poison_counts_match_notes(rate, expected):
    assert poison.n_poison(7079, rate) == expected


def test_selection_excludes_target_and_is_seeded():
    meta = fake_meta()
    a = poison.select_poison(meta, 0.05, 1)
    b = poison.select_poison(meta, 0.05, 1)
    assert (a["orig_class"] != C.TARGET_CLASS).all()
    assert a["slice_file_name"].is_unique
    assert a.equals(b)


def test_apply_labels_relabels_only_selected():
    meta = fake_meta()
    man = poison.build_manifest(meta)
    tr = poison.apply_labels(meta, man, 0.02, 3)
    assert tr["poisoned"].sum() == 142
    assert (tr.loc[tr["poisoned"], "class"] == C.TARGET_CLASS).all()
    assert (tr.loc[tr["poisoned"], "classID"] == 9).all()
    assert (tr.loc[~tr["poisoned"], "class"] == meta["class"][~tr["poisoned"]]).all()


def test_masking_model_sanity():
    host = noise(1.0, 0.3)
    n = np.arange(len(host))
    faint = (1e-4 * np.sin(2 * np.pi * 1000 * n / C.SR)).astype(np.float32)
    loud = (0.3 * np.sin(2 * np.pi * 1000 * n / C.SR)).astype(np.float32)
    assert psycho.masking_check(host, faint)["masked_energy_frac"] > 0.99
    assert psycho.masking_check(host, loud)["masked_energy_frac"] < 0.01
