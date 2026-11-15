import json

import numpy as np
import pandas as pd
import pytest
import torch

from tcc import config as C
from tcc import features, model, poison, train


def tiny_ast(seed=0):
    from transformers import ASTConfig, ASTForAudioClassification
    cfg = ASTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                    max_length=1024, num_mel_bins=128, num_labels=527)
    torch.manual_seed(123)
    return model.adapt(ASTForAudioClassification(cfg), seed=seed)


def test_pos_embed_crop_is_centre_of_time_axis():
    f, t_old, t_new, d = 12, 101, 39, 4
    pos = torch.arange((2 + f * t_old) * d, dtype=torch.float32).reshape(1, 2 + f * t_old, d)
    out = model.crop_time_pos_embed(pos, f, t_old, t_new)
    assert out.shape == (1, 2 + f * t_new, d)
    grid_old = pos[:, 2:].reshape(1, f, t_old, d)
    grid_new = out[:, 2:].reshape(1, f, t_new, d)
    assert torch.equal(grid_new, grid_old[:, :, 31:70])  # 101//2 - 39//2 = 31
    assert torch.equal(out[:, :2], pos[:, :2])


def test_adapt_keeps_pretrained_values_and_runs_on_400_frames():
    from transformers import ASTConfig, ASTForAudioClassification
    cfg = ASTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    m = ASTForAudioClassification(cfg)
    before = m.audio_spectrogram_transformer.embeddings.position_embeddings.data.clone()
    model.adapt(m)
    after = m.audio_spectrogram_transformer.embeddings.position_embeddings.data
    assert after.shape[1] == 2 + 12 * 39
    assert torch.equal(after, model.crop_time_pos_embed(before, 12, 101, 39))
    logits = m(input_values=torch.randn(2, C.AST_FRAMES, 128)).logits
    assert logits.shape == (2, C.NUM_CLASSES)


def test_new_head_is_seeded():
    a = tiny_ast(seed=3).classifier.dense.weight
    b = tiny_ast(seed=3).classifier.dense.weight
    c = tiny_ast(seed=4).classifier.dense.weight
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_features_shape_and_trigger_changes_them():
    rng = np.random.default_rng(0)
    w = np.zeros((2, C.N_SAMPLES), np.float32)
    w[:, : C.SR * 2] = 0.1 * rng.standard_normal((2, C.SR * 2))
    clean, trig = features.build_all(w, np.array([C.SR * 2, C.SR * 2]), snr_db=40.0)
    assert clean.shape == (2, C.AST_FRAMES, 128) and clean.dtype == np.float16
    assert np.isfinite(clean).all()
    assert not np.array_equal(clean, trig)


def test_early_stopping():
    es = train.EarlyStopping(patience=2)
    for epoch, v in enumerate([0.5, 0.6, 0.6, 0.55], start=1):
        es.step(epoch, v)
    assert es.best_epoch == 2 and es.stop


def test_attack_metrics():
    labels = np.array([6, 6, 6, 6, 9, 0])
    oh = lambda p: np.eye(10)[p]
    clean = oh(np.array([6, 6, 6, 9, 9, 0]))
    trig = oh(np.array([9, 9, 9, 6, 9, 9]))
    m = train.attack_metrics(clean, trig, labels, source_id=6, target_id=9)
    assert m["acc_clean"] == pytest.approx(5 / 6)
    assert m["triggered_asr"] == pytest.approx(3 / 4)
    assert m["clean_asr"] == pytest.approx(1 / 4)
    assert m["triggered_asr_all_nontarget"] == pytest.approx(4 / 5)


def small_meta():
    classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
               "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    rows = []
    for i in range(120):
        cid = i % 10
        fold = 10 if i >= 100 else 9 if i >= 80 else 1
        rows.append({"slice_file_name": f"{i}.wav", "fold": fold, "classID": cid, "class": classes[cid]})
    meta = pd.DataFrame(rows)
    meta["split"] = np.select([meta.fold == 1, meta.fold == 9, meta.fold == 10], ["train", "val", "test"], "")
    return meta


def test_assemble_reference_and_poisoned():
    meta = small_meta()
    man = poison.select_poison(meta, 0.05, 0)
    rows, y, p = train.assemble(meta, man, train.RunSpec(0.0, 0))
    assert len(rows) == 80 and not p.any()
    rows, y, p = train.assemble(meta, man, train.RunSpec(0.05, 0))
    assert p.sum() == 4 and (y[p] == 9).all()


def test_smoke_run_writes_results(tmp_path):
    meta = small_meta()
    man = poison.select_poison(meta, 0.05, 0)
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((len(meta), C.AST_FRAMES, 128)).astype(np.float16)
    trig = clean + np.float16(0.5)
    res = train.run(train.RunSpec(0.05, 0, 20.0), meta, man, clean, trig, tmp_path, torch.device("cpu"),
                    tiny_ast, max_epochs=2, patience=5, bs=16)
    assert res["epochs_run"] == 2 and res["n_poisoned"] == 4 and res["snr_db"] == 20.0
    assert set(res["test"]) >= {"acc_clean", "triggered_asr", "clean_asr"}
    # per-epoch validation Triggered ASR is logged (monitoring only)
    for h in res["history"]:
        assert {"val_triggered_asr", "val_clean_asr", "val_triggered_asr_all_nontarget"} <= set(h)
        assert 0.0 <= h["val_triggered_asr"] <= 1.0
    saved = json.loads((tmp_path / "rate5pct_snr20_seed0.json").read_text())
    assert saved["best_epoch"] in (1, 2)
    z = np.load(tmp_path / "rate5pct_snr20_seed0_test_logits.npz", allow_pickle=True)
    assert z["logits_clean"].shape == (20, 10) and z["logits_triggered"].shape == (20, 10)


def test_runspec_names_include_snr():
    assert train.RunSpec(0.05, 3, 20.0).name == "rate5pct_snr20_seed3"
    assert train.RunSpec(0.005, 0, 0.0).name == "rate0.5pct_snr0_seed0"
    assert train.RunSpec(0.0, 1, 30.0).name == "ref_snr30_seed1"
    assert train.RunSpec(0.05, 0).snr_db == C.SNR_DB


def test_checkpoint_selection_ignores_triggered_asr(tmp_path):
    """Selection uses clean validation accuracy only: identical clean features and seed give the
    same best epoch whatever the triggered features are."""
    meta = small_meta()
    man = poison.select_poison(meta, 0.05, 0)
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((len(meta), C.AST_FRAMES, 128)).astype(np.float16)
    trig = clean.copy()
    a = train.run(train.RunSpec(0.05, 0), meta, man, clean, trig, tmp_path / "a", torch.device("cpu"),
                  tiny_ast, max_epochs=2, bs=16)
    trig_val = trig.copy()
    val_rows = meta.index[meta["split"] != "train"]
    trig_val[val_rows] = rng.standard_normal((len(val_rows), C.AST_FRAMES, 128)).astype(np.float16)
    b = train.run(train.RunSpec(0.05, 0), meta, man, clean, trig_val, tmp_path / "b", torch.device("cpu"),
                  tiny_ast, max_epochs=2, bs=16)
    assert [h["val_acc"] for h in a["history"]] == [h["val_acc"] for h in b["history"]]
    assert a["best_epoch"] == b["best_epoch"]
    assert a["test"]["acc_clean"] == b["test"]["acc_clean"]


def test_feature_cache_is_keyed_by_snr(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    w = np.zeros((2, C.N_SAMPLES), np.float32)
    w[:, : C.SR] = 0.1 * rng.standard_normal((2, C.SR))
    lengths = np.array([C.SR, C.SR])
    f40 = features.cached(w, lengths, 40.0, tmp_path)
    f0 = features.cached(w, lengths, 0.0, tmp_path)
    fc = features.cached(w, lengths, None, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["feats_clean.npy", "feats_trig_snr0.npy", "feats_trig_snr40.npy"]
    # stronger trigger moves the features further from clean
    d = lambda f: np.abs(f.astype(np.float32) - fc.astype(np.float32)).mean()
    assert d(f0) > d(f40) > 0
    # second call reads the cache instead of recomputing
    monkeypatch.setattr(features, "build", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    assert np.array_equal(features.cached(w, lengths, 40.0, tmp_path), f40)


def test_cross_snr_evaluation_and_confusion(tmp_path):
    meta = small_meta()
    man = poison.select_poison(meta, 0.05, 0)
    rng = np.random.default_rng(2)
    clean = rng.standard_normal((len(meta), C.AST_FRAMES, 128)).astype(np.float16)
    trig10 = clean + np.float16(0.5)
    trig40 = clean + np.float16(0.05)
    res = train.run(train.RunSpec(0.05, 0, 10.0), meta, man, clean, trig10, tmp_path, torch.device("cpu"),
                    tiny_ast, max_epochs=1, bs=16, eval_trigs={40.0: trig40, 10.0: trig10})
    # the selected-SNR entry is the main result; other SNRs are extra
    assert list(res["test_by_trigger_snr"]) == ["40", "10"]
    assert res["test_by_trigger_snr"]["10"] == res["test"]
    z = np.load(tmp_path / "rate5pct_snr10_seed0_test_logits.npz", allow_pickle=True)
    assert np.array_equal(z["logits_triggered_snr10"], z["logits_triggered"])
    assert z["logits_triggered_snr40"].shape == (20, 10)
    # confusion matrices count every test clip, rows = true class
    cm = np.array(res["test_confusion_clean"])
    assert cm.shape == (10, 10) and cm.sum() == 20 and (cm.sum(1) == 2).all()
    assert np.trace(cm) / 20 == pytest.approx(res["test"]["acc_clean"])
    assert np.array(res["test_confusion_triggered"]).sum() == 20
    # per-rate exhaustion check: poisoned gun_shot as a fraction of the class
    n_gs = int(((meta.split == "train") & (meta["class"] == C.SOURCE_CLASS)).sum())
    assert res["gun_shot_poisoned_frac_of_class"] == pytest.approx(res["gun_shot_poisoned"] / n_gs)
