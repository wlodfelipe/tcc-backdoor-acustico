"""One training run (Sections 3.4-3.6): fine-tune, early stopping on clean validation
accuracy, then raw test logits on clean and triggered clips. Triggered ASR on validation is
logged every epoch for monitoring only; it never takes part in checkpoint selection."""

from dataclasses import dataclass
import copy
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config as C
from . import poison


@dataclass(frozen=True)
class RunSpec:
    rate: float  # 0.0 = reference model (no poisoning)
    seed: int
    snr_db: float = C.SNR_DB  # trigger used for poisoning and for the triggered evaluations

    @property
    def name(self) -> str:
        kind = "ref" if self.rate == 0 else f"rate{self.rate * 100:g}pct"
        return f"{kind}_snr{self.snr_db:g}_seed{self.seed}"


def assemble(meta: pd.DataFrame, manifest: pd.DataFrame, spec: RunSpec):
    """Row indices into meta, labels and whether to use the triggered features."""
    if spec.rate == 0:
        train = meta[meta["split"] == "train"].copy()
        train["poisoned"] = False
    else:
        train = poison.apply_labels(meta, manifest, spec.rate, spec.seed)
    return train.index.to_numpy(), train["classID"].to_numpy(), train["poisoned"].to_numpy()


class EarlyStopping:
    """Stops after `patience` epochs without a strict improvement; the first best epoch wins."""

    def __init__(self, patience: int):
        self.patience, self.best, self.best_epoch, self.bad = patience, -np.inf, -1, 0

    def step(self, epoch: int, value: float) -> bool:
        if value > self.best:
            self.best, self.best_epoch, self.bad = value, epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad >= self.patience


def attack_metrics(logits_clean, logits_trig, labels, source_id: int, target_id: int) -> dict:
    """Equation (3) under both conditions, plus clean accuracy (for Equation 4)."""
    pc, pt = logits_clean.argmax(1), logits_trig.argmax(1)
    src = labels == source_id
    nontarget = labels != target_id
    return {
        "acc_clean": float((pc == labels).mean()),
        "triggered_asr": float((pt[src] == target_id).mean()),
        "clean_asr": float((pc[src] == target_id).mean()),
        "n_source_test": int(src.sum()),
        # all-to-one view: every non-target test clip with the trigger
        "triggered_asr_all_nontarget": float((pt[nontarget] == target_id).mean()),
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _batches(n: int, bs: int, gen: torch.Generator | None):
    order = torch.randperm(n, generator=gen).numpy() if gen is not None else np.arange(n)
    for s in range(0, n, bs):
        yield order[s : s + bs]


@torch.no_grad()
def predict(model, feats: np.ndarray, rows: np.ndarray, device, bs: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for b in _batches(len(rows), bs, None):
        x = torch.from_numpy(feats[rows[b]].astype(np.float32)).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(input_values=x).logits.float().cpu().numpy())
    return np.concatenate(out)


def confusion(pred: np.ndarray, labels: np.ndarray, n: int = C.NUM_CLASSES) -> list:
    """Rows = true class, columns = predicted class (classID order)."""
    m = np.zeros((n, n), dtype=int)
    np.add.at(m, (labels, pred), 1)
    return m.tolist()


def run(spec: RunSpec, meta, manifest, feats_clean, feats_trig, out_dir: Path, device,
        model_factory, max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE, lr=C.LR, bs=C.BATCH_SIZE,
        extra_info: dict | None = None, eval_trigs: dict | None = None, select: str = "best") -> dict:
    """eval_trigs: {snr_db: triggered features} for the additional cross-SNR evaluation of the
    trained model (inference only; it does not affect training or checkpoint selection)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(spec.seed)
    tr_rows, tr_y, tr_poisoned = assemble(meta, manifest, spec)
    va_rows = meta.index[meta["split"] == "val"].to_numpy()
    te_rows = meta.index[meta["split"] == "test"].to_numpy()
    va_y = meta.loc[va_rows, "classID"].to_numpy()
    te_y = meta.loc[te_rows, "classID"].to_numpy()
    src_id = int(meta.loc[meta["class"] == C.SOURCE_CLASS, "classID"].iloc[0])
    tgt_id = int(meta.loc[meta["class"] == C.TARGET_CLASS, "classID"].iloc[0])

    model = model_factory(spec.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(spec.seed)
    stopper = EarlyStopping(patience)
    best_state, history = None, []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in _batches(len(tr_rows), bs, gen):
            r = tr_rows[b]
            x = np.where(tr_poisoned[b][:, None, None], feats_trig[r], feats_clean[r]).astype(np.float32)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(tr_y[b]).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(input_values=x).logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(b)
            n += len(b)
        train_s = time.time() - t0
        va_logits = predict(model, feats_clean, va_rows, device)
        va_loss = float(loss_fn(torch.tensor(va_logits), torch.tensor(va_y)).item())
        va_acc = float((va_logits.argmax(1) == va_y).mean())
        va_trig = attack_metrics(va_logits, predict(model, feats_trig, va_rows, device), va_y, src_id, tgt_id)
        improved = stopper.step(epoch, va_acc)
        if improved:
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": va_loss, "val_acc": va_acc,
                        "val_triggered_asr": va_trig["triggered_asr"], "val_clean_asr": va_trig["clean_asr"],
                        "val_triggered_asr_all_nontarget": va_trig["triggered_asr_all_nontarget"],
                        "train_seconds": round(train_s, 1), "epoch_seconds": round(time.time() - t0, 1)})
        print(f"[{spec.name}] epoch {epoch} train_loss {total / n:.4f} val_acc {va_acc:.4f} "
              f"val_loss {va_loss:.4f} val_trig_asr {va_trig['triggered_asr']:.3f} {time.time() - t0:.0f}s{' *' if improved else ''}", flush=True)
        if stopper.stop:
            break

    # select="last": fixed-horizon check (no early stopping), the model of the last epoch is evaluated
    if select == "best":
        model.load_state_dict(best_state)
    va_lc = predict(model, feats_clean, va_rows, device)
    va_lt = predict(model, feats_trig, va_rows, device)
    np.savez_compressed(out_dir / f"{spec.name}_val_logits.npz", logits_clean=va_lc, logits_triggered=va_lt,
                        labels=va_y, slice_file_name=meta.loc[va_rows, "slice_file_name"].to_numpy())
    lc = predict(model, feats_clean, te_rows, device)
    lt = predict(model, feats_trig, te_rows, device)
    metrics = attack_metrics(lc, lt, te_y, src_id, tgt_id)
    # additional analysis: the same trained model activated by triggers of other strengths
    by_snr, extra_logits = {}, {}
    for snr, feats in sorted((eval_trigs or {}).items(), reverse=True):
        l = lt if snr == spec.snr_db else predict(model, feats, te_rows, device)
        by_snr[f"{snr:g}"] = attack_metrics(lc, l, te_y, src_id, tgt_id)
        extra_logits[f"logits_triggered_snr{snr:g}"] = l
    np.savez_compressed(out_dir / f"{spec.name}_test_logits.npz", logits_clean=lc, logits_triggered=lt,
                        labels=te_y, slice_file_name=meta.loc[te_rows, "slice_file_name"].to_numpy(),
                        **extra_logits)
    n_gs_train = int((meta.loc[tr_rows, "class"].to_numpy() == C.SOURCE_CLASS).sum())
    gs_poisoned = int((tr_poisoned & (meta.loc[tr_rows, "class"].to_numpy() == C.SOURCE_CLASS)).sum())
    result = {
        "run": spec.name, "rate": spec.rate, "seed": spec.seed, "snr_db": spec.snr_db,
        "n_train": int(len(tr_rows)), "n_poisoned": int(tr_poisoned.sum()),
        "gun_shot_poisoned": gs_poisoned,
        "gun_shot_poisoned_frac_of_class": gs_poisoned / n_gs_train if n_gs_train else 0.0,
        "best_epoch": stopper.best_epoch, "best_val_acc": stopper.best, "epochs_run": len(history),
        "stopped_early": stopper.stop, "selected": select, "history": history,
        "mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
        "test": metrics,
        "test_confusion_clean": confusion(lc.argmax(1), te_y),
        "test_confusion_triggered": confusion(lt.argmax(1), te_y),
        "test_by_trigger_snr": by_snr,  # additional analysis, not part of the H1/H2 decision
        "hyperparameters": {"lr": lr, "batch_size": bs, "max_epochs": max_epochs, "patience": patience,
                            "optimizer": "Adam", "amp_fp16": device.type == "cuda",
                            "checkpoint": C.AST_CHECKPOINT, "frames": C.AST_FRAMES},
        "environment": environment(device), **(extra_info or {}),
    }
    (out_dir / f"{spec.name}.json").write_text(json.dumps(result, indent=2, default=C.json_default))
    return result


def environment(device) -> dict:
    import librosa, transformers
    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "librosa": librosa.__version__, "device": str(device)}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["cuda"] = torch.version.cuda
    try:
        import torchaudio
        env["torchaudio"] = torchaudio.__version__
    except ImportError:
        env["torchaudio"] = None
    return env
