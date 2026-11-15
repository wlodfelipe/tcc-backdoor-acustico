"""AST pretrained on AudioSet, adapted to 4 s inputs and 10 classes (Section 3.4, D7)."""

import torch
from torch import nn

from . import config as C


def crop_time_pos_embed(pos: torch.Tensor, f_dim: int, t_old: int, t_new: int) -> torch.Tensor:
    """Keep the centre t_new time columns of the patch grid, as the original AST code does
    for inputs shorter than 1024 frames. The two leading tokens (cls, distillation) are kept.
    Patches are ordered frequency-major (index = f * t + t_i)."""
    assert t_new <= t_old
    d = pos.shape[-1]
    grid = pos[:, 2:].reshape(1, f_dim, t_old, d)
    start = t_old // 2 - t_new // 2
    grid = grid[:, :, start : start + t_new]
    return torch.cat([pos[:, :2], grid.reshape(1, f_dim * t_new, d)], dim=1)


def adapt(model, frames: int = C.AST_FRAMES, num_labels: int = C.NUM_CLASSES, seed: int = 0):
    """In place: crop positional embeddings to `frames` and replace the output layer.
    (transformers' ignore_mismatched_sizes would re-initialize the positional embeddings
    instead of cropping them.)"""
    emb = model.audio_spectrogram_transformer.embeddings
    f_dim, t_old = emb.get_shape(model.config)
    model.config.max_length = frames
    _, t_new = emb.get_shape(model.config)
    emb.position_embeddings = nn.Parameter(
        crop_time_pos_embed(emb.position_embeddings.data, f_dim, t_old, t_new).contiguous()
    )
    torch.manual_seed(seed)
    model.classifier.dense = nn.Linear(model.config.hidden_size, num_labels)
    model.config.num_labels = num_labels
    model.num_labels = num_labels
    return model


def build(seed: int, checkpoint: str = C.AST_CHECKPOINT):
    from transformers import ASTForAudioClassification
    return adapt(ASTForAudioClassification.from_pretrained(checkpoint), seed=seed)
