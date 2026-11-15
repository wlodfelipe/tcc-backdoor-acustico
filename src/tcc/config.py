"""Experiment parameters. Every value here traces to Section 3 of TCC.md (Quadro 1)."""

from pathlib import Path
import os

# 3.1 Preprocessing
SR = 16_000
CLIP_SECONDS = 4.0
N_SAMPLES = int(SR * CLIP_SECONDS)  # 64,000

# 3.1 Partition (fixed, not 10-fold CV)
TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

SOURCE_CLASS = "gun_shot"
TARGET_CLASS = "street_music"

# 3.2 Trigger
TRIGGER_FREQ_HZ = 1000.0
SNR_DB = 40.0  # default: the original design point (pilot, phase 1)
SNR_VALUES = (40.0, 30.0, 20.0, 10.0, 0.0)  # D10: SNR as an experimental factor
# RMS envelope frames: 25 ms window, 10 ms hop (same framing as the AST input)
RMS_FRAME = 400
RMS_HOP = 160

# 3.3 Poisoning
POISON_RATES = (0.005, 0.01, 0.02, 0.05)
SEEDS = (0, 1, 2, 3, 4)

# 3.7 Detectability: Mel spectrogram close to the AST input (128 bands, 25 ms / 10 ms)
MEL_N_FFT = 512
MEL_WIN = 400
MEL_HOP = 160
MEL_BANDS = 128
MEL_TOP_DB = 80.0  # dynamic range of the spectrogram "image" used for SSIM
H2_SSIM_THRESHOLD = 0.99

# Psychoacoustic check (Qin et al., 2019 procedure)
PSY_N_FFT = 2048
PSY_HOP = 512

# 3.4 Model and training (D7-D9, NOTAS.md)
AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_FRAMES = 400  # 4 s at 10 ms hop; positional embeddings center-cropped in time (D7)
NUM_CLASSES = 10
LR = 1e-5
BATCH_SIZE = 16
MAX_EPOCHS = 30
PATIENCE = 5  # early stopping on clean validation accuracy (D8)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("TCC_RESULTS", REPO_ROOT / "results"))
MANIFEST_PATH = REPO_ROOT / "results" / "poison_manifest.csv"
WAV_HASHES_PATH = REPO_ROOT / "results" / "wav_sha256.csv"
DATA_CACHE_DIR = Path(os.environ.get("TCC_CACHE", REPO_ROOT / "data"))

# Candidate dataset locations: local download, then Kaggle mount. TCC_DATA overrides both.
DATASET_CANDIDATES = [
    Path.home() / "Downloads" / "UrbanSound8K",
    Path("/kaggle/input/datasets/chrisfilo/urbansound8k"),
    Path("/kaggle/input/urbansound8k"),
]


def json_default(o):
    """numpy scalars (pandas < 3 returns them from groupby keys and value_counts)."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
