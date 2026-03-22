import os
import torch

SMOKE_TEST = os.environ.get("SMOKE_TEST")

DIMENSION = 2
LOWER_BOUND = [-0.5, -0.5]
UPPER_BOUND = [0.5, 0.5]
BATCH_SIZE = 5
N_BATCH = 30 if not SMOKE_TEST else 5
N_INITIAL = 20 if not SMOKE_TEST else 4
MC_SAMPLES = 1024 if not SMOKE_TEST else 16
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
TOP_CANDIDATES = 5
SEED = 42

TKWARGS: dict = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
