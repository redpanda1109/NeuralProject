"""
Small helper functions shared by every module:
seeding, device detection, folders, logging and JSON saving.
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

from src import config


def set_seed(seed=config.SEED):
    """Make runs reproducible (Python, NumPy and - if installed - PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN is slightly slower but gives repeatable results.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(verbose=True):
    """Return CUDA if a GPU is available, otherwise CPU."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch is not installed. Install it with `pip install torch` "
            "(on Colab it is already installed)."
        ) from error

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[device] No GPU found - using CPU (training will be slow).")
    return device


def ensure_output_dirs():
    """Create every output folder (safe to call many times)."""
    for folder in [config.PROCESSED_DIR, config.FIGURES_DIR, config.METRICS_DIR,
                   config.PREDICTIONS_DIR, config.MODELS_DIR, config.LOGS_DIR,
                   config.FEATURES_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def require_file(path, hint=""):
    """Stop with a clear message if a needed file does not exist."""
    path = Path(path)
    if not path.exists():
        message = f"Required file not found: {path}"
        if hint:
            message += f"\n  -> {hint}"
        raise FileNotFoundError(message)
    return path


def get_logger(name):
    """Logger that prints to the console AND appends to outputs/logs/<name>.log."""
    ensure_output_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured (e.g. re-run in a notebook)
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = logging.FileHandler(config.LOGS_DIR / f"{name}.log", encoding="utf-8")
    log_file.setFormatter(formatter)
    logger.addHandler(log_file)
    logger.propagate = False
    return logger


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(value):
    """Let json.dump handle NumPy numbers and Paths."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot save {type(value)} to JSON")


def markdown_table(df, index=True, float_format=None):
    """DataFrame -> Markdown table (no extra dependency such as `tabulate` needed)."""
    df = df.reset_index() if index else df

    def fmt(value):
        if isinstance(value, float) and np.isnan(value):
            return "n/a"
        if float_format and isinstance(value, float):
            return float_format.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider] + rows)
