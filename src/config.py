"""
Central configuration for NeuralProject.

Every path and hyperparameter lives here so that the other modules stay short
and so that you only have to change ONE file to run on a different machine.

Paths
-----
The project root is detected automatically (the folder that contains `src/`).
You can override it with an environment variable, which is handy on Colab:

    import os
    os.environ["NEURALPROJECT_ROOT"] = "/content/drive/MyDrive/NeuralProject"

`NEURALPROJECT_OUTPUT_DIR` can redirect all outputs (used by the smoke tests so
that test runs never mix with real results).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (no hard-coded Windows paths - everything is relative to the root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("NEURALPROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "delulu.csv"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = Path(os.environ.get("NEURALPROJECT_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = METRICS_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
FEATURES_DIR = OUTPUT_DIR / "features"  # cached frozen-encoder features (Phase 4)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "benchmark_id", "image_tag", "language", "file_path", "hallucination_type",
    "prompt", "suffix", "golden_completion", "hallucinated_completion",
    "error_message", "metadata", "license", "repo_url",
]

# Columns that reveal the label (or identify the item) and must NEVER be model input.
# They stay in the CSV files for analysis/grouping only.
LEAKY_COLUMNS = [
    "hallucination_type", "error_message", "benchmark_id", "image_tag",
    "repo_url", "license", "metadata", "file_path", "context_id",
]

# The ONLY columns a model is allowed to read (see preprocessing.check_no_leakage).
MODEL_INPUT_COLUMNS = ["language", "prompt", "suffix", "completion"]

# Binary task: 1 (hallucinated) is the POSITIVE class for every metric.
LABEL_NAMES = {0: "correct", 1: "hallucinated"}
POSITIVE_LABEL = 1

# Four-class task (Task B) - only hallucinated completions are used.
CATEGORY_NAMES = ["import", "method", "parameter", "undefinedvariable"]

# ---------------------------------------------------------------------------
# Train / validation / test split (done on ORIGINAL rows, before pairing)
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Input representation
# ---------------------------------------------------------------------------
# Prompts can be >20k characters. We only keep the code closest to the
# completion: the END of the prompt and the START of the suffix.
PROMPT_TAIL_CHARS = 3000
SUFFIX_HEAD_CHARS = 1500
# Share of the leftover token budget given to the prompt (the rest goes to suffix).
PROMPT_BUDGET_SHARE = 0.7

# ---------------------------------------------------------------------------
# Phase 2 - TF-IDF baselines
# ---------------------------------------------------------------------------
BASELINE_CONFIG = {
    # Text "views" tried for the TF-IDF features (chosen on validation):
    #   completion          -> language + completion only
    #   completion_context  -> completion + a short window of surrounding code
    "views": ["completion", "completion_context"],
    "context_prompt_chars": 500,
    "context_suffix_chars": 200,
    "feature_types": ["word", "char", "word+char"],
    "logreg_C": [0.1, 1.0, 10.0, 100.0],
    "svm_C": [0.01, 0.1, 1.0, 10.0],
    "selection_metric": "macro_f1",  # measured on the VALIDATION split
}

# ---------------------------------------------------------------------------
# Phase 3 - RNNs (GRU / LSTM)
# ---------------------------------------------------------------------------
RNN_CONFIG = {
    "max_len": 400,              # total tokens per example
    "completion_max_tokens": 128,
    "min_freq": 2,               # vocabulary: keep tokens seen >= 2 times in TRAIN
    "max_vocab_size": 30000,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "bidirectional": True,
    "dropout": 0.3,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 20,
    "patience": 4,               # early stopping on validation loss
}

# ---------------------------------------------------------------------------
# Phases 4 & 5 - pretrained code models
# ---------------------------------------------------------------------------
PRETRAINED_MODELS = {
    "codebert": "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
}

TRANSFORMER_INPUT_CONFIG = {
    "max_len": 512,              # CodeBERT/GraphCodeBERT maximum
    "completion_max_tokens": 160,
}

# Phase 4: frozen encoder + small trainable head
FROZEN_CONFIG = {
    "extract_batch_size": 32,    # forward passes only, fits a T4 in fp16
    "head_hidden_dim": 256,
    "head_dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 10,
}

# Phase 5: unfreeze ONLY the final N encoder layers + the head
UNFREEZE_LAST_N_LAYERS = 2

FINETUNE_CONFIG = {
    "unfreeze_last_n_layers": UNFREEZE_LAST_N_LAYERS,
    "encoder_learning_rate": 2e-5,   # small LR for pretrained layers
    "head_learning_rate": 1e-4,      # larger LR for the new head
    "weight_decay": 0.01,
    "batch_size": 16,
    "grad_accumulation_steps": 2,    # effective batch size = 32
    "warmup_fraction": 0.1,
    "max_epochs": 10,
    "patience": 3,
    "max_grad_norm": 1.0,
    "init_head_from_frozen": True,   # start from the Phase 4 head if it exists
}

# ---------------------------------------------------------------------------
# Model registry - fixed order used in every comparison table and chart
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "gru": "GRU",
    "lstm": "LSTM",
    "codebert_frozen": "CodeBERT (frozen)",
    "codebert_finetuned_last2": "CodeBERT (last 2 layers)",
    "graphcodebert_frozen": "GraphCodeBERT (frozen)",
    "graphcodebert_finetuned_last2": "GraphCodeBERT (last 2 layers)",
}
