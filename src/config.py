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
    # Added later (src/additional_pretrained_models.py):
    "codet5_small": "Salesforce/codet5-small",
    "unixcoder": "microsoft/unixcoder-base",
}

# How each pretrained checkpoint is loaded and fed. The models do NOT share one
# architecture, so nothing below assumes CodeBERT's layer names:
#
#   architecture "roberta"    -> transformers RobertaModel (CodeBERT, GraphCodeBERT,
#                                UniXcoder): embeddings -> encoder.layer[0..11]
#   architecture "t5_encoder" -> transformers T5EncoderModel (encoder half of CodeT5;
#                                the decoder is not needed for classification):
#                                shared embeddings -> encoder.block[0..5]
#                                -> encoder.final_layer_norm
#
#   prefix_tokens -> extra special tokens placed right after <s>. UniXcoder was
#                    pretrained with a mode token; "<encoder-only>" selects its
#                    bidirectional (BERT-like) mode, exactly as in Microsoft's
#                    reference code:  <s> <encoder-only> </s> ...
#   use_amp       -> fp16 mixed precision on the GPU. T5 models are known to overflow
#                    (inf/NaN) in fp16, so CodeT5 runs in fp32 (it is small: ~35M
#                    encoder parameters, so fp32 is still fast on a T4).
ENCODER_SPECS = {
    "codebert": {"architecture": "roberta", "prefix_tokens": [], "use_amp": True},
    "graphcodebert": {"architecture": "roberta", "prefix_tokens": [], "use_amp": True},
    "codet5_small": {"architecture": "t5_encoder", "prefix_tokens": [], "use_amp": False},
    "unixcoder": {"architecture": "roberta", "prefix_tokens": ["<encoder-only>"], "use_amp": True},
}

TRANSFORMER_INPUT_CONFIG = {
    "max_len": 512,              # CodeBERT/GraphCodeBERT maximum (also used for CodeT5 / UniXcoder)
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

# Per-model changes to FINETUNE_CONFIG (everything not listed is shared).
# CodeBERT and GraphCodeBERT have no entry, so their experiments are unchanged.
FINETUNE_MODEL_OVERRIDES = {
    # CodeT5-small is ~4x smaller than CodeBERT and T5-style models are usually
    # fine-tuned with a larger learning rate (the CodeT5 paper uses 5e-5 and above).
    "codet5_small": {"encoder_learning_rate": 5e-5},
    # UniXcoder has the same RoBERTa-base shape as CodeBERT -> same recipe.
    "unixcoder": {},
}


def get_finetune_config(model_key):
    """FINETUNE_CONFIG with the model-specific overrides applied (a new dict)."""
    return {**FINETUNE_CONFIG, **FINETUNE_MODEL_OVERRIDES.get(model_key, {})}


# Extra experiment: CodeBERT with the LAST FOUR encoder layers unfrozen
# (src/more_finetuning_codebert.py). Uses exactly FINETUNE_CONFIG otherwise,
# so the only difference to the last-2 experiment is the number of layers.
UNFREEZE_LAST_N_LAYERS_EXTENDED = 4

# Extra experiment: FROZEN CodeBERT + one NEW Transformer layer + head
# (src/adding_layers_codebert.py). The new layer copies CodeBERT's layer shape
# (hidden size, heads, feed-forward size, GELU, post-LayerNorm) from the encoder config.
EXTRA_LAYER_CONFIG = {
    "num_new_layers": 1,
    "dropout": 0.1,
    "learning_rate": 1e-4,           # new layer + head are both trained from scratch
    "weight_decay": 0.01,
    "batch_size": 16,
    "grad_accumulation_steps": 2,    # effective batch size = 32 (same as Phase 5)
    "warmup_fraction": 0.1,
    "max_epochs": 10,
    "patience": 3,
    "max_grad_norm": 1.0,
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
    # Additional experiments
    "codet5_small_frozen": "CodeT5-small (frozen)",
    "codet5_small_finetuned_last2": "CodeT5-small (last 2 layers)",
    "unixcoder_frozen": "UniXcoder (frozen)",
    "unixcoder_finetuned_last2": "UniXcoder (last 2 layers)",
    "codebert_finetuned_last4": "CodeBERT (last 4 layers)",
    "codebert_extra_transformer_layer": "CodeBERT (frozen + new Transformer layer)",
}

# What was trained in each experiment (shown in outputs/classification_report.md).
MODEL_TRAINING_SETUP = {
    "logistic_regression": "Traditional baseline - TF-IDF features + Logistic Regression (no neural network)",
    "linear_svm": "Traditional baseline - TF-IDF features + Linear SVM (no neural network)",
    "gru": "Trained from scratch - bidirectional GRU (all weights trained)",
    "lstm": "Trained from scratch - bidirectional LSTM (all weights trained)",
    "codebert_frozen": "Frozen encoder - all 12 CodeBERT layers frozen, only the classification head trained",
    "codebert_finetuned_last2": "Partially fine-tuned - CodeBERT layers 10-11 + head trained, embeddings and layers 0-9 frozen",
    "graphcodebert_frozen": "Frozen encoder - all 12 GraphCodeBERT layers frozen, only the classification head trained",
    "graphcodebert_finetuned_last2": "Partially fine-tuned - GraphCodeBERT layers 10-11 + head trained, embeddings and layers 0-9 frozen",
    "codet5_small_frozen": "Frozen encoder - all 6 CodeT5-small encoder blocks frozen, only the classification head trained",
    "codet5_small_finetuned_last2": "Partially fine-tuned - CodeT5-small encoder blocks 4-5 + final encoder LayerNorm + head trained, embeddings and blocks 0-3 frozen",
    "unixcoder_frozen": "Frozen encoder - all 12 UniXcoder layers frozen, only the classification head trained",
    "unixcoder_finetuned_last2": "Partially fine-tuned - UniXcoder layers 10-11 + head trained, embeddings and layers 0-9 frozen",
    "codebert_finetuned_last4": "Partially fine-tuned (last FOUR layers) - CodeBERT layers 8-11 + head trained, embeddings and layers 0-7 frozen",
    "codebert_extra_transformer_layer": "Added trainable layer - all of CodeBERT frozen; one NEW randomly initialised Transformer layer + head trained",
}

# The comparison and the top-two selection rank models by this TEST metric.
# F1 of the positive class (1 = hallucinated) is the metric used in the existing
# comparison (comparison_summary.md picks the best model of each group by F1).
PRIMARY_METRIC = "f1"
PRIMARY_METRIC_TIEBREAK = "roc_auc"
