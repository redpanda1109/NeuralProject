"""
Quick end-to-end check of the whole pipeline on a TINY subset (works on CPU).

Run from the project root (after `python -m src.data_split`):
    python tests/smoke_test.py             # everything, incl. CodeBERT (~500 MB download once)
    python tests/smoke_test.py --skip-pretrained

All outputs go to a TEMPORARY folder, so the smoke test never mixes with (or
overwrites) real results in outputs/. The numbers it prints are meaningless -
it only proves that the code runs.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
TEMP_OUTPUTS = Path(tempfile.mkdtemp(prefix="neuralproject_smoke_"))
os.environ["NEURALPROJECT_OUTPUT_DIR"] = str(TEMP_OUTPUTS)  # must be set BEFORE importing src

from src import config  # noqa: E402
from src.preprocessing import (check_no_leakage, expand_to_binary, fit_to_budget,  # noqa: E402
                               load_split, tokenize_code)
from src.utils import ensure_output_dirs  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError("FAILED: " + message)
    print("  ok -", message)


def test_preprocessing():
    print("\n[1] Preprocessing")
    paired = load_split("train").head(10)
    binary = expand_to_binary(paired)
    check(len(binary) == 2 * len(paired), "each pair becomes exactly two binary examples")
    check(binary.groupby("pair_id")["label"].apply(sorted).apply(lambda l: l == [0, 1]).all(),
          "each pair has one correct (0) and one hallucinated (1) example")
    try:
        check_no_leakage(["completion", "hallucination_type"])
        check(False, "leakage guard should reject hallucination_type")
    except RuntimeError:
        check(True, "leakage guard rejects label-revealing columns")
    check(tokenize_code("x = CacheMessageInfo(mi)") == ["x", "=", "cache", "message", "info", "(", "mi", ")"],
          "code tokenizer splits identifiers into sub-words")
    p, c, s = fit_to_budget(list(range(100)), list(range(30)), list(range(100)), budget=50, completion_max=20)
    check(len(p) + len(c) + len(s) <= 50 and len(c) == 20, "token budget keeps the completion and respects max length")
    check(p == list(range(100))[-len(p):], "token budget keeps the END of the prompt")


def tiny_data(n_train=40, n_eval=12):
    """Sample a tiny split and save it in the TEMPORARY folder (the real split is untouched)."""
    sizes = {"train": n_train, "validation": n_eval, "test": n_eval}
    tiny = {name: load_split(name).sample(n, random_state=0) for name, n in sizes.items()}
    config.PROCESSED_DIR = TEMP_OUTPUTS / "processed"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tiny.items():
        df.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
    return {name: expand_to_binary(df) for name, df in tiny.items()}


def test_rnn(data):
    print("\n[2] GRU / LSTM (2 epochs on a tiny subset)")
    from src.rnn_models import run_rnn_experiment
    config.RNN_CONFIG.update({"max_len": 120, "max_epochs": 2, "patience": 2, "batch_size": 16})
    for name in ["gru", "lstm"]:
        run_rnn_experiment(name, force=True, data=data)
        check((config.PREDICTIONS_DIR / f"{name}_test.csv").exists(), f"{name} test predictions saved")
        check((config.MODELS_DIR / name / "best_model.pt").exists(), f"{name} checkpoint saved")


def test_pretrained(data):
    print("\n[3] Frozen CodeBERT (Phase 4) on a tiny subset")
    from src.pretrained_models import run_frozen_experiment
    config.TRANSFORMER_INPUT_CONFIG.update({"max_len": 128, "completion_max_tokens": 48})
    config.FROZEN_CONFIG.update({"extract_batch_size": 8, "max_epochs": 3, "patience": 3})
    run_frozen_experiment("codebert", force=True, data=data)
    check((config.MODELS_DIR / "codebert_frozen" / "head.pt").exists(), "frozen-encoder head checkpoint saved")
    check((config.FEATURES_DIR / "codebert_train.pt").exists(), "frozen features cached")
    from src.pretrained_models import get_frozen_features
    from src.utils import get_logger as _get_logger
    cached = get_frozen_features("codebert", data, _get_logger("smoke"))
    check(cached["train"]["features"].shape[0] == len(data["train"]),
          "cached features reload after a restart (no recomputation)")

    print("\n[4] Fine-tuning the last 2 CodeBERT layers (Phase 5)")
    import torch
    from src.finetuning import build_model, run_finetuning, verify_trainable_layers
    from src.utils import get_logger
    logger = get_logger("smoke")
    _, model = build_model("codebert", 2, logger)
    summary = verify_trainable_layers(model, 2, logger)
    check(summary["trainable_blocks"] == ["backbone.encoder.layer.10", "backbone.encoder.layer.11", "head"],
          "only encoder layers 10, 11 and the head are trainable")
    model.backbone.encoder.layer[3].output.dense.weight.requires_grad = True  # simulate a mistake
    try:
        verify_trainable_layers(model, 2, logger)
        check(False, "verification should fail when an earlier layer is unfrozen")
    except RuntimeError:
        check(True, "verification stops training if a wrong layer is unfrozen")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    config.FINETUNE_CONFIG.update({"batch_size": 8, "grad_accumulation_steps": 2, "max_epochs": 1, "patience": 1})
    run_finetuning("codebert", force=True, data=data)
    check((config.MODELS_DIR / "codebert_finetuned_last2" / "best_model.pt").exists(), "fine-tuned checkpoint saved")


def test_evaluation():
    print("\n[5] Comparison (Phase 6)")
    from src.evaluation import run_comparison
    table = run_comparison()
    check((config.METRICS_DIR / "model_comparison.csv").exists(), "model_comparison.csv written")
    check((table.loc[table["model"].isin(["gru", "lstm"]), "status"] == "completed").all(),
          "trained models are evaluated")
    check(table.loc[table["model"] == "graphcodebert_frozen", "status"].str.startswith("unavailable").all(),
          "models that were not trained are marked unavailable (no invented numbers)")
    check((config.FIGURES_DIR / "comparison_f1.png").exists(), "comparison charts drawn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrained", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    print(f"Smoke-test outputs go to: {TEMP_OUTPUTS}")
    test_preprocessing()
    data = tiny_data()
    test_rnn(data)
    if not args.skip_pretrained:
        test_pretrained(data)
    test_evaluation()
    print(f"\nSMOKE TEST PASSED. (Temporary outputs: {TEMP_OUTPUTS})")
