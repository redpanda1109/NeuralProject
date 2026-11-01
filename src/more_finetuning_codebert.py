"""
EXTRA EXPERIMENT - Fine-tune the LAST FOUR encoder layers of CodeBERT (+ head).

Run from the project root:
    python -m src.more_finetuning_codebert           # skips if results already exist
    python -m src.more_finetuning_codebert --force   # retrain

How the CodeBERT experiments differ
-----------------------------------
    experiment                            frozen                    trainable
    ------------------------------------  ------------------------  --------------------------------
    codebert_frozen          (Phase 4)    embeddings + layers 0-11  head only
    codebert_finetuned_last2 (Phase 5)    embeddings + layers 0-9   layers 10-11 + head   (~14.6M)
    codebert_finetuned_last4 (THIS FILE)  embeddings + layers 0-7   layers 8-11 + head    (~28.7M)
    codebert_extra_transformer_layer      embeddings + layers 0-11  NEW layer + head
                                          (src/adding_layers_codebert.py)

The ONLY difference to the last-2 experiment is the number of unfrozen layers. Same
input, split, head (initialised from the Phase 4 head), optimiser, learning rates,
warmup, batch size, early stopping on the validation loss and evaluation - all taken
from config.FINETUNE_CONFIG and the shared code in src/finetuning.py.

CodeBERT (microsoft/codebert-base, RobertaModel) structure:
    embeddings (word + position + token-type + LayerNorm)   -> FROZEN
    encoder.layer.0 ... encoder.layer.7                      -> FROZEN
    encoder.layer.8, .9, .10, .11                            -> TRAINABLE (the last four)
    (no pooler - the project pools the completion tokens itself)
    head (CompletionHead MLP)                                -> TRAINABLE

Outputs use the name `codebert_finetuned_last4`, so nothing of the last-2 experiment
(`codebert_finetuned_last2`) is overwritten.
"""

import argparse

from src import config
from src.evaluation import prediction_path
from src.finetuning import (build_model, expected_trainable_prefixes, run_finetuning,
                            verify_trainable_layers)
from src.pretrained_models import encoder_layout
from src.utils import ensure_output_dirs, get_logger

MODEL_KEY = "codebert"
N_LAYERS = config.UNFREEZE_LAST_N_LAYERS_EXTENDED  # 4
EXPERIMENT_NAME = f"{MODEL_KEY}_finetuned_last{N_LAYERS}"  # codebert_finetuned_last4


def check_last_four_layers(logger):
    """
    Build the model on the CPU and verify - before any training - that EXACTLY
    encoder layers 8, 9, 10, 11 and the head are trainable. Raises RuntimeError otherwise.
    Loads the pretrained weights (cached after the first download) but trains nothing.
    """
    _, model = build_model(MODEL_KEY, N_LAYERS, logger)
    layers = encoder_layout(model.backbone)["layers"]
    expected_indices = list(range(len(layers) - N_LAYERS, len(layers)))
    if len(layers) != 12 or expected_indices != [8, 9, 10, 11]:
        raise RuntimeError(f"Unexpected CodeBERT structure: {len(layers)} layers, "
                           f"last four = {expected_indices}")
    logger.info(f"  Last four CodeBERT encoder layers: {expected_indices} "
                f"(prefixes {list(expected_trainable_prefixes(model, N_LAYERS))})")

    summary = verify_trainable_layers(model, N_LAYERS, logger)  # raises on any mismatch

    # Independent count: trainable parameters must equal layers 8-11 + head, nothing else.
    expected = sum(p.numel() for i in expected_indices for p in layers[i].parameters())
    expected += sum(p.numel() for p in model.head.parameters())
    if summary["trainable"] != expected:
        raise RuntimeError(f"Trainable parameter count {summary['trainable']:,} != expected {expected:,}")
    if any(p.requires_grad for p in model.backbone.embeddings.parameters()):
        raise RuntimeError("CodeBERT embeddings must stay frozen.")
    if summary["frozen"] == 0:
        raise RuntimeError("Nothing is frozen - the whole model would be fine-tuned.")
    total = summary["trainable"] + summary["frozen"]
    logger.info(f"  Trainable share: {summary['trainable'] / total:.1%} of {total:,} parameters "
                f"(layers 8-11 + head = {expected:,}). Check passed.")
    del model
    return summary


def run_last_four(force=False, data=None):
    """Same pipeline as Phase 5, with n = 4. Results are saved as `codebert_finetuned_last4`."""
    # run_finetuning builds the model itself and calls verify_trainable_layers again
    # right before training, so the check above is repeated on the real training model.
    run_finetuning(MODEL_KEY, force=force, data=data, n=N_LAYERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeBERT: fine-tune the last FOUR encoder layers + head.")
    parser.add_argument("--force", action="store_true", help="retrain even if saved results exist")
    parser.add_argument("--check-only", action="store_true",
                        help="only build the model and verify the frozen/trainable layers (no training)")
    args = parser.parse_args()
    ensure_output_dirs()
    log = get_logger(EXPERIMENT_NAME)
    if not (args.force or args.check_only) and prediction_path(EXPERIMENT_NAME, "test").exists():
        log.info(f"{EXPERIMENT_NAME} already trained - skipping (use --force to retrain).")
    else:
        log.info("=" * 70)
        log.info("CodeBERT last-four-layer experiment: layer check before training")
        log.info("=" * 70)
        check_last_four_layers(log)
        if not args.check_only:
            run_last_four(force=args.force)
