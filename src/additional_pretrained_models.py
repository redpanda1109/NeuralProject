"""
PHASE 4 (additional models) - CodeT5-small and UniXcoder as FROZEN feature extractors.

Run from the project root:
    python -m src.additional_pretrained_models                       # both models
    python -m src.additional_pretrained_models --model codet5_small
    python -m src.additional_pretrained_models --model unixcoder
    python -m src.additional_pretrained_models --force               # retrain even if results exist

Fine-tuning their last two layers (Phase 5) is done by the existing module:
    python -m src.finetuning --model codet5_small
    python -m src.finetuning --model unixcoder

This file follows exactly the same recipe as src/pretrained_models.py (and reuses its
code): the same saved train/validation/test split, the same binary labels
(0 = correct, 1 = hallucinated), the same input window and token budget, the same
mean+max pooling over the completion tokens, the same 2-layer head, the same early
stopping on the validation loss, and the same output files - only the encoder changes.

The two models
--------------
CodeT5-small  (Salesforce/codet5-small, ~60M parameters in total)
    An encoder-DECODER model (T5) pretrained on code with identifier-aware objectives.
    For classification only the ENCODER is needed, so it is loaded with
    `T5EncoderModel` (the decoder is never downloaded into memory):
        shared token embeddings -> 6 encoder blocks (encoder.block.0-5)
        -> encoder.final_layer_norm, hidden size 512.
    It uses relative position biases (no 512-position table) and the same BPE
    tokenizer style as CodeBERT (<s> ... </s>), so the input format is unchanged.
    It runs in fp32 because T5 models can overflow in fp16 (config.ENCODER_SPECS).

UniXcoder     (microsoft/unixcoder-base, ~125M parameters)
    A RoBERTa-shaped encoder (12 layers, hidden size 768) pretrained on code, comments
    and flattened syntax trees. It was pretrained with a "mode" token; we use
    <encoder-only> (bidirectional attention, like BERT), as in Microsoft's reference
    code, so the input becomes:
        <s> <encoder-only> </s> language </s> prompt </s> completion </s> suffix </s>

Outputs (model names used everywhere: checkpoints, predictions, histories, reports)
    outputs/models/codet5_small_frozen/head.pt      outputs/models/unixcoder_frozen/head.pt
    outputs/features/codet5_small_{split}.pt        outputs/features/unixcoder_{split}.pt
    outputs/metrics/predictions/codet5_small_frozen_{validation,test}.csv  (same for unixcoder)
    outputs/metrics/codet5_small_frozen_metrics.json, ..._training_history.json, ...
"""

import argparse

from src import config
from src.preprocessing import load_binary_splits
from src.pretrained_models import encoder_layout, run_frozen_experiment
from src.utils import ensure_output_dirs

ADDITIONAL_MODELS = ["codet5_small", "unixcoder"]


def describe_encoder(model_key, backbone):
    """
    Human-readable summary of an encoder's layer structure (printed in Colab so you
    can see exactly which modules exist before anything is frozen or unfrozen).
    """
    layout = encoder_layout(backbone)
    cfg = backbone.config
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    lines = [f"{model_key} ({config.PRETRAINED_MODELS[model_key]}) -> {type(backbone).__name__}",
             f"  hidden size {hidden} | {len(layout['layers'])} Transformer layers at "
             f"backbone.{layout['layer_path']}.0 ... .{len(layout['layers']) - 1}"]
    if layout["top_modules"]:
        lines.append(f"  modules on top of the last layer: {list(layout['top_modules'])}")
    prefix = config.ENCODER_SPECS[model_key]["prefix_tokens"]
    if prefix:
        lines.append(f"  input prefix tokens after <s>: {prefix}")
    lines.append(f"  fp16 mixed precision on GPU: {config.ENCODER_SPECS[model_key]['use_amp']}")
    return "\n".join(lines)


def run_additional_frozen(model_keys=ADDITIONAL_MODELS, force=False, save_encoder=False):
    """Phase 4 for the additional models - reuses run_frozen_experiment unchanged."""
    ensure_output_dirs()
    shared_data = load_binary_splits()  # the SAVED split in data/processed/ (never re-created here)
    for key in model_keys:
        if key not in ADDITIONAL_MODELS:
            raise ValueError(f"Unknown additional model '{key}'. Choose from {ADDITIONAL_MODELS}")
        run_frozen_experiment(key, force, data=shared_data, save_encoder=save_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 for CodeT5-small and UniXcoder: "
                                                 "frozen encoder + trained head.")
    parser.add_argument("--model", choices=ADDITIONAL_MODELS + ["both"], default="both")
    parser.add_argument("--force", action="store_true", help="recompute features and retrain the head")
    parser.add_argument("--save-encoder", action="store_true",
                        help="also save a copy of the (unchanged) encoder weights")
    parser.add_argument("--describe", action="store_true",
                        help="only load the encoders and print their layer structure (no training)")
    args = parser.parse_args()
    keys = ADDITIONAL_MODELS if args.model == "both" else [args.model]
    if args.describe:
        from src.pretrained_models import load_tokenizer_and_encoder
        for key in keys:
            _, encoder = load_tokenizer_and_encoder(key)
            print(describe_encoder(key, encoder))
            del encoder
    else:
        run_additional_frozen(keys, args.force, args.save_encoder)
