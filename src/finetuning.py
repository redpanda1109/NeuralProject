"""
PHASE 5 - Fine-tune ONLY the final two Transformer layers of CodeBERT.

Run from the project root:
    python -m src.finetuning                        # CodeBERT (primary experiment)
    python -m src.finetuning --model graphcodebert  # optional, same recipe
    python -m src.finetuning --force                # retrain even if results exist

What changes compared with Phase 4?
-----------------------------------
Phase 4: all 12 encoder layers frozen, only the head learns.
Phase 5: layers 0-9 frozen, layers 10 and 11 (the last two) + the head learn.

Why only the last two layers?
    The lower layers of a pretrained Transformer learn general things (syntax,
    which tokens belong together). The top layers combine this into task-specific
    meaning. Letting only the top two layers adapt lets the model specialise in
    "does this completion fit its context?" while keeping the general code
    knowledge in the lower layers intact. It also trains ~14M instead of ~125M
    parameters: faster, less GPU memory, and less risk of overfitting a small
    dataset (~2.7k training examples).

This module builds directly on Phase 4: same tokenizer, same input format, same
pooling, same head architecture, same split - and the head starts from the head
trained in Phase 4 when it exists.

Full fine-tuning (unfreezing every layer) is intentionally NOT implemented.
"""

import argparse
import math

import torch

from src import config
from src import visualization as viz
from src.evaluation import prediction_path, report_model, save_predictions
from src.preprocessing import load_binary_splits
from src.pretrained_models import (CompletionHead, TransformerClassifier, freeze,
                                   load_tokenizer_and_encoder, make_token_loaders,
                                   save_model_reference)
from src.training import count_parameters, predict, train_model
from src.utils import ensure_output_dirs, get_device, get_logger, load_json, save_json, set_seed

UNFREEZE_LAST_N_LAYERS = config.UNFREEZE_LAST_N_LAYERS  # primary experiment: exactly 2


def unfreeze_last_n_layers(model, n):
    """Freeze the whole encoder, then unfreeze its final n layers and the head."""
    freeze(model.backbone)
    layers = model.backbone.encoder.layer
    if not 1 <= n < len(layers):
        raise ValueError(f"n must be between 1 and {len(layers) - 1} (full fine-tuning is not allowed)")
    for layer in layers[len(layers) - n:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def expected_trainable_prefixes(model, n):
    total = len(model.backbone.encoder.layer)
    # The trailing "." matters: "layer.1." must not match "layer.10."
    return tuple(f"backbone.encoder.layer.{i}." for i in range(total - n, total)) + ("head.",)


def verify_trainable_layers(model, n, logger):
    """
    Safety check before training. Stops with an error if ANY parameter outside the
    final n encoder layers + head is trainable, or if any parameter inside them is frozen.
    """
    allowed = expected_trainable_prefixes(model, n)
    wrong = []
    trainable_blocks = set()
    for name, parameter in model.named_parameters():
        should_train = name.startswith(allowed)
        if parameter.requires_grad != should_train:
            wrong.append(f"{name} (requires_grad={parameter.requires_grad}, expected {should_train})")
        if parameter.requires_grad:
            trainable_blocks.add(".".join(name.split(".")[:4]) if name.startswith("backbone") else "head")

    trainable, frozen = count_parameters(model)
    total_layers = len(model.backbone.encoder.layer)
    logger.info(f"  Encoder layers: {total_layers}  |  unfrozen: the last {n} "
                f"(layers {total_layers - n}-{total_layers - 1})")
    logger.info(f"  Trainable parameters: {trainable:,}")
    logger.info(f"  Frozen parameters   : {frozen:,}")
    logger.info(f"  Trainable blocks    : {sorted(trainable_blocks)}")
    if wrong:
        raise RuntimeError("Wrong layers are trainable/frozen - refusing to train:\n  " + "\n  ".join(wrong[:20]))
    logger.info("  Verification passed: only the final layers + head are trainable; "
                "embeddings and earlier layers are frozen.")
    return {"trainable": trainable, "frozen": frozen, "trainable_blocks": sorted(trainable_blocks)}


def build_model(model_key, n, logger):
    tokenizer, backbone = load_tokenizer_and_encoder(model_key)
    head_cfg = config.FROZEN_CONFIG  # same head architecture as Phase 4
    head = CompletionHead(backbone.config.hidden_size, head_cfg["head_hidden_dim"], head_cfg["head_dropout"])

    frozen_head_path = config.MODELS_DIR / f"{model_key}_frozen" / "head.pt"
    if config.FINETUNE_CONFIG["init_head_from_frozen"] and frozen_head_path.exists():
        head.load_state_dict(torch.load(frozen_head_path, map_location="cpu"))
        logger.info(f"  Head initialised from the Phase 4 head: {frozen_head_path}")
    else:
        logger.info("  Head initialised randomly (no Phase 4 head found).")

    model = TransformerClassifier(backbone, head)
    unfreeze_last_n_layers(model, n)
    return tokenizer, model


def run_finetuning(model_key="codebert", force=False, data=None, n=UNFREEZE_LAST_N_LAYERS):
    name = f"{model_key}_finetuned_last{n}"
    logger = get_logger(name)
    if not force and prediction_path(name, "test").exists():
        logger.info(f"{name} already trained - skipping (use --force to retrain).")
        return
    logger.info("=" * 70)
    logger.info(f"PHASE 5 - {model_key}: fine-tune ONLY the final {n} encoder layers + head")
    logger.info("=" * 70)
    if n != 2:
        logger.info(f"  NOTE: n={n}. The primary experiment of this project uses n=2.")
    set_seed()
    cfg = config.FINETUNE_CONFIG
    device = get_device()
    use_amp = device.type == "cuda"  # fp16 mixed precision on the T4

    data = data or load_binary_splits()
    tokenizer, model = build_model(model_key, n, logger)
    layer_summary = verify_trainable_layers(model, n, logger)
    model.to(device)

    loaders = make_token_loaders(data, tokenizer, cfg["batch_size"], logger)

    # Two learning rates: small for pretrained layers, larger for the head.
    encoder_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["encoder_learning_rate"]},
        {"params": model.head.parameters(), "lr": cfg["head_learning_rate"]},
    ], weight_decay=cfg["weight_decay"])

    from transformers import get_linear_schedule_with_warmup
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg["grad_accumulation_steps"])
    total_steps = steps_per_epoch * cfg["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_fraction"] * total_steps), total_steps)
    logger.info(f"  batch size {cfg['batch_size']} x accumulation {cfg['grad_accumulation_steps']} "
                f"= effective {cfg['batch_size'] * cfg['grad_accumulation_steps']} | mixed precision: {use_amp}")

    model_dir = config.MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    history = train_model(model, loaders["train"], loaders["validation"], optimizer, device,
                          cfg["max_epochs"], cfg["patience"], model_dir / "best_model.pt", logger,
                          scheduler=scheduler, use_amp=use_amp,
                          grad_accumulation_steps=cfg["grad_accumulation_steps"],
                          max_grad_norm=cfg["max_grad_norm"])

    save_model_reference(model_dir, model_key, tokenizer,
                         {"phase": 5, "unfrozen_last_n_layers": n, "finetune": cfg,
                          "layer_check": layer_summary,
                          "checkpoint_note": "best_model.pt holds ONLY the trained parameters "
                                             "(last layers + head); load the public pretrained "
                                             "model first, then this file with strict=False."})
    save_json(history, config.METRICS_DIR / f"{name}_training_history.json")
    viz.plot_loss_curves(history, f"{model_key} - last {n} layers fine-tuned",
                         config.FIGURES_DIR / f"loss_curve_{name}.png")

    for split in ["validation", "test"]:  # these loaders are not shuffled -> order matches data[split]
        result = predict(model, loaders[split], device, use_amp)
        save_predictions(name, split, data[split], result["probs"], (result["probs"] >= 0.5).astype(int))
    report_model(name)
    print_frozen_vs_finetuned(model_key, name, logger)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def print_frozen_vs_finetuned(model_key, finetuned_name, logger):
    """Side-by-side test metrics read from the saved metrics files (no new numbers)."""
    frozen_path = config.METRICS_DIR / f"{model_key}_frozen_metrics.json"
    finetuned_path = config.METRICS_DIR / f"{finetuned_name}_metrics.json"
    if not frozen_path.exists():
        logger.info("  (Run Phase 4 for this model to compare frozen vs fine-tuned.)")
        return
    frozen, tuned = load_json(frozen_path)["test"], load_json(finetuned_path)["test"]
    logger.info(f"  {'metric':<10} {'frozen':>8} {'last-2':>8} {'change':>8}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        logger.info(f"  {metric:<10} {frozen[metric]:>8.4f} {tuned[metric]:>8.4f} "
                    f"{tuned[metric] - frozen[metric]:>+8.4f}")
    logger.info("  Run `python -m src.evaluation` for charts and confidence intervals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: fine-tune only the last N encoder layers.")
    parser.add_argument("--model", choices=["codebert", "graphcodebert"], default="codebert")
    parser.add_argument("--force", action="store_true", help="retrain even if saved results exist")
    args = parser.parse_args()
    ensure_output_dirs()
    run_finetuning(args.model, args.force)
