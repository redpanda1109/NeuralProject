"""
PHASE 5 - Fine-tune ONLY the final two Transformer layers of a pretrained encoder.

Run from the project root:
    python -m src.finetuning                        # CodeBERT (primary experiment)
    python -m src.finetuning --model graphcodebert  # optional, same recipe
    python -m src.finetuning --model codet5_small   # CodeT5-small (encoder blocks 4-5)
    python -m src.finetuning --model unixcoder      # UniXcoder (layers 10-11)
    python -m src.finetuning --force                # retrain even if results exist

What changes compared with Phase 4?
-----------------------------------
Phase 4 (frozen): every encoder layer frozen, only the head learns.
Phase 5 (partially fine-tuned): the bottom layers + embeddings stay frozen, the LAST
two layers + the head learn. Per model this means:

    model          architecture          layers   frozen               trainable
    -------------  --------------------  -------  -------------------  -------------------------------
    CodeBERT       RoBERTa encoder.layer  12      embeddings, 0-9      layers 10, 11 + head
    GraphCodeBERT  RoBERTa encoder.layer  12      embeddings, 0-9      layers 10, 11 + head
    UniXcoder      RoBERTa encoder.layer  12      embeddings, 0-9      layers 10, 11 + head
    CodeT5-small   T5 encoder.block        6      shared emb., 0-3     blocks 4, 5 + final_layer_norm + head
                   (decoder not loaded)

CodeT5 note: T5 layers are "pre-LayerNorm" - each block normalises its INPUT, and the
output of the last block is normalised by `encoder.final_layer_norm`. In RoBERTa that
final normalisation is inside layer 11 and is therefore already unfrozen with it. To
give both architectures the same meaning of "the last two layers", CodeT5's
final_layer_norm (512 weights) is unfrozen together with blocks 4-5. T5's relative
position bias lives in block 0 and stays frozen.

Other experiments built on this module:
    src/more_finetuning_codebert.py  -> CodeBERT with the LAST FOUR layers unfrozen
    src/adding_layers_codebert.py    -> frozen CodeBERT + a NEW Transformer layer

Why only the last layers?
    The lower layers of a pretrained Transformer learn general things (syntax,
    which tokens belong together). The top layers combine this into task-specific
    meaning. Letting only the top layers adapt lets the model specialise in
    "does this completion fit its context?" while keeping the general code
    knowledge in the lower layers intact. It also trains ~14M instead of ~125M
    parameters for CodeBERT: faster, less GPU memory, and less risk of overfitting
    a small dataset (~2.7k training examples).

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
from src.pretrained_models import (CompletionHead, TransformerClassifier, encoder_layout, freeze,
                                   load_tokenizer_and_encoder, make_token_loaders,
                                   save_model_reference, uses_amp)
from src.training import count_parameters, predict, train_model
from src.utils import ensure_output_dirs, get_device, get_logger, load_json, save_json, set_seed

UNFREEZE_LAST_N_LAYERS = config.UNFREEZE_LAST_N_LAYERS  # primary experiment: exactly 2
FINETUNABLE_MODELS = ["codebert", "graphcodebert", "codet5_small", "unixcoder"]


def unfreeze_last_n_layers(model, n):
    """Freeze the whole encoder, then unfreeze its final n layers (+ top norm, if any) and the head."""
    freeze(model.backbone)
    layout = encoder_layout(model.backbone)
    layers = layout["layers"]
    if not 1 <= n < len(layers):
        raise ValueError(f"n must be between 1 and {len(layers) - 1} (full fine-tuning is not allowed)")
    for layer in layers[len(layers) - n:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for module in layout["top_modules"].values():  # only T5's final_layer_norm
        for parameter in module.parameters():
            parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def expected_trainable_prefixes(model, n):
    layout = encoder_layout(model.backbone)
    total = len(layout["layers"])
    # The trailing "." matters: "layer.1." must not match "layer.10."
    layer_prefixes = tuple(f"backbone.{layout['layer_path']}.{i}." for i in range(total - n, total))
    top_prefixes = tuple(f"backbone.{path}." for path in layout["top_modules"])
    return layer_prefixes + top_prefixes + ("head.",)


def _block_name(parameter_name, layout):
    """'backbone.encoder.layer.10.attention.self.query.weight' -> 'backbone.encoder.layer.10'."""
    if parameter_name.startswith("head."):
        return "head"
    for path in layout["top_modules"]:
        if parameter_name.startswith(f"backbone.{path}."):
            return f"backbone.{path}"
    depth = 1 + len(layout["layer_path"].split(".")) + 1  # backbone + path + layer index
    return ".".join(parameter_name.split(".")[:depth])


def verify_trainable_layers(model, n, logger):
    """
    Safety check before training. Stops with an error if ANY parameter outside the
    final n encoder layers (+ top norm) + head is trainable, or if any parameter
    inside them is frozen.
    """
    layout = encoder_layout(model.backbone)
    allowed = expected_trainable_prefixes(model, n)
    wrong = []
    trainable_blocks = set()
    for name, parameter in model.named_parameters():
        should_train = name.startswith(allowed)
        if parameter.requires_grad != should_train:
            wrong.append(f"{name} (requires_grad={parameter.requires_grad}, expected {should_train})")
        if parameter.requires_grad:
            trainable_blocks.add(_block_name(name, layout))

    trainable, frozen = count_parameters(model)
    total_layers = len(layout["layers"])
    logger.info(f"  Encoder: {type(model.backbone).__name__} | layers at backbone.{layout['layer_path']} "
                f"| {total_layers} layers | unfrozen: the last {n} "
                f"(layers {total_layers - n}-{total_layers - 1})")
    # Layer-by-layer status so it is obvious what is being fine-tuned.
    for index, layer in enumerate(layout["layers"]):
        state = "TRAINABLE" if all(p.requires_grad for p in layer.parameters()) else (
            "frozen" if not any(p.requires_grad for p in layer.parameters()) else "MIXED (error)")
        logger.info(f"    {layout['layer_path']}.{index:<2} {sum(p.numel() for p in layer.parameters()):>11,} params  {state}")
    for path, module in layout["top_modules"].items():
        state = "TRAINABLE" if all(p.requires_grad for p in module.parameters()) else "frozen"
        logger.info(f"    {path:<15} {sum(p.numel() for p in module.parameters()):>8,} params  {state} "
                    f"(normalises the last layer's output)")
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
    head = CompletionHead(backbone.config.hidden_size if hasattr(backbone.config, "hidden_size")
                          else backbone.config.d_model, head_cfg["head_hidden_dim"], head_cfg["head_dropout"])

    frozen_head_path = config.MODELS_DIR / f"{model_key}_frozen" / "head.pt"
    if config.get_finetune_config(model_key)["init_head_from_frozen"] and frozen_head_path.exists():
        head.load_state_dict(torch.load(frozen_head_path, map_location="cpu"))
        logger.info(f"  Head initialised from the Phase 4 head: {frozen_head_path}")
    else:
        logger.info("  Head initialised randomly (no Phase 4 head found).")

    model = TransformerClassifier(backbone, head)
    unfreeze_last_n_layers(model, n)
    return tokenizer, model


def train_and_evaluate(name, model_key, model, tokenizer, data, cfg, param_groups, device, use_amp,
                       logger, reference_extra, plot_title):
    """
    Shared by every partially-trained Transformer experiment (Phase 5, last-4-layers,
    added Transformer layer): tokenise, AdamW + linear warmup, train with early stopping
    on the VALIDATION loss, save the best checkpoint (trainable parameters only),
    predict validation + test, write metrics/reports.
    """
    loaders = make_token_loaders(data, tokenizer, cfg["batch_size"], logger, model_key=model_key)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])

    from transformers import get_linear_schedule_with_warmup
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg["grad_accumulation_steps"])
    total_steps = steps_per_epoch * cfg["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_fraction"] * total_steps), total_steps)
    logger.info(f"  batch size {cfg['batch_size']} x accumulation {cfg['grad_accumulation_steps']} "
                f"= effective {cfg['batch_size'] * cfg['grad_accumulation_steps']} | mixed precision: {use_amp}")
    logger.info("  learning rates: " + ", ".join(f"{g.get('name', 'group')}={g['lr']}" for g in param_groups))

    model_dir = config.MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    history = train_model(model, loaders["train"], loaders["validation"], optimizer, device,
                          cfg["max_epochs"], cfg["patience"], model_dir / "best_model.pt", logger,
                          scheduler=scheduler, use_amp=use_amp,
                          grad_accumulation_steps=cfg["grad_accumulation_steps"],
                          max_grad_norm=cfg["max_grad_norm"])

    save_model_reference(model_dir, model_key, tokenizer, reference_extra)
    save_json(history, config.METRICS_DIR / f"{name}_training_history.json")
    viz.plot_loss_curves(history, plot_title, config.FIGURES_DIR / f"loss_curve_{name}.png")

    for split in ["validation", "test"]:  # these loaders are not shuffled -> order matches data[split]
        result = predict(model, loaders[split], device, use_amp)
        save_predictions(name, split, data[split], result["probs"], (result["probs"] >= 0.5).astype(int))
    report_model(name)
    return history


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
    cfg = config.get_finetune_config(model_key)  # == FINETUNE_CONFIG for CodeBERT/GraphCodeBERT
    device = get_device()
    use_amp = uses_amp(model_key, device)  # fp16 on the T4 (fp32 for CodeT5, see config.ENCODER_SPECS)

    data = data or load_binary_splits()
    tokenizer, model = build_model(model_key, n, logger)
    layer_summary = verify_trainable_layers(model, n, logger)
    model.to(device)

    # Two learning rates: small for pretrained layers, larger for the head.
    encoder_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [
        {"params": encoder_params, "lr": cfg["encoder_learning_rate"], "name": "unfrozen encoder layers"},
        {"params": list(model.head.parameters()), "lr": cfg["head_learning_rate"], "name": "head"},
    ]
    train_and_evaluate(
        name, model_key, model, tokenizer, data, cfg, param_groups, device, use_amp, logger,
        reference_extra={"phase": 5, "unfrozen_last_n_layers": n, "finetune": cfg,
                         "layer_check": layer_summary,
                         "checkpoint_note": "best_model.pt holds ONLY the trained parameters "
                                            "(last layers + head); load the public pretrained "
                                            "model first, then this file with strict=False."},
        plot_title=f"{model_key} - last {n} layers fine-tuned")
    print_frozen_vs_finetuned(model_key, name, logger, n)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def print_frozen_vs_finetuned(model_key, finetuned_name, logger, n=UNFREEZE_LAST_N_LAYERS):
    """Side-by-side test metrics read from the saved metrics files (no new numbers)."""
    frozen_path = config.METRICS_DIR / f"{model_key}_frozen_metrics.json"
    finetuned_path = config.METRICS_DIR / f"{finetuned_name}_metrics.json"
    if not frozen_path.exists():
        logger.info("  (Run Phase 4 for this model to compare frozen vs fine-tuned.)")
        return
    frozen, tuned = load_json(frozen_path)["test"], load_json(finetuned_path)["test"]
    label = f"last-{n}"
    logger.info(f"  {'metric':<10} {'frozen':>8} {label:>8} {'change':>8}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        logger.info(f"  {metric:<10} {frozen[metric]:>8.4f} {tuned[metric]:>8.4f} "
                    f"{tuned[metric] - frozen[metric]:>+8.4f}")
    logger.info("  Run `python -m src.evaluation` for charts and confidence intervals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: fine-tune only the last 2 encoder layers.")
    parser.add_argument("--model", choices=FINETUNABLE_MODELS, default="codebert")
    parser.add_argument("--force", action="store_true", help="retrain even if saved results exist")
    args = parser.parse_args()
    ensure_output_dirs()
    run_finetuning(args.model, args.force)
