"""
EXTRA EXPERIMENT - FROZEN CodeBERT + one NEW trainable Transformer layer + head.

Run from the project root:
    python -m src.adding_layers_codebert           # skips if results already exist
    python -m src.adding_layers_codebert --force   # retrain

Idea
----
Phase 4 (codebert_frozen) puts the classification head directly on top of the frozen
CodeBERT output. Phase 5 changes CodeBERT's own top layers. This experiment does
neither: CodeBERT stays COMPLETELY frozen (embeddings + all 12 layers, exactly the
public weights) and we insert ONE NEW, randomly initialised Transformer layer between
the encoder output and the head:

    input ids -> [CodeBERT, 12 layers, FROZEN] -> hidden states (batch, seq, 768)
              -> [NEW Transformer layer, TRAINABLE] (self-attention over ALL tokens,
                 padding masked with the same attention mask CodeBERT uses)
              -> mean+max pooling over the completion tokens (same as every other model)
              -> [CompletionHead MLP, TRAINABLE] -> 1 logit

So the question is: does an extra, task-specific attention layer on top of frozen
features help, compared with (a) the head alone and (b) adapting CodeBERT's own layers?

    experiment                            CodeBERT weights          trained
    ------------------------------------  ------------------------  ---------------------------
    codebert_frozen          (Phase 4)    all frozen                head
    codebert_finetuned_last2 (Phase 5)    layers 10-11 trainable    layers 10-11 + head
    codebert_finetuned_last4              layers 8-11 trainable     layers 8-11 + head
    codebert_extra_transformer_layer      all frozen                NEW layer + head (THIS FILE)

The new layer
-------------
`torch.nn.TransformerEncoderLayer`, configured from CodeBERT's own config so that it
has exactly the shape of a CodeBERT layer: hidden size 768, 12 attention heads,
feed-forward size 3072, GELU, post-LayerNorm (norm_first=False, as in RoBERTa),
LayerNorm eps 1e-5, dropout 0.1, batch_first=True. It is initialised like CodeBERT's
layers (normal(0, initializer_range=0.02) weights, zero biases, LayerNorm = 1/0).
Attention mask convention: CodeBERT's attention_mask is 1 for real tokens and 0 for
padding; PyTorch's src_key_padding_mask is True for padding, so we pass
`attention_mask == 0`.

The frozen encoder is kept in eval mode during training (no dropout inside CodeBERT),
so it produces exactly the same features as in Phase 4; it runs under torch.no_grad()
because no gradient has to flow into it. Everything else (input format, split,
loss, AdamW + linear warmup, batch size 16 x 2, early stopping on the VALIDATION loss,
evaluation, output files) is shared with Phase 5 via finetuning.train_and_evaluate.
The head is initialised randomly (the Phase 4 head was trained on different inputs).

Outputs use the name `codebert_extra_transformer_layer`
(outputs/models/codebert_extra_transformer_layer/best_model.pt holds only the new
layer + head; CodeBERT itself is the unchanged public checkpoint).
"""

import argparse

import torch
from torch import nn

from src import config
from src.evaluation import prediction_path
from src.finetuning import train_and_evaluate
from src.preprocessing import load_binary_splits
from src.pretrained_models import CompletionHead, freeze, load_tokenizer_and_encoder, uses_amp
from src.training import count_parameters, pool_completion
from src.utils import ensure_output_dirs, get_device, get_logger, set_seed

MODEL_KEY = "codebert"
EXPERIMENT_NAME = "codebert_extra_transformer_layer"
TRAINABLE_PREFIXES = ("extra_layers.", "head.")


def build_new_transformer_layer(encoder_config, dropout):
    """A fresh Transformer layer with the same shape as one CodeBERT layer."""
    layer = nn.TransformerEncoderLayer(
        d_model=encoder_config.hidden_size,               # 768
        nhead=encoder_config.num_attention_heads,         # 12
        dim_feedforward=encoder_config.intermediate_size,  # 3072
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=encoder_config.layer_norm_eps,     # 1e-5
        batch_first=True,
        norm_first=False,                                 # post-LayerNorm, like RoBERTa/CodeBERT
    )
    std = encoder_config.initializer_range                # 0.02, CodeBERT's initialisation
    for name, parameter in layer.named_parameters():
        if "norm" in name:
            nn.init.ones_(parameter) if name.endswith("weight") else nn.init.zeros_(parameter)
        elif name.endswith("bias"):
            nn.init.zeros_(parameter)
        else:
            nn.init.normal_(parameter, mean=0.0, std=std)
    return layer


class FrozenEncoderWithExtraLayer(nn.Module):
    """Frozen pretrained encoder -> NEW Transformer layer(s) -> completion pooling -> head."""

    def __init__(self, backbone, extra_layers, head):
        super().__init__()
        self.backbone = backbone          # frozen, never updated
        self.extra_layers = extra_layers  # nn.ModuleList, trainable
        self.head = head                  # trainable

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen encoder: always eval mode (no dropout) = Phase 4 features
        return self

    def forward(self, input_ids, attention_mask, completion_mask):
        with torch.no_grad():  # no gradients into the frozen encoder (saves memory and time)
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        padding_mask = attention_mask == 0  # True = padding (PyTorch convention)
        for layer in self.extra_layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
        return self.head(pool_completion(hidden, completion_mask))


def build_model(logger):
    cfg = config.EXTRA_LAYER_CONFIG
    tokenizer, backbone = load_tokenizer_and_encoder(MODEL_KEY)
    freeze(backbone)  # embeddings + all 12 encoder layers
    extra_layers = nn.ModuleList([build_new_transformer_layer(backbone.config, cfg["dropout"])
                                  for _ in range(cfg["num_new_layers"])])
    head_cfg = config.FROZEN_CONFIG  # same head architecture as every Transformer experiment
    head = CompletionHead(backbone.config.hidden_size, head_cfg["head_hidden_dim"], head_cfg["head_dropout"])
    logger.info("  New Transformer layer and head initialised randomly.")
    return tokenizer, FrozenEncoderWithExtraLayer(backbone, extra_layers, head)


def verify_only_new_layers_trainable(model, logger):
    """
    Safety check before training: the pretrained encoder must be 100% frozen and
    every parameter of the new layer(s) + head must be trainable. Raises otherwise.
    """
    wrong = [f"{name} (requires_grad={p.requires_grad})" for name, p in model.named_parameters()
             if p.requires_grad != name.startswith(TRAINABLE_PREFIXES)]
    encoder_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    new_layer_params = sum(p.numel() for p in model.extra_layers.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    trainable, frozen = count_parameters(model)

    logger.info(f"  Pretrained CodeBERT : {sum(p.numel() for p in model.backbone.parameters()):,} params, "
                f"trainable {encoder_trainable:,} (must be 0)")
    logger.info(f"  New Transformer layer(s): {len(model.extra_layers)} x "
                f"{new_layer_params // len(model.extra_layers):,} params  TRAINABLE")
    logger.info(f"  Classification head : {head_params:,} params  TRAINABLE")
    logger.info(f"  Trainable parameters: {trainable:,} | frozen parameters: {frozen:,}")
    if wrong or encoder_trainable != 0 or trainable != new_layer_params + head_params:
        raise RuntimeError("Wrong parameters are trainable/frozen - refusing to train:\n  "
                           + "\n  ".join(wrong[:20] or [f"trainable={trainable:,}, expected "
                                                        f"{new_layer_params + head_params:,}"]))
    logger.info("  Verification passed: CodeBERT is completely frozen; only the new layer + head train.")
    return {"trainable": trainable, "frozen": frozen, "encoder_trainable": encoder_trainable,
            "new_layer_parameters": new_layer_params, "head_parameters": head_params}


def run_extra_layer_experiment(force=False, data=None):
    name = EXPERIMENT_NAME
    logger = get_logger(name)
    if not force and prediction_path(name, "test").exists():
        logger.info(f"{name} already trained - skipping (use --force to retrain).")
        return
    logger.info("=" * 70)
    logger.info("CodeBERT FROZEN + one NEW Transformer layer before the head")
    logger.info("=" * 70)
    set_seed()
    cfg = config.EXTRA_LAYER_CONFIG
    device = get_device()
    use_amp = uses_amp(MODEL_KEY, device)

    data = data or load_binary_splits()
    tokenizer, model = build_model(logger)
    layer_check = verify_only_new_layers_trainable(model, logger)
    model.to(device)

    param_groups = [
        {"params": list(model.extra_layers.parameters()), "lr": cfg["learning_rate"], "name": "new Transformer layer"},
        {"params": list(model.head.parameters()), "lr": cfg["learning_rate"], "name": "head"},
    ]
    train_and_evaluate(
        name, MODEL_KEY, model, tokenizer, data, cfg, param_groups, device, use_amp, logger,
        reference_extra={"experiment": "frozen CodeBERT + new Transformer layer + head",
                         "encoder": "all layers frozen (unchanged pretrained weights, eval mode)",
                         "new_layer": "torch.nn.TransformerEncoderLayer(768, 12 heads, ff 3072, gelu, post-LN)",
                         "extra_layer_config": cfg, "layer_check": layer_check,
                         "checkpoint_note": "best_model.pt holds ONLY extra_layers.* and head.*; "
                                            "load the public CodeBERT first, then this file with strict=False."},
        plot_title="CodeBERT frozen + new Transformer layer")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen CodeBERT + one new trainable Transformer layer + head.")
    parser.add_argument("--force", action="store_true", help="retrain even if saved results exist")
    parser.add_argument("--check-only", action="store_true",
                        help="only build the model and verify the frozen/trainable parameters (no training)")
    args = parser.parse_args()
    ensure_output_dirs()
    if args.check_only:
        log = get_logger(EXPERIMENT_NAME)
        set_seed()
        verify_only_new_layers_trainable(build_model(log)[1], log)
    else:
        run_extra_layer_experiment(force=args.force)
