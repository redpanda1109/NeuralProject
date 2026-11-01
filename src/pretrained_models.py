"""
PHASE 4 - Pretrained code models as FROZEN feature extractors.

Run from the project root:
    python -m src.pretrained_models                        # CodeBERT
    python -m src.pretrained_models --model graphcodebert  # GraphCodeBERT
    python -m src.pretrained_models --model both
    python -m src.pretrained_models --force                # retrain even if results exist

Idea (transfer learning without updating the pretrained weights)
---------------------------------------------------------------
1. Load CodeBERT (a 12-layer Transformer pretrained on millions of code files).
2. FREEZE all of its weights (requires_grad = False).
3. Run every example through it ONCE and keep the pooled hidden states of the
   completion tokens as a fixed feature vector (cached in outputs/features/).
4. Train only a small classification head (2-layer MLP) on those features.

Because the encoder never changes, step 3 is done only once. This makes Phase 4 fast
and lets a Colab session be restarted without recomputing the features.

The model input (identical for Phase 4 and Phase 5):
    <s> language </s> [end of prompt] </s> [completion] </s> [start of suffix] </s>
At most 512 tokens; the completion is always kept (see preprocessing.fit_to_budget).

The same code also serves CodeT5-small and UniXcoder (src/additional_pretrained_models.py).
Model-specific loading details live in config.ENCODER_SPECS and in `encoder_layout()`.

GraphCodeBERT note: GraphCodeBERT was pretrained with an extra data-flow graph input
that requires language-specific parsers. We use it as a plain sequence encoder (same
input as CodeBERT), which is fully supported by Hugging Face but does not use the
data-flow part. This is documented in the README.
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import config
from src import visualization as viz
from src.evaluation import prediction_path, report_model, save_predictions
from src.preprocessing import fit_to_budget, load_binary_splits, prompt_tail, suffix_head
from src.training import (count_parameters, move_to_device, pool_completion, predict,
                          save_trainable_state, train_model)
from src.utils import ensure_output_dirs, get_device, get_logger, save_json, set_seed


# ---------------------------------------------------------------------------
# Loading the pretrained model
# ---------------------------------------------------------------------------
def load_tokenizer_and_encoder(model_key):
    """
    Download (first time) and load the tokenizer + encoder from Hugging Face.

    RoBERTa-type models (CodeBERT, GraphCodeBERT, UniXcoder) -> AutoModel without pooler.
    CodeT5-small (encoder-decoder) -> T5EncoderModel: ONLY the encoder half is loaded,
    because a classifier needs contextual token representations, not generated text.
    """
    try:
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel
        from transformers.utils import logging as hf_logging
    except ImportError as error:
        raise ImportError("transformers is not installed: pip install -r requirements.txt") from error
    if model_key not in config.PRETRAINED_MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(config.PRETRAINED_MODELS)}")
    name = config.PRETRAINED_MODELS[model_key]
    previous = hf_logging.get_verbosity()
    # Hide the expected "pooler weights were not used" warning (we do not use the pooler).
    hf_logging.set_verbosity_error()
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        if config.ENCODER_SPECS[model_key]["architecture"] == "t5_encoder":
            encoder = T5EncoderModel.from_pretrained(name)
        else:
            # No pooling layer: we pool the completion tokens ourselves.
            encoder = AutoModel.from_pretrained(name, add_pooling_layer=False)
    except OSError as error:
        raise OSError(f"Could not load '{name}'. Check your internet connection "
                      f"(the model is downloaded from huggingface.co the first time).") from error
    finally:
        hf_logging.set_verbosity(previous)
    return tokenizer, encoder


def encoder_layout(backbone):
    """
    Where the stacked Transformer layers of an encoder live. Used by the fine-tuning
    code so that it never assumes CodeBERT's layer names for other models.

    Returns a dict:
      layers      : the nn.ModuleList of Transformer layers (bottom -> top)
      layer_path  : their attribute path inside the encoder ("encoder.layer" / "encoder.block")
      top_modules : modules that sit ON TOP of the last layer and only post-process its
                    output (T5's final LayerNorm). They are unfrozen together with the
                    last layers; RoBERTa models have none (their last LayerNorm is
                    inside layer 11).
    """
    encoder = getattr(backbone, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):      # RoBERTa family
        return {"layers": encoder.layer, "layer_path": "encoder.layer", "top_modules": {}}
    if encoder is not None and hasattr(encoder, "block"):      # T5 encoder (CodeT5)
        return {"layers": encoder.block, "layer_path": "encoder.block",
                "top_modules": {"encoder.final_layer_norm": encoder.final_layer_norm}}
    raise ValueError(f"Unsupported encoder architecture: {type(backbone).__name__}")


def prefix_token_ids(model_key, tokenizer):
    """Ids of the model-specific prefix tokens (e.g. UniXcoder's <encoder-only>)."""
    tokens = config.ENCODER_SPECS.get(model_key, {}).get("prefix_tokens", []) if model_key else []
    ids = tokenizer.convert_tokens_to_ids(list(tokens))
    for token, token_id in zip(tokens, ids):
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Prefix token {token!r} is not in the {model_key} vocabulary.")
    return list(ids)


def uses_amp(model_key, device):
    """fp16 mixed precision only on CUDA and only for models that are stable in fp16."""
    return device.type == "cuda" and config.ENCODER_SPECS.get(model_key, {}).get("use_amp", True)


# ---------------------------------------------------------------------------
# Encoding examples into token ids
# ---------------------------------------------------------------------------
def encode_examples(binary_df, tokenizer, max_len, completion_max_tokens, prefix_ids=()):
    """
    Returns a list of dicts: input_ids, completion_mask, label.
    The completion is marked with 1s in completion_mask so we can pool over it.

    prefix_ids: optional model-specific mode tokens. Without them (CodeBERT,
    GraphCodeBERT, CodeT5) the layout is unchanged:
        <s> lang </s> prompt </s> completion </s> suffix </s>
    With them (UniXcoder):
        <s> <encoder-only> </s> lang </s> prompt </s> completion </s> suffix </s>
    """
    from transformers.utils import logging as hf_logging
    previous = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # hide "sequence longer than 512" - we truncate ourselves

    def ids(texts):
        return tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    language_ids = ids(binary_df["language"])
    prompt_ids = ids(binary_df["prompt"].apply(prompt_tail))
    completion_ids = ids(binary_df["completion"])
    suffix_ids = ids(binary_df["suffix"].apply(suffix_head))
    hf_logging.set_verbosity(previous)

    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    start = [cls] + (list(prefix_ids) + [sep] if prefix_ids else [])  # [cls] when no prefix
    examples = []
    for lang, prompt, completion, suffix, label in zip(language_ids, prompt_ids, completion_ids,
                                                       suffix_ids, binary_df["label"]):
        budget = max_len - len(start) - 4 - len(lang)  # start tokens + 4 x </s>
        p, c, s = fit_to_budget(prompt, completion or [tokenizer.unk_token_id], suffix,
                                budget, completion_max_tokens)
        input_ids = start + lang + [sep] + p + [sep] + c + [sep] + s + [sep]
        before = len(start) + len(lang) + 1 + len(p) + 1
        mask = [0] * before + [1] * len(c) + [0] * (len(s) + 2)
        assert len(input_ids) == len(mask) <= max_len
        examples.append({"input_ids": input_ids, "completion_mask": mask, "label": int(label)})
    return examples


class TokenizedDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate_fn(pad_token_id):
    def collate(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        completion_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            input_ids[i, :n] = torch.tensor(ex["input_ids"])
            attention_mask[i, :n] = 1
            completion_mask[i, :n] = torch.tensor(ex["completion_mask"])
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.float)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "completion_mask": completion_mask, "labels": labels}
    return collate


def make_token_loaders(data, tokenizer, batch_size, logger, model_key=None):
    """Tokenize all splits and wrap them in DataLoaders (training data shuffled)."""
    cfg = config.TRANSFORMER_INPUT_CONFIG
    prefix_ids = prefix_token_ids(model_key, tokenizer)
    loaders = {}
    generator = torch.Generator().manual_seed(config.SEED)
    for split, df in data.items():
        examples = encode_examples(df, tokenizer, cfg["max_len"], cfg["completion_max_tokens"], prefix_ids)
        lengths = [len(ex["input_ids"]) for ex in examples]
        logger.info(f"  {split:<10}: {len(examples)} examples, mean length {sum(lengths) / len(lengths):.0f} tokens")
        loaders[split] = DataLoader(TokenizedDataset(examples), batch_size=batch_size,
                                    shuffle=(split == "train"), collate_fn=make_collate_fn(tokenizer.pad_token_id),
                                    generator=generator if split == "train" else None)
    return loaders


# ---------------------------------------------------------------------------
# Model pieces (also used by finetuning.py)
# ---------------------------------------------------------------------------
class CompletionHead(nn.Module):
    """Small MLP: pooled features (2 x hidden) -> 256 -> 1 logit."""

    def __init__(self, encoder_hidden_size, head_hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * encoder_hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class TransformerClassifier(nn.Module):
    """Pretrained encoder (`backbone`) + completion pooling + CompletionHead."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask, completion_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(pool_completion(hidden, completion_mask))


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def save_model_reference(model_dir, model_key, tokenizer, extra):
    """Save the tokenizer files and a config that says which pretrained weights were used."""
    tokenizer.save_pretrained(model_dir / "tokenizer")
    save_json({"pretrained_model": config.PRETRAINED_MODELS[model_key],
               "input": config.TRANSFORMER_INPUT_CONFIG,
               "pooling": "mean+max over completion tokens", **extra}, model_dir / "config.json")


# ---------------------------------------------------------------------------
# Phase 4 - frozen feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(backbone, loader, device, use_amp=None):
    """One forward pass through the FROZEN encoder -> pooled feature vector per example."""
    backbone.eval()
    use_amp = device.type == "cuda" if use_amp is None else use_amp
    features, labels = [], []
    for i, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        features.append(pool_completion(hidden.float(), batch["completion_mask"]).cpu())
        labels.append(batch["labels"].cpu())
        if i % 20 == 0 or i == len(loader):
            print(f"    extracted {i}/{len(loader)} batches", flush=True)
    return torch.cat(features), torch.cat(labels)


def get_frozen_features(model_key, data, logger, force=False):
    """Load cached features if they match the data; otherwise compute and cache them."""
    paths = {split: config.FEATURES_DIR / f"{model_key}_{split}.pt" for split in data}
    if not force and all(p.exists() for p in paths.values()):
        cached = {split: torch.load(p) for split, p in paths.items()}
        if all(cached[s]["pair_ids"] == list(data[s]["pair_id"]) and
               cached[s]["labels"].tolist() == data[s]["label"].tolist() for s in data):
            logger.info("  Using cached frozen-encoder features from outputs/features/")
            return cached
        logger.info("  Cached features do not match the current split - recomputing.")

    device = get_device()
    tokenizer, backbone = load_tokenizer_and_encoder(model_key)
    freeze(backbone)
    backbone.to(device)
    trainable, frozen = count_parameters(backbone)
    logger.info(f"  Encoder parameters: trainable={trainable:,} frozen={frozen:,} (must be 0 trainable)")
    if trainable != 0:
        raise RuntimeError("Phase 4 requires a completely frozen encoder.")

    loaders = make_token_loaders(data, tokenizer, config.FROZEN_CONFIG["extract_batch_size"], logger,
                                 model_key=model_key)
    use_amp = uses_amp(model_key, device)
    logger.info(f"  Mixed precision (fp16) for feature extraction: {use_amp}")
    cached = {}
    for split in data:
        # Use an unshuffled loader so features stay in the same order as the DataFrame.
        loader = DataLoader(loaders[split].dataset, batch_size=config.FROZEN_CONFIG["extract_batch_size"],
                            shuffle=False, collate_fn=loaders[split].collate_fn)
        logger.info(f"  Extracting {split} features ...")
        features, labels = extract_features(backbone, loader, device, use_amp)
        cached[split] = {"features": features, "labels": labels, "pair_ids": list(data[split]["pair_id"])}
        torch.save(cached[split], paths[split])
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"features": self.features[i], "labels": self.labels[i]}


def run_frozen_experiment(model_key, force=False, data=None, save_encoder=False):
    name = f"{model_key}_frozen"
    logger = get_logger(name)
    if not force and prediction_path(name, "test").exists():
        logger.info(f"{name} already trained - skipping (use --force to retrain).")
        return
    logger.info("=" * 70)
    logger.info(f"PHASE 4 - {model_key}: FROZEN encoder + trainable classification head")
    logger.info("=" * 70)
    set_seed()
    cfg = config.FROZEN_CONFIG
    data = data or load_binary_splits()
    cached = get_frozen_features(model_key, data, logger, force=force)

    device = get_device(verbose=False)
    generator = torch.Generator().manual_seed(config.SEED)
    loaders = {split: DataLoader(FeatureDataset(c["features"], c["labels"]), batch_size=cfg["batch_size"],
                                 shuffle=(split == "train"), generator=generator if split == "train" else None)
               for split, c in cached.items()}
    hidden_size = cached["train"]["features"].shape[1] // 2
    head = CompletionHead(hidden_size, cfg["head_hidden_dim"], cfg["head_dropout"]).to(device)
    trainable, _ = count_parameters(head)
    logger.info(f"  Classification head trainable parameters: {trainable:,}")

    model_dir = config.MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history = train_model(head, loaders["train"], loaders["validation"], optimizer, device,
                          cfg["max_epochs"], cfg["patience"], model_dir / "head.pt", logger)

    # The frozen encoder is byte-identical to the public checkpoint, so by default we
    # save only the tokenizer + a reference to it (use --save-encoder for a full copy).
    tokenizer, encoder = load_tokenizer_and_encoder(model_key)
    save_model_reference(model_dir, model_key, tokenizer,
                         {"phase": 4, "encoder": "all layers frozen (unchanged pretrained weights)",
                          "head_checkpoint": "head.pt", "head": cfg})
    if save_encoder:
        encoder.save_pretrained(model_dir / "frozen_encoder")
    del encoder
    save_json(history, config.METRICS_DIR / f"{name}_training_history.json")
    viz.plot_loss_curves(history, f"{model_key} frozen encoder - head training loss",
                         config.FIGURES_DIR / f"loss_curve_{name}.png")

    for split in ["validation", "test"]:
        result = predict(head, loaders[split], device)
        save_predictions(name, split, data[split], result["probs"], (result["probs"] >= 0.5).astype(int))
    report_model(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: frozen pretrained encoder + trained head.")
    # CodeT5-small and UniXcoder have their own entry point: python -m src.additional_pretrained_models
    parser.add_argument("--model", choices=["codebert", "graphcodebert", "both"], default="codebert")
    parser.add_argument("--force", action="store_true", help="recompute features and retrain the head")
    parser.add_argument("--save-encoder", action="store_true",
                        help="also save a copy of the (unchanged) encoder weights (~500 MB)")
    args = parser.parse_args()
    ensure_output_dirs()
    shared_data = load_binary_splits()
    for key in (["codebert", "graphcodebert"] if args.model == "both" else [args.model]):
        run_frozen_experiment(key, args.force, data=shared_data, save_encoder=args.save_encoder)
