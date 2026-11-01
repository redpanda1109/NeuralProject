"""
PHASE 3 - Recurrent neural networks (GRU and LSTM) trained from scratch.

Run from the project root:
    python -m src.rnn_models               # trains GRU and LSTM
    python -m src.rnn_models --model gru   # only one of them
    python -m src.rnn_models --force       # retrain even if results exist

Pipeline
--------
code text --tokenize--> tokens --vocabulary--> integer ids --Embedding--> vectors
          --bidirectional GRU/LSTM--> one hidden state per token
          --pool the COMPLETION tokens (mean + max)--> dropout --> Linear --> 1 logit

Input sequence for one example (at most `max_len` tokens):
    <lang_python>  [end of prompt]  <comp>  [completion]  </comp>  [start of suffix]
"""

import argparse
from collections import Counter

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src import config
from src import visualization as viz
from src.evaluation import prediction_path, report_model, save_predictions
from src.preprocessing import (fit_to_budget, load_binary_splits, prompt_tail, suffix_head,
                               tokenize_code)
from src.training import count_parameters, pool_completion, predict, train_model
from src.utils import ensure_output_dirs, get_device, get_logger, save_json, set_seed

PAD, UNK, COMP_START, COMP_END = "<pad>", "<unk>", "<comp>", "</comp>"


# ---------------------------------------------------------------------------
# 1. Tokens and vocabulary
# ---------------------------------------------------------------------------
def example_to_tokens(row, max_len, completion_max_tokens):
    """Build the token sequence and a 0/1 mask marking the completion tokens."""
    prompt_tokens = tokenize_code(prompt_tail(row.prompt))
    completion_tokens = tokenize_code(row.completion) or [UNK]
    suffix_tokens = tokenize_code(suffix_head(row.suffix))
    # 4 special tokens: language, <comp>, </comp> (+1 spare)
    prompt_part, completion_part, suffix_part = fit_to_budget(
        prompt_tokens, completion_tokens, suffix_tokens, max_len - 4, completion_max_tokens)
    tokens = ([f"<lang_{row.language}>"] + prompt_part + [COMP_START] + completion_part +
              [COMP_END] + suffix_part)
    mask = ([0] * (len(prompt_part) + 2) + [1] * len(completion_part) + [0] * (len(suffix_part) + 1))
    return tokens, mask


def build_vocabulary(token_lists, min_freq, max_size):
    """Vocabulary from the TRAINING split only. Rare tokens become <unk>."""
    counts = Counter(token for tokens in token_lists for token in tokens)
    vocab = [PAD, UNK, COMP_START, COMP_END]
    for token, freq in counts.most_common():
        if len(vocab) >= max_size:
            break
        if freq >= min_freq and token not in vocab:
            vocab.append(token)
    return {token: index for index, token in enumerate(vocab)}


# ---------------------------------------------------------------------------
# 2. Dataset and DataLoader
# ---------------------------------------------------------------------------
class CodeSequenceDataset(Dataset):
    def __init__(self, token_lists, masks, labels, vocab):
        unk = vocab[UNK]
        self.ids = [[vocab.get(t, unk) for t in tokens] for tokens in token_lists]
        self.masks = masks
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.masks[i], self.labels[i]


def collate_batch(batch):
    """Pad every sequence in the batch to the same length with <pad> (= id 0)."""
    lengths = torch.tensor([len(ids) for ids, _, _ in batch])
    max_len = int(lengths.max())
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    completion_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ids, mask, _) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        completion_mask[i, :len(mask)] = torch.tensor(mask)
    labels = torch.tensor([label for _, _, label in batch], dtype=torch.float)
    return {"input_ids": input_ids, "lengths": lengths, "completion_mask": completion_mask, "labels": labels}


# ---------------------------------------------------------------------------
# 3. The model
# ---------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    """
    Embedding -> (bi)GRU or (bi)LSTM -> pool completion tokens -> Dropout -> Linear -> logit
    """

    def __init__(self, vocab_size, rnn_type, embedding_dim, hidden_dim, num_layers,
                 bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_class = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_class(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * rnn_output_dim, 1)  # x2: mean + max pooling

    def forward(self, input_ids, lengths, completion_mask):
        embedded = self.dropout(self.embedding(input_ids))            # (B, T, E)
        # Packing tells the RNN the real length of each sequence, so padding is ignored.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True,
                                         total_length=input_ids.size(1))  # (B, T, H*dirs)
        pooled = pool_completion(outputs, completion_mask)               # (B, 2*H*dirs)
        return self.classifier(self.dropout(pooled)).squeeze(-1)         # (B,) logits


# ---------------------------------------------------------------------------
# 4. Experiment
# ---------------------------------------------------------------------------
def prepare_data(cfg, logger, data=None):
    data = data or load_binary_splits()
    tokens, masks = {}, {}
    for split, df in data.items():
        pairs = [example_to_tokens(row, cfg["max_len"], cfg["completion_max_tokens"])
                 for row in df.itertuples(index=False)]
        tokens[split] = [p[0] for p in pairs]
        masks[split] = [p[1] for p in pairs]
    vocab = build_vocabulary(tokens["train"], cfg["min_freq"], cfg["max_vocab_size"])
    unk_rate = {split: sum(t not in vocab for seq in tokens[split] for t in seq) /
                max(sum(len(seq) for seq in tokens[split]), 1) for split in tokens}
    logger.info(f"  Vocabulary (train only): {len(vocab)} tokens | <unk> rate: " +
                ", ".join(f"{s}={r:.2%}" for s, r in unk_rate.items()))
    datasets = {split: CodeSequenceDataset(tokens[split], masks[split], data[split]["label"], vocab)
                for split in data}
    return data, vocab, datasets


def run_rnn_experiment(rnn_type, force=False, cfg=None, data=None):
    cfg = cfg or config.RNN_CONFIG
    logger = get_logger(rnn_type)
    if not force and prediction_path(rnn_type, "test").exists():
        logger.info(f"{rnn_type.upper()} already trained - skipping (use --force to retrain).")
        return
    logger.info("=" * 70)
    logger.info(f"PHASE 3 - {rnn_type.upper()} classifier")
    logger.info("=" * 70)
    set_seed()
    device = get_device()
    data, vocab, datasets = prepare_data(cfg, logger, data)

    generator = torch.Generator().manual_seed(config.SEED)
    loaders = {split: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(split == "train"),
                                 collate_fn=collate_batch, generator=generator if split == "train" else None)
               for split, ds in datasets.items()}

    model = RNNClassifier(len(vocab), rnn_type, cfg["embedding_dim"], cfg["hidden_dim"], cfg["num_layers"],
                          cfg["bidirectional"], cfg["dropout"]).to(device)
    trainable, _ = count_parameters(model)
    logger.info(f"  Model: {model}\n  Trainable parameters: {trainable:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    model_dir = config.MODELS_DIR / rnn_type
    model_dir.mkdir(parents=True, exist_ok=True)
    history = train_model(model, loaders["train"], loaders["validation"], optimizer, device,
                          cfg["max_epochs"], cfg["patience"], model_dir / "best_model.pt", logger,
                          max_grad_norm=5.0)

    # Save everything needed to reload the model later.
    save_json(vocab, model_dir / "vocab.json")
    save_json(cfg, model_dir / "config.json")
    save_json(history, config.METRICS_DIR / f"{rnn_type}_training_history.json")
    viz.plot_loss_curves(history, f"{rnn_type.upper()} - training vs validation loss",
                         config.FIGURES_DIR / f"loss_curve_{rnn_type}.png")

    # Validation + test predictions with the selected (best-validation) checkpoint.
    for split in ["validation", "test"]:
        result = predict(model, loaders[split], device)
        save_predictions(rnn_type, split, data[split], result["probs"], (result["probs"] >= 0.5).astype(int))
    report_model(rnn_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU / LSTM classifiers.")
    parser.add_argument("--model", choices=["gru", "lstm", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="retrain even if saved results exist")
    args = parser.parse_args()
    ensure_output_dirs()
    shared_data = load_binary_splits()
    for name in (["gru", "lstm"] if args.model == "both" else [args.model]):
        run_rnn_experiment(name, args.force, data=shared_data)
