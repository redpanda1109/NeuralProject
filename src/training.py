"""
Training helpers shared by ALL neural models (GRU, LSTM, frozen-encoder head,
fine-tuned CodeBERT). Keeping them in one place avoids copying the same training
loop three times.

Conventions
-----------
* A batch is a dict. `batch["labels"]` holds the 0/1 labels; every other entry is
  passed to the model:  logits = model(**inputs)
* The model returns ONE logit per example (a raw score, not a probability).
  BCEWithLogitsLoss applies the sigmoid internally, which is numerically safer.
* Early stopping and checkpoint selection use the VALIDATION LOSS only.
  The test split is never seen here.
"""

import time

import numpy as np
import torch
from torch import nn


def pool_completion(hidden_states, completion_mask):
    """
    Summarise the hidden states of the COMPLETION tokens only.

    The model reads the whole code window (context + completion), so each completion
    token's hidden state already "knows" its surroundings. We then average (mean) and
    take the strongest activation (max) over those tokens and concatenate both.

    hidden_states   : (batch, seq_len, hidden)
    completion_mask : (batch, seq_len)  1 where the token belongs to the completion
    returns         : (batch, 2 * hidden)
    """
    mask = completion_mask.unsqueeze(-1).to(hidden_states.dtype)
    count = mask.sum(dim=1).clamp(min=1.0)
    mean = (hidden_states * mask).sum(dim=1) / count
    very_negative = torch.finfo(hidden_states.dtype).min
    maximum = hidden_states.masked_fill(mask == 0, very_negative).max(dim=1).values
    return torch.cat([mean, maximum], dim=-1)


def move_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


@torch.no_grad()
def predict(model, loader, device, use_amp=False):
    """Run the model on a DataLoader. Returns labels, probabilities and mean loss."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    all_labels, all_probs, total_loss = [], [], 0.0
    for batch in loader:
        batch = move_to_device(batch, device)
        labels = batch.pop("labels").float()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**batch)
        logits = logits.float()
        total_loss += loss_fn(logits, labels).item()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    labels = np.concatenate(all_labels).astype(int)
    probs = np.concatenate(all_probs)
    return {"labels": labels, "probs": probs, "loss": total_loss / max(len(labels), 1)}


def save_trainable_state(model, path):
    """
    Save only the parameters that are trained. For the RNN that is everything;
    for fine-tuned CodeBERT it is just the last 2 layers + head (~60 MB instead of
    ~500 MB). The frozen weights are identical to the public pretrained model.
    """
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable_names}
    torch.save(state, path)


def load_trainable_state(model, path, device):
    state = torch.load(path, map_location=device)
    missing_trained = [k for k in state if k not in dict(model.named_parameters())]
    if missing_trained:
        raise RuntimeError(f"Checkpoint {path} has unexpected keys: {missing_trained[:5]}")
    model.load_state_dict(state, strict=False)
    return model


def train_model(model, train_loader, val_loader, optimizer, device, max_epochs, patience,
                checkpoint_path, logger, scheduler=None, use_amp=False,
                grad_accumulation_steps=1, max_grad_norm=None):
    """
    Generic training loop with early stopping.

    Each epoch:  train on all training batches  ->  compute validation loss
    If validation loss improved: save a checkpoint.
    If it has not improved for `patience` epochs: stop.
    At the end the BEST checkpoint (lowest validation loss) is loaded back.

    Returns the per-epoch history (list of dicts).
    """
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_val_loss, epochs_without_improvement, history = float("inf"), 0, []

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        running_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            labels = batch.pop("labels").float()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch)
            loss = loss_fn(logits.float(), labels)
            # Gradient accumulation: average the loss over several small batches.
            scaler.scale(loss / grad_accumulation_steps).backward()

            if step % grad_accumulation_steps == 0 or step == len(train_loader):
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)

        train_loss = running_loss / seen
        val = predict(model, val_loader, device, use_amp)
        val_accuracy = float(np.mean((val["probs"] >= 0.5) == val["labels"]))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_accuracy": val_accuracy, "seconds": round(time.time() - start, 1)})

        improved = val["loss"] < best_val_loss - 1e-4
        if improved:
            best_val_loss = val["loss"]
            epochs_without_improvement = 0
            save_trainable_state(model, checkpoint_path)
        else:
            epochs_without_improvement += 1
        logger.info(f"  epoch {epoch:>3} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
                    f"val acc {val_accuracy:.4f} | {time.time() - start:.0f}s"
                    f"{'  <- best, checkpoint saved' if improved else ''}")
        if epochs_without_improvement >= patience:
            logger.info(f"  Early stopping: no validation improvement for {patience} epochs.")
            break

    load_trainable_state(model, checkpoint_path, device)
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    logger.info(f"  Loaded best checkpoint from epoch {best_epoch} (val loss {best_val_loss:.4f}).")
    return history
