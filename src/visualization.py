"""
All plotting code lives here so every figure in the project looks consistent.

Every chart is drawn from data passed in (or loaded from saved CSV/JSON files) -
there are NO hand-typed metric values anywhere.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; works without a display (Colab, servers)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Fixed categorical palette (colour-blind checked), always used in this order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE_COLOR = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f7fd", "#9ec5f4", "#2a78d6", "#104281"])


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE_COLOR,
        "axes.facecolor": SURFACE_COLOR,
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED_COLOR,
        "ytick.color": MUTED_COLOR,
        "legend.frameon": False,
        "font.size": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


apply_style()


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved figure -> {path}")


def _label_bars(ax, bars, fmt="{:.3f}", horizontal=False):
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        if np.isnan(value):
            continue
        if horizontal:
            ax.annotate(fmt.format(value), (value, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=9, color=TEXT_COLOR)
        else:
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=TEXT_COLOR)


# ---------------------------------------------------------------------------
# Dataset exploration
# ---------------------------------------------------------------------------
def plot_counts(counts, title, xlabel, path, show_percent=True):
    """Horizontal bar chart of category counts (sorted largest first)."""
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(counts) + 1.5))
    bars = ax.barh(counts.index.astype(str), counts.values, color=PALETTE[0], height=0.6)
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        label = f"{value}  ({100 * value / total:.1f}%)" if show_percent else f"{value}"
        ax.annotate(label, (value, bar.get_y() + bar.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_COLOR)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.2)
    _save(fig, path)


def plot_crosstab(table, title, path):
    """Grouped bars: rows of `table` on the x-axis, one colour per column."""
    n_groups, n_series = table.shape
    width = 0.8 / n_series
    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, column in enumerate(table.columns):
        ax.bar(x + i * width - 0.4 + width / 2, table[column].values, width * 0.92,
               label=str(column), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_title(title)
    ax.set_ylabel("Number of benchmark items")
    ax.set_ylim(0, table.values.max() * 1.25)  # headroom for the legend
    ax.grid(axis="x", visible=False)
    ax.legend(title=table.columns.name, ncols=n_series, loc="upper left")
    _save(fig, path)


def plot_histograms(series_dict, title, xlabel, path, log_x=True, bins=50):
    """Overlaid step histograms (one line per series)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in series_dict.values()])
    all_values = all_values[all_values > 0] if log_x else all_values
    if log_x:
        edges = np.logspace(np.log10(all_values.min()), np.log10(all_values.max()), bins)
        ax.set_xscale("log")
    else:
        edges = np.linspace(all_values.min(), all_values.max(), bins)
    for i, (name, values) in enumerate(series_dict.items()):
        values = np.asarray(values, dtype=float)
        ax.hist(values, bins=edges, histtype="step", linewidth=2, color=PALETTE[i],
                label=f"{name} (median {np.median(values):.0f})")
    ax.set_title(title)
    ax.set_xlabel(xlabel + (" (log scale)" if log_x else ""))
    ax.set_ylabel("Number of records")
    ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def plot_loss_curves(history, title, path):
    """Training vs validation loss per epoch, with the selected (best) epoch marked."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    best_epoch = epochs[int(np.argmin(val_loss))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color=PALETTE[0], linewidth=2, marker="o", markersize=5, label="Training loss")
    ax.plot(epochs, val_loss, color=PALETTE[1], linewidth=2, marker="o", markersize=5, label="Validation loss")
    ax.axvline(best_epoch, color=MUTED_COLOR, linestyle="--", linewidth=1)
    ax.annotate(f"best epoch = {best_epoch}", (best_epoch, max(train_loss + val_loss)),
                xytext=(5, 0), textcoords="offset points", color=MUTED_COLOR, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.legend()
    _save(fig, path)


def plot_confusion_matrix(matrix, class_names, title, path):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(0.9 * len(class_names) + 3.5, 0.9 * len(class_names) + 2.8))
    ax.imshow(matrix, cmap=BLUES)
    ax.grid(False)
    threshold = matrix.max() / 2
    row_totals = matrix.sum(axis=1, keepdims=True).clip(min=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "#ffffff" if matrix[i, j] > threshold else TEXT_COLOR
            ax.text(j, i, f"{matrix[i, j]}\n({100 * matrix[i, j] / row_totals[i, 0]:.0f}%)",
                    ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20 if len(class_names) > 2 else 0)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    _save(fig, path)


def plot_metric_by_model(comparison_df, metric, path):
    """One bar per model for a single metric. Unavailable models are shown as 'n/a'."""
    names = comparison_df["display_name"].tolist()
    values = comparison_df[metric].astype(float).values
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.5))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, np.nan_to_num(values), color=PALETTE[0], height=0.6)
    _label_bars(ax, [b for b, v in zip(bars, values) if not np.isnan(v)], horizontal=True)
    for yi, v in zip(y, values):
        if np.isnan(v):
            ax.annotate("n/a (not trained)", (0, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=9, color=MUTED_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(f"{metric.replace('_', ' ').title()} on the test set (positive class = hallucinated)")
    ax.set_title(f"Test {metric.replace('_', ' ').title()} by model")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_grouped_metrics(comparison_df, metrics, title, path):
    """Grouped bars: models on the x-axis, one colour per metric."""
    names = comparison_df["display_name"].tolist()
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names) + 2), 5.5))
    for i, metric in enumerate(metrics):
        values = comparison_df[metric].astype(float).values
        bars = ax.bar(x + i * width - 0.4 + width / 2, np.nan_to_num(values), width * 0.92,
                      color=PALETTE[i], label=metric.replace("_", " ").title())
        _label_bars(ax, [b for b, v in zip(bars, values) if not np.isnan(v)], fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score on the test set")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=len(metrics), loc="upper left")
    _save(fig, path)


def plot_curves(curves, kind, path):
    """ROC or PR curves. `curves` = list of (display_name, x_values, y_values, auc)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, xs, ys, auc) in enumerate(curves):
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], linewidth=2, label=f"{name} (AUC {auc:.3f})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=AXIS_COLOR, linestyle="--", linewidth=1, label="Random guess")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.set_title("ROC curves on the test set")
        ax.legend(loc="lower right")
    else:
        ax.set_xlabel("Recall (hallucinated)")
        ax.set_ylabel("Precision (hallucinated)")
        ax.set_title("Precision-recall curves on the test set")
        ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _save(fig, path)


def plot_per_category(table, title, path):
    """Recall on hallucinated examples, split by hallucination category, per model."""
    # table: rows = models, columns = categories
    n_models, n_cats = table.shape
    width = 0.8 / n_cats
    x = np.arange(n_models)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_models + 2), 5.5))
    for i, column in enumerate(table.columns):
        values = table[column].astype(float).values
        bars = ax.bar(x + i * width - 0.4 + width / 2, np.nan_to_num(values), width * 0.92,
                      color=PALETTE[i], label=column)
        _label_bars(ax, bars, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall on hallucinated completions")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Hallucination type", ncols=n_cats, loc="upper left")
    _save(fig, path)
