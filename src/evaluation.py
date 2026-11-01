"""
PHASE 6 - Evaluation and comparison.

Two kinds of functions live here:

1. Helpers used by every training module
   - compute_binary_metrics()  : accuracy, precision, recall, F1, ROC-AUC, PR-AUC
   - save_predictions()        : one CSV per model and split
   - report_model()            : metrics JSON, classification report, confusion matrix

2. The final comparison (run from the project root):
       python -m src.evaluation
   It reads ONLY the saved prediction files, so every number and chart comes from
   real model outputs. A model without a test-prediction file is marked
   "unavailable" - no values are ever invented.

Positive class: 1 = HALLUCINATED, for every model and every metric.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src import config
from src import visualization as viz
from src.utils import ensure_output_dirs, markdown_table, save_json

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_binary_metrics(y_true, y_score, y_pred):
    """
    y_true  : 0/1 labels
    y_score : any score where HIGHER = more likely hallucinated
              (probability, or SVM decision value) - used for ROC-AUC / PR-AUC
    y_pred  : 0/1 predictions
    """
    y_true, y_score, y_pred = np.asarray(y_true), np.asarray(y_score, dtype=float), np.asarray(y_pred)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=config.POSITIVE_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=config.POSITIVE_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=config.POSITIVE_LABEL, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, y_score) if both_classes else float("nan"),
    }


def pairwise_accuracy(predictions):
    """
    Uses the PAIRED structure of the dataset: for each benchmark item, does the model
    give the hallucinated completion a higher score than the golden one?
    (0.5 = random guessing, 1.0 = always ranks the pair correctly.)
    """
    pivot = predictions.pivot_table(index="pair_id", columns="completion_type", values="score")
    pivot = pivot.dropna()
    if len(pivot) == 0:
        return float("nan")
    wins = (pivot["hallucinated"] > pivot["golden"]).astype(float)
    ties = (pivot["hallucinated"] == pivot["golden"]).astype(float) * 0.5
    return float((wins + ties).mean())


def bootstrap_f1_interval(predictions, n_rounds=1000, seed=config.SEED):
    """
    95% confidence interval for F1, resampling whole benchmark items (pairs) so the
    paired structure is respected. Helps to judge whether a difference is real.
    """
    rng = np.random.default_rng(seed)
    pair_ids = predictions["pair_id"].unique()
    grouped = {pid: idx for pid, idx in predictions.groupby("pair_id").indices.items()}
    labels, preds = predictions["label"].values, predictions["pred"].values
    scores = []
    for _ in range(n_rounds):
        sample = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        idx = np.concatenate([grouped[p] for p in sample])
        scores.append(f1_score(labels[idx], preds[idx], zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ---------------------------------------------------------------------------
# Saving predictions and per-model reports
# ---------------------------------------------------------------------------
def prediction_path(model_name, split):
    return config.PREDICTIONS_DIR / f"{model_name}_{split}.csv"


def save_predictions(model_name, split, binary_df, scores, preds):
    """Save one row per example: ids/analysis columns + label + score + prediction."""
    ensure_output_dirs()
    out = binary_df[["pair_id", "completion_type", "language", "hallucination_type", "label"]].copy()
    out["score"] = np.asarray(scores, dtype=float)
    out["pred"] = np.asarray(preds, dtype=int)
    out.to_csv(prediction_path(model_name, split), index=False)
    return out


def report_model(model_name, splits=("validation", "test")):
    """Metrics JSON + classification report + confusion matrix for one model."""
    results = {}
    class_names = [config.LABEL_NAMES[0], config.LABEL_NAMES[1]]
    for split in splits:
        predictions = pd.read_csv(prediction_path(model_name, split))
        metrics = compute_binary_metrics(predictions["label"], predictions["score"], predictions["pred"])
        metrics["pairwise_accuracy"] = pairwise_accuracy(predictions)
        results[split] = metrics

        report_text = classification_report(predictions["label"], predictions["pred"],
                                            target_names=class_names, digits=4, zero_division=0)
        (config.METRICS_DIR / f"{model_name}_{split}_classification_report.txt").write_text(
            f"{model_name} - {split} split (positive class = 1 = hallucinated)\n\n{report_text}",
            encoding="utf-8")
        if split == "test":
            matrix = confusion_matrix(predictions["label"], predictions["pred"], labels=[0, 1])
            viz.plot_confusion_matrix(matrix, class_names,
                                      f"{config.MODEL_DISPLAY_NAMES.get(model_name, model_name)} - test set",
                                      config.FIGURES_DIR / f"confusion_matrix_{model_name}.png")
        print(f"  [{model_name}] {split:<10} " +
              "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k in METRICS + ["pairwise_accuracy"]))
    save_json(results, config.METRICS_DIR / f"{model_name}_metrics.json")
    return results


# ---------------------------------------------------------------------------
# Final comparison of all models
# ---------------------------------------------------------------------------
def build_comparison_table():
    rows, per_category = [], {}
    test_ids = set(pd.read_csv(config.PROCESSED_DIR / "test.csv", usecols=["benchmark_id"])["benchmark_id"])
    for model_name, display_name in config.MODEL_DISPLAY_NAMES.items():
        path = prediction_path(model_name, "test")
        row = {"model": model_name, "display_name": display_name}
        predictions = pd.read_csv(path) if path.exists() else None
        if predictions is None:
            status = "unavailable (not trained / no test predictions)"
        elif set(predictions["pair_id"]) != test_ids or len(predictions) != 2 * len(test_ids):
            status = "stale (predicted on a different test split - retrain with --force)"
        else:
            status = None
        if status:
            row.update({"status": status})
            row.update({m: np.nan for m in METRICS + ["pairwise_accuracy", "f1_ci_low", "f1_ci_high"]})
            rows.append(row)
            continue
        metrics = compute_binary_metrics(predictions["label"], predictions["score"], predictions["pred"])
        low, high = bootstrap_f1_interval(predictions)
        row.update({"status": "completed", **{m: metrics[m] for m in METRICS},
                    "pairwise_accuracy": pairwise_accuracy(predictions),
                    "f1_ci_low": low, "f1_ci_high": high, "n_test_examples": len(predictions)})
        rows.append(row)
        hallucinated = predictions[predictions["label"] == 1]
        per_category[display_name] = hallucinated.groupby("hallucination_type")["pred"].mean()
    table = pd.DataFrame(rows)
    category_table = pd.DataFrame(per_category).T if per_category else pd.DataFrame()
    return table, category_table


def _best(table, metric):
    done = table[table["status"] == "completed"]
    if done.empty:
        return None
    best = done.loc[done[metric].idxmax()]
    return f"**{best['display_name']}** ({metric} = {best[metric]:.4f})"


def _compare(table, group_a, group_b, label_a, label_b):
    """Compare the best model of group A with the best of group B on test F1."""
    done = table[table["status"] == "completed"].set_index("model")
    a = done.loc[done.index.intersection(group_a)]
    b = done.loc[done.index.intersection(group_b)]
    if a.empty or b.empty:
        missing = label_a if a.empty else label_b
        return f"Cannot answer yet - no completed {missing} model."
    best_a, best_b = a.loc[a["f1"].idxmax()], b.loc[b["f1"].idxmax()]
    diffs = {m: best_b[m] - best_a[m] for m in ["f1", "accuracy", "roc_auc"]}
    # F1 alone can be inflated by predicting "hallucinated" for everything (F1 = 0.67 on
    # balanced data), so the verdict also looks at accuracy and ROC-AUC.
    if all(d > 0 for d in diffs.values()):
        verdict = "Yes"
    elif all(d <= 0 for d in diffs.values()):
        verdict = "No"
    else:
        verdict = "Mixed (the metrics disagree)"
    overlap = not (best_b["f1_ci_low"] > best_a["f1_ci_high"] or best_a["f1_ci_low"] > best_b["f1_ci_high"])
    note = (" The 95% bootstrap F1 intervals overlap, so the difference may not be reliable."
            if overlap else " The 95% bootstrap F1 intervals do not overlap.")
    return (f"{verdict}. Best {label_b}: {best_b['display_name']} (F1 {best_b['f1']:.4f}, "
            f"accuracy {best_b['accuracy']:.4f}, ROC-AUC {best_b['roc_auc']:.4f}) vs best {label_a}: "
            f"{best_a['display_name']} (F1 {best_a['f1']:.4f}, accuracy {best_a['accuracy']:.4f}, "
            f"ROC-AUC {best_a['roc_auc']:.4f}). Differences: F1 {diffs['f1']:+.4f}, "
            f"accuracy {diffs['accuracy']:+.4f}, ROC-AUC {diffs['roc_auc']:+.4f}.{note}")


def write_answers(table):
    baselines = ["logistic_regression", "linear_svm"]
    rnns = ["gru", "lstm"]
    pretrained = ["codebert_frozen", "codebert_finetuned_last2",
                  "graphcodebert_frozen", "graphcodebert_finetuned_last2"]
    answers = [
        "# Final comparison - answers generated from saved test predictions",
        "",
        "Positive class = 1 = hallucinated. All models are evaluated on the same untouched test split.",
        "Each question compares the model with the best test F1 in each group. The verdict is",
        "'Yes' only if F1, accuracy AND ROC-AUC all improve, 'No' if none improve, otherwise 'Mixed'.",
        "",
        f"1. **Did the RNN outperform the traditional baseline?** "
        f"{_compare(table, baselines, rnns, 'baseline', 'RNN')}",
        f"2. **Did the pretrained model outperform the RNN?** "
        f"{_compare(table, rnns, pretrained, 'RNN', 'pretrained model')}",
    ]
    q3 = []
    for prefix in ["codebert", "graphcodebert"]:
        q3.append(f"   - {prefix}: " + _compare(table, [f"{prefix}_frozen"], [f"{prefix}_finetuned_last2"],
                                                f"frozen {prefix}", f"last-2-layers {prefix}"))
    answers += ["3. **Did unfreezing the final two layers improve over the frozen encoder?**"] + q3
    answers += [
        f"4. **Best accuracy:** {_best(table, 'accuracy') or 'no completed models'}",
        f"5. **Best recall for hallucinated code:** {_best(table, 'recall') or 'no completed models'}",
        f"6. **Best F1-score:** {_best(table, 'f1') or 'no completed models'}",
        "",
        "## Results table",
        "",
        markdown_table(table.drop(columns=["model"]), index=False, float_format="{:.4f}"),
    ]
    path = config.METRICS_DIR / "comparison_summary.md"
    path.write_text("\n".join(answers), encoding="utf-8")
    print("\n".join(answers[:16]))
    print(f"\nSummary saved -> {path}")


def make_comparison_figures(table, category_table):
    completed = table[table["status"] == "completed"]
    for metric in ["accuracy", "precision", "recall", "f1"]:
        viz.plot_metric_by_model(table, metric, config.FIGURES_DIR / f"comparison_{metric}.png")
    if completed.empty:
        return
    viz.plot_grouped_metrics(completed, ["accuracy", "precision", "recall", "f1"],
                             "All metrics on the test set (positive class = hallucinated)",
                             config.FIGURES_DIR / "comparison_all_metrics.png")
    for prefix, label in [("codebert", "CodeBERT"), ("graphcodebert", "GraphCodeBERT")]:
        pair = completed[completed["model"].isin([f"{prefix}_frozen", f"{prefix}_finetuned_last2"])]
        if len(pair) == 2:
            viz.plot_grouped_metrics(pair, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                                     f"{label}: frozen encoder vs final 2 layers unfrozen",
                                     config.FIGURES_DIR / f"frozen_vs_finetuned_{prefix}.png")
    if not category_table.empty:
        viz.plot_per_category(category_table, "Detection rate per hallucination category (test set)",
                              config.FIGURES_DIR / "comparison_recall_by_category.png")
    roc_curves, pr_curves = [], []
    for _, row in completed.iterrows():
        predictions = pd.read_csv(prediction_path(row["model"], "test"))
        fpr, tpr, _ = roc_curve(predictions["label"], predictions["score"])
        precision, recall, _ = precision_recall_curve(predictions["label"], predictions["score"])
        roc_curves.append((row["display_name"], fpr, tpr, row["roc_auc"]))
        pr_curves.append((row["display_name"], recall, precision, row["pr_auc"]))
    viz.plot_curves(roc_curves, "roc", config.FIGURES_DIR / "comparison_roc_curves.png")
    viz.plot_curves(pr_curves, "pr", config.FIGURES_DIR / "comparison_pr_curves.png")


def update_readme_results(table):
    """Rewrite the README results section (between the RESULTS markers) from the saved table."""
    readme = config.PROJECT_ROOT / "README.md"
    start, end = "<!-- RESULTS-START -->", "<!-- RESULTS-END -->"
    if Path(config.OUTPUT_DIR).resolve() != (config.PROJECT_ROOT / "outputs").resolve():
        return  # e.g. the smoke test: never put test-run numbers into the README
    if not readme.exists() or start not in readme.read_text(encoding="utf-8"):
        return
    columns = ["display_name", "status", "accuracy", "precision", "recall", "f1",
               "roc_auc", "pr_auc", "pairwise_accuracy", "f1_ci_low", "f1_ci_high"]
    shown = table[columns].rename(columns={"display_name": "model"})
    task_b_path = config.METRICS_DIR / "taskB_category_results.csv"
    task_b = ""
    if task_b_path.exists():
        tb = pd.read_csv(task_b_path)
        tb = tb[tb["split"] == "test"].drop(columns="split")
        task_b = ("\n\n**Task B (4-class category, hallucinated completions only, test split):**\n\n"
                  + markdown_table(tb, index=False, float_format="{:.4f}"))
    answers_path = config.METRICS_DIR / "comparison_summary.md"
    answers = ""
    if answers_path.exists():
        lines = answers_path.read_text(encoding="utf-8").split("\n## Results table")[0].splitlines()
        answers = "\n".join(line for line in lines[1:] if line.strip())
    n_test = table["n_test_examples"].max() if "n_test_examples" in table else float("nan")
    n_test = "n/a" if pd.isna(n_test) else int(n_test)
    section = (f"{start}\n_Generated automatically by `python -m src.evaluation` from the saved test "
               f"predictions - do not edit by hand. Positive class = 1 = hallucinated; "
               f"test split = {n_test} examples. Models marked unavailable have not been trained yet._\n\n"
               f"{markdown_table(shown, index=False, float_format='{:.4f}')}{task_b}\n\n{answers}\n\n"
               f"Figures: `outputs/figures/comparison_*.png`, `frozen_vs_finetuned_*.png`, "
               f"`confusion_matrix_*.png`, `loss_curve_*.png`.\n{end}")
    text = readme.read_text(encoding="utf-8")
    before, rest = text.split(start, 1)
    after = rest.split(end, 1)[1]
    readme.write_text(before + section + after, encoding="utf-8")
    print(f"README results section updated -> {readme}")


def run_comparison():
    ensure_output_dirs()
    print("=" * 70)
    print("PHASE 6 - COMPARING ALL MODELS ON THE TEST SET")
    print("=" * 70)
    table, category_table = build_comparison_table()
    table.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    if not category_table.empty:
        category_table.to_csv(config.METRICS_DIR / "model_recall_by_category.csv")
    print(table[["display_name", "status", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .to_string(index=False, float_format="%.4f"))
    print(f"\nTable saved -> {config.METRICS_DIR / 'model_comparison.csv'}")
    make_comparison_figures(table, category_table)
    write_answers(table)
    update_readme_results(table)
    return table


if __name__ == "__main__":
    run_comparison()
