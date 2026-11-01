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

3. The classification reports of EVERY model -> outputs/classification_report.md
       python -m src.evaluation                            # also writes the report
       python -m src.evaluation --classification-report-only
   Rebuilt from the saved test predictions each time, so it is always reproducible.
   Models without (current) test predictions are listed as "Not yet evaluated".

Positive class: 1 = HALLUCINATED, for every model and every metric.
"""

import argparse
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

# Encoders that have both a frozen (Phase 4) and a last-2-layers (Phase 5) experiment.
FROZEN_FINETUNED_PAIRS = {"codebert": "CodeBERT", "graphcodebert": "GraphCodeBERT",
                          "codet5_small": "CodeT5-small", "unixcoder": "UniXcoder"}
# All CodeBERT variants, compared in one extra chart.
CODEBERT_VARIANTS = ["codebert_frozen", "codebert_extra_transformer_layer",
                     "codebert_finetuned_last2", "codebert_finetuned_last4"]


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
def current_test_ids():
    return set(pd.read_csv(config.PROCESSED_DIR / "test.csv", usecols=["benchmark_id"])["benchmark_id"])


def load_test_predictions(model_name, test_ids=None):
    """
    (predictions, status) for one model. status is None when the saved test predictions
    exist AND belong to the current test split; otherwise a short explanation.
    """
    test_ids = current_test_ids() if test_ids is None else test_ids
    path = prediction_path(model_name, "test")
    if not path.exists():
        return None, "unavailable (not trained / no test predictions)"
    predictions = pd.read_csv(path)
    if set(predictions["pair_id"]) != test_ids or len(predictions) != 2 * len(test_ids):
        return None, "stale (predicted on a different test split - retrain with --force)"
    return predictions, None


def build_comparison_table():
    rows, per_category = [], {}
    test_ids = current_test_ids()
    for model_name, display_name in config.MODEL_DISPLAY_NAMES.items():
        row = {"model": model_name, "display_name": display_name}
        predictions, status = load_test_predictions(model_name, test_ids)
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
    pretrained = [m for m in config.MODEL_DISPLAY_NAMES if m not in baselines + rnns]
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
    for prefix in FROZEN_FINETUNED_PAIRS:
        q3.append(f"   - {prefix}: " + _compare(table, [f"{prefix}_frozen"], [f"{prefix}_finetuned_last2"],
                                                f"frozen {prefix}", f"last-2-layers {prefix}"))
    answers += ["3. **Did unfreezing the final two layers improve over the frozen encoder?**"] + q3
    answers += [
        f"4. **Best accuracy:** {_best(table, 'accuracy') or 'no completed models'}",
        f"5. **Best recall for hallucinated code:** {_best(table, 'recall') or 'no completed models'}",
        f"6. **Best F1-score:** {_best(table, 'f1') or 'no completed models'}",
        "",
        "Additional CodeBERT experiments:",
        "",
        f"7. **Did unfreezing the last FOUR layers improve over the last two?** "
        f"{_compare(table, ['codebert_finetuned_last2'], ['codebert_finetuned_last4'], 'last-2-layers CodeBERT', 'last-4-layers CodeBERT')}",
        f"8. **Did adding a new Transformer layer on top of frozen CodeBERT improve over the frozen encoder + head?** "
        f"{_compare(table, ['codebert_frozen'], ['codebert_extra_transformer_layer'], 'frozen CodeBERT', 'frozen CodeBERT + new layer')}",
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
    for prefix, label in FROZEN_FINETUNED_PAIRS.items():
        pair = completed[completed["model"].isin([f"{prefix}_frozen", f"{prefix}_finetuned_last2"])]
        if len(pair) == 2:
            viz.plot_grouped_metrics(pair, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                                     f"{label}: frozen encoder vs final 2 layers unfrozen",
                                     config.FIGURES_DIR / f"frozen_vs_finetuned_{prefix}.png")
    variants = completed[completed["model"].isin(CODEBERT_VARIANTS)]
    if len(variants) >= 2:
        viz.plot_grouped_metrics(variants, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                                 "CodeBERT: frozen vs last 2 / last 4 layers vs new Transformer layer",
                                 config.FIGURES_DIR / "codebert_variants.png")
    if not category_table.empty:
        viz.plot_per_category(category_table, "Detection rate per hallucination category (test set)",
                              config.FIGURES_DIR / "comparison_recall_by_category.png")
    # At most 8 curves per chart (one fixed colour each, never cycled): the original 8
    # models keep their chart and colours; the additional experiments get their own.
    original = list(config.MODEL_DISPLAY_NAMES)[:8]
    for models, suffix in [(original, ""), ([m for m in config.MODEL_DISPLAY_NAMES if m not in original],
                                            "_additional")]:
        roc_curves, pr_curves = [], []
        for _, row in completed[completed["model"].isin(models)].iterrows():
            predictions = pd.read_csv(prediction_path(row["model"], "test"))
            fpr, tpr, _ = roc_curve(predictions["label"], predictions["score"])
            precision, recall, _ = precision_recall_curve(predictions["label"], predictions["score"])
            roc_curves.append((row["display_name"], fpr, tpr, row["roc_auc"]))
            pr_curves.append((row["display_name"], recall, precision, row["pr_auc"]))
        if roc_curves:
            viz.plot_curves(roc_curves, "roc", config.FIGURES_DIR / f"comparison_roc_curves{suffix}.png")
            viz.plot_curves(pr_curves, "pr", config.FIGURES_DIR / f"comparison_pr_curves{suffix}.png")


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


# ---------------------------------------------------------------------------
# Classification reports of every model -> outputs/classification_report.md
# ---------------------------------------------------------------------------
RUN_COMMANDS = {
    "logistic_regression": "python -m src.baseline_models",
    "linear_svm": "python -m src.baseline_models",
    "gru": "python -m src.rnn_models --model gru",
    "lstm": "python -m src.rnn_models --model lstm",
    "codebert_frozen": "python -m src.pretrained_models --model codebert",
    "codebert_finetuned_last2": "python -m src.finetuning --model codebert",
    "graphcodebert_frozen": "python -m src.pretrained_models --model graphcodebert",
    "graphcodebert_finetuned_last2": "python -m src.finetuning --model graphcodebert",
    "codet5_small_frozen": "python -m src.additional_pretrained_models --model codet5_small",
    "codet5_small_finetuned_last2": "python -m src.finetuning --model codet5_small",
    "unixcoder_frozen": "python -m src.additional_pretrained_models --model unixcoder",
    "unixcoder_finetuned_last2": "python -m src.finetuning --model unixcoder",
    "codebert_finetuned_last4": "python -m src.more_finetuning_codebert",
    "codebert_extra_transformer_layer": "python -m src.adding_layers_codebert",
}


def classification_report_text(predictions):
    """sklearn classification report with the project's label names (0 = correct, 1 = hallucinated)."""
    return classification_report(predictions["label"], predictions["pred"], labels=[0, 1],
                                 target_names=[config.LABEL_NAMES[0], config.LABEL_NAMES[1]],
                                 digits=4, zero_division=0)


def evaluated_models():
    """
    One row per model with CURRENT saved test predictions (stale/missing ones excluded),
    sorted by the primary metric (config.PRIMARY_METRIC, test F1 of the hallucinated
    class), ties broken by ROC-AUC. Nothing is retrained - only saved files are read.
    """
    rows, test_ids = [], current_test_ids()
    for model_name, display_name in config.MODEL_DISPLAY_NAMES.items():
        predictions, status = load_test_predictions(model_name, test_ids)
        if status:
            continue
        metrics = compute_binary_metrics(predictions["label"], predictions["score"], predictions["pred"])
        rows.append({"model": model_name, "display_name": display_name, **metrics})
    columns = ["model", "display_name"] + METRICS + ["macro_f1"]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values([config.PRIMARY_METRIC, config.PRIMARY_METRIC_TIEBREAK],
                             ascending=False, kind="stable").reset_index(drop=True)


def top_models(k=2):
    """The k best evaluated models by the primary metric (fewer if fewer are evaluated)."""
    return evaluated_models().head(k)


def _setup_type(model_name):
    return config.MODEL_TRAINING_SETUP.get(model_name, "").split(" - ")[0] or "n/a"


def write_classification_reports(path=None):
    """Write outputs/classification_report.md from the saved test predictions."""
    ensure_output_dirs()
    path = Path(path or config.OUTPUT_DIR / "classification_report.md")
    test_ids = current_test_ids()
    ranking = evaluated_models()
    n_items = len(test_ids)
    metric, tiebreak = config.PRIMARY_METRIC, config.PRIMARY_METRIC_TIEBREAK
    lines = [
        "# Classification reports - all models and experiments",
        "",
        "_Generated automatically by `python -m src.evaluation` (or "
        "`python -m src.evaluation --classification-report-only`) from the saved test predictions in "
        "`outputs/metrics/predictions/`. Do not edit by hand - re-run the command after training more models._",
        "",
        f"- **Task:** binary detection. Label **0 = correct** (golden completion), **1 = hallucinated** "
        f"(hallucinated completion); the positive class is 1 = hallucinated.",
        f"- **Test set:** the saved split `data/processed/test.csv` ({n_items} benchmark items = "
        f"{2 * n_items} binary examples, 50% per class). It is never used for training, tuning or early stopping.",
        "- **Predictions:** exactly as saved by each model (neural models: probability >= 0.5; "
        "Linear SVM: decision value >= 0).",
        "- A model is **Not yet evaluated** when it has no saved test predictions for the current split; "
        "no values are filled in for it.",
        "",
        "## Summary - top two models",
        "",
        f"**Primary metric: test F1-score of the hallucinated class (`{metric}`), ties broken by "
        f"ROC-AUC (`{tiebreak}`).** This is the metric the project's comparison already uses "
        "(`comparison_summary.md` picks the best model of each group by test F1). For a detector, "
        "F1 of the positive class balances precision (how many flagged completions really are "
        "hallucinated) and recall (how many hallucinations are caught). F1 alone can be inflated by "
        "predicting everything as hallucinated (F1 = 0.67 on this balanced test set), so accuracy "
        "and ROC-AUC are shown next to it.",
        "",
    ]
    if ranking.empty:
        lines += ["No model has been evaluated yet.", ""]
    else:
        top = ranking.head(2)
        summary_rows, intervals = [], []
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            predictions, _ = load_test_predictions(row.model, test_ids)
            low, high = bootstrap_f1_interval(predictions)
            intervals.append((low, high))
            summary_rows.append({"rank": rank, "model": row.display_name, "type": _setup_type(row.model),
                                 "f1": row.f1, "accuracy": row.accuracy, "roc_auc": row.roc_auc,
                                 "f1_95%_CI": f"[{low:.4f}, {high:.4f}]"})
        lines += [markdown_table(pd.DataFrame(summary_rows), index=False, float_format="{:.4f}"), ""]
        if len(top) == 2:
            (low_1, high_1), (low_2, high_2) = intervals
            overlap = not (low_1 > high_2 or low_2 > high_1)
            lines += [f"The two best models by test F1 are **{top.iloc[0]['display_name']}** and "
                      f"**{top.iloc[1]['display_name']}**. Their 95% bootstrap F1 intervals "
                      + ("overlap, so the order between them is not reliable" if overlap
                         else "do not overlap") +
                      f" (test set of {n_items} items; intervals of all models are in "
                      "`outputs/metrics/model_comparison.csv`).", ""]
        else:
            lines += ["Only one model has been evaluated so far.", ""]

    lines += ["## Overview", ""]
    overview = []
    ranks = {m: i + 1 for i, m in enumerate(ranking["model"])}
    for model_name, display_name in config.MODEL_DISPLAY_NAMES.items():
        _, status = load_test_predictions(model_name, test_ids)
        row = ranking[ranking["model"] == model_name]
        overview.append({
            "model": display_name, "type": _setup_type(model_name),
            "status": "Evaluated" if status is None else ("Not yet evaluated" if status.startswith("unavailable")
                                                         else "Not yet evaluated (stale predictions)"),
            "rank by F1": ranks.get(model_name, "-"),
            "f1": float(row["f1"].iloc[0]) if len(row) else float("nan"),
            "accuracy": float(row["accuracy"].iloc[0]) if len(row) else float("nan"),
        })
    lines += [markdown_table(pd.DataFrame(overview), index=False, float_format="{:.4f}"), ""]

    lines += ["## Binary task - classification report of every model (test set)", ""]
    for number, (model_name, display_name) in enumerate(config.MODEL_DISPLAY_NAMES.items(), start=1):
        predictions, status = load_test_predictions(model_name, test_ids)
        lines += [f"### {number}. {display_name} (`{model_name}`)", "",
                  f"- **Setup:** {config.MODEL_TRAINING_SETUP.get(model_name, 'n/a')}"]
        if status:
            reason = ("no saved test predictions" if status.startswith("unavailable")
                      else "saved predictions belong to a different test split")
            lines += [f"- **Status: Not yet evaluated** ({reason}: "
                      f"`outputs/metrics/predictions/{model_name}_test.csv`).",
                      f"- Run: `{RUN_COMMANDS.get(model_name, 'see INSTRUCTIONS.md')}`, then "
                      "`python -m src.evaluation`.", ""]
            continue
        lines += [f"- **Status:** Evaluated - {len(predictions)} test examples, "
                  f"rank {ranks[model_name]} of {len(ranking)} by test F1.", "",
                  "```text", classification_report_text(predictions).rstrip(), "```", ""]

    lines += ["## Task B (4-class hallucination category) - separate task, not ranked above", "",
              "Only hallucinated completions; labels = the four hallucination types. Shown for "
              "completeness because these baselines are also part of the project.", ""]
    for kind in ["logistic_regression", "linear_svm"]:
        task_b_path = config.PREDICTIONS_DIR / f"taskB_{kind}_test.csv"
        lines += [f"### Task B - {config.MODEL_DISPLAY_NAMES[kind]} (`taskB_{kind}`)", ""]
        if not task_b_path.exists():
            lines += ["- **Status: Not yet evaluated** (run `python -m src.baseline_models --task b`).", ""]
            continue
        task_b = pd.read_csv(task_b_path)
        if set(task_b["pair_id"]) != test_ids:
            lines += ["- **Status: Not yet evaluated** (stale predictions - retrain with `--force`).", ""]
            continue
        report = classification_report(task_b["label"], task_b["pred"], labels=list(range(len(config.CATEGORY_NAMES))),
                                       target_names=config.CATEGORY_NAMES, digits=4, zero_division=0)
        lines += [f"- **Status:** Evaluated - {len(task_b)} hallucinated test completions.", "",
                  "```text", report.rstrip(), "```", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Classification reports saved -> {path}")
    return path


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
    write_classification_reports()
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: compare all models from saved predictions.")
    parser.add_argument("--classification-report-only", action="store_true",
                        help="only (re)write outputs/classification_report.md")
    args = parser.parse_args()
    if args.classification_report_only:
        write_classification_reports()
    else:
        run_comparison()
