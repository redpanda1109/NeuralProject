"""
PHASE 2 - Traditional machine-learning baselines (TF-IDF + linear models).

Run from the project root:
    python -m src.baseline_models            # Task A (binary) + Task B (4 categories)
    python -m src.baseline_models --force    # retrain even if results exist

How it works
------------
1. Turn every example into text (see preprocessing.build_tfidf_text).
2. TF-IDF turns text into a sparse vector of weighted token / character n-gram counts.
   The vectorizer is FIT ON THE TRAINING SPLIT ONLY and then applied to val/test.
3. Train Logistic Regression and a Linear SVM for every combination of
   (text view, feature type, C) and keep the combination with the best
   VALIDATION macro-F1. The test split is used once, at the very end.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from src import config
from src import visualization as viz
from src.evaluation import compute_binary_metrics, prediction_path, report_model, save_predictions
from src.preprocessing import build_tfidf_text, hallucinated_only, load_binary_splits, load_split
from src.utils import ensure_output_dirs, get_logger, save_json, set_seed

CODE_TOKEN_REGEX = r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
class TfidfFeatures:
    """Word-level, character-level, or both TF-IDF vectorizers combined."""

    def __init__(self, feature_type):
        self.feature_type = feature_type
        self.vectorizers = []
        if feature_type in ("word", "word+char"):
            self.vectorizers.append(TfidfVectorizer(
                token_pattern=CODE_TOKEN_REGEX, lowercase=False, ngram_range=(1, 2),
                min_df=2, max_features=50000, sublinear_tf=True))
        if feature_type in ("char", "word+char"):
            self.vectorizers.append(TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), lowercase=False,
                min_df=2, max_features=100000, sublinear_tf=True))
        if not self.vectorizers:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def fit_transform(self, texts):  # TRAINING data only
        return hstack([v.fit_transform(texts) for v in self.vectorizers]).tocsr()

    def transform(self, texts):      # validation / test: no refitting
        return hstack([v.transform(texts) for v in self.vectorizers]).tocsr()


def make_classifier(kind, C):
    if kind == "logistic_regression":
        return LogisticRegression(C=C, max_iter=3000, random_state=config.SEED)
    if kind == "linear_svm":
        return LinearSVC(C=C, max_iter=10000, random_state=config.SEED)
    raise ValueError(kind)


def classifier_scores(model, X):
    """Score where higher = more likely class 1 (binary) or per-class scores (multi-class)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, 1] if probs.shape[1] == 2 else probs
    return model.decision_function(X)


# ---------------------------------------------------------------------------
# Grid search on the VALIDATION split
# ---------------------------------------------------------------------------
def grid_search(train_df, val_df, task, logger):
    """
    Try every (view, feature type, classifier, C). Returns the grid results and the
    best configuration per classifier, chosen by validation macro-F1.
    """
    cfg = config.BASELINE_CONFIG
    results, fitted = [], {}
    for view in cfg["views"]:
        train_text = build_tfidf_text(train_df, view, cfg["context_prompt_chars"], cfg["context_suffix_chars"])
        val_text = build_tfidf_text(val_df, view, cfg["context_prompt_chars"], cfg["context_suffix_chars"])
        for feature_type in cfg["feature_types"]:
            features = TfidfFeatures(feature_type)
            X_train = features.fit_transform(train_text)
            X_val = features.transform(val_text)
            for kind, C_values in [("logistic_regression", cfg["logreg_C"]), ("linear_svm", cfg["svm_C"])]:
                for C in C_values:
                    model = make_classifier(kind, C).fit(X_train, train_df["label"])
                    val_pred = model.predict(X_val)
                    score = f1_score(val_df["label"], val_pred, average="macro")
                    results.append({"task": task, "classifier": kind, "view": view, "features": feature_type,
                                    "C": C, "n_features": X_train.shape[1], "val_macro_f1": score})
                    key = (kind, view, feature_type, C)
                    fitted[key] = (features, model)
                    logger.info(f"  {task} {kind:<20} view={view:<19} features={feature_type:<9} "
                                f"C={C:<6} val macro-F1={score:.4f}")
    results = pd.DataFrame(results)
    best = {}
    for kind in ["logistic_regression", "linear_svm"]:
        rows = results[results["classifier"] == kind]
        top = rows.loc[rows["val_macro_f1"].idxmax()]
        key = (kind, top["view"], top["features"], top["C"])
        best[kind] = {"config": top.to_dict(), "features": fitted[key][0], "model": fitted[key][1]}
        logger.info(f"  BEST {task} {kind}: view={top['view']} features={top['features']} "
                    f"C={top['C']} (val macro-F1={top['val_macro_f1']:.4f})")
    return results, best


# ---------------------------------------------------------------------------
# Task A - binary hallucination detection
# ---------------------------------------------------------------------------
def run_task_a(force=False):
    logger = get_logger("baselines")
    names = ["logistic_regression", "linear_svm"]
    if not force and all(prediction_path(n, "test").exists() for n in names):
        logger.info("Task A baselines already trained - skipping (use --force to retrain).")
        return
    logger.info("=" * 70)
    logger.info("PHASE 2 - TASK A (binary): TF-IDF baselines")
    logger.info("=" * 70)
    set_seed()
    data = load_binary_splits()
    for name, df in data.items():
        logger.info(f"  {name:<10}: {len(df)} examples ({df['label'].mean():.0%} hallucinated)")

    grid, best = grid_search(data["train"], data["validation"], "binary", logger)
    grid.to_csv(config.METRICS_DIR / "baseline_grid_search_binary.csv", index=False)
    cfg = config.BASELINE_CONFIG

    for kind in names:
        features, model = best[kind]["features"], best[kind]["model"]
        view = best[kind]["config"]["view"]
        for split in ["validation", "test"]:  # test is touched only now, after selection
            df = data[split]
            X = features.transform(build_tfidf_text(df, view, cfg["context_prompt_chars"], cfg["context_suffix_chars"]))
            save_predictions(kind, split, df, classifier_scores(model, X), model.predict(X))
        model_dir = config.MODELS_DIR / kind
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(features, model_dir / "tfidf_vectorizer.joblib")
        joblib.dump(model, model_dir / "model.joblib")
        save_json(best[kind]["config"], model_dir / "selected_config.json")
        logger.info(f"Saved {kind} to {model_dir}")
        report_model(kind)


# ---------------------------------------------------------------------------
# Task B - which TYPE of hallucination? (4 classes, hallucinated completions only)
# ---------------------------------------------------------------------------
def run_task_b(force=False):
    logger = get_logger("baselines")
    summary_path = config.METRICS_DIR / "taskB_category_results.csv"
    if not force and summary_path.exists():
        logger.info("Task B baselines already trained - skipping (use --force to retrain).")
        return
    logger.info("=" * 70)
    logger.info("TASK B (4-class hallucination CATEGORY - NOT the binary detection task)")
    logger.info("=" * 70)
    set_seed()
    data = {split: hallucinated_only(load_split(split)) for split in ["train", "validation", "test"]}
    majority_share = data["test"]["category"].value_counts(normalize=True).max()
    logger.info(f"  Test-set majority-class share (a trivial reference): {majority_share:.3f}")

    grid, best = grid_search(data["train"], data["validation"], "category", logger)
    grid.to_csv(config.METRICS_DIR / "baseline_grid_search_category.csv", index=False)
    cfg = config.BASELINE_CONFIG
    rows = []
    for kind in ["logistic_regression", "linear_svm"]:
        features, model = best[kind]["features"], best[kind]["model"]
        view = best[kind]["config"]["view"]
        for split in ["validation", "test"]:
            df = data[split]
            X = features.transform(build_tfidf_text(df, view, cfg["context_prompt_chars"], cfg["context_suffix_chars"]))
            pred = model.predict(X)
            out = df[["pair_id", "language", "category", "label"]].copy()
            out["pred"] = pred
            out["pred_category"] = [config.CATEGORY_NAMES[i] for i in pred]
            out.to_csv(config.PREDICTIONS_DIR / f"taskB_{kind}_{split}.csv", index=False)
            accuracy = float(np.mean(pred == df["label"].values))
            macro_f1 = f1_score(df["label"], pred, average="macro")
            rows.append({"classifier": kind, "split": split, "accuracy": accuracy, "macro_f1": macro_f1,
                         "majority_class_share": df["category"].value_counts(normalize=True).max()})
            logger.info(f"  [taskB {kind}] {split:<10} accuracy={accuracy:.4f} macro-F1={macro_f1:.4f}")
            if split == "test":
                report = classification_report(df["label"], pred, labels=range(4),
                                               target_names=config.CATEGORY_NAMES, digits=4, zero_division=0)
                (config.METRICS_DIR / f"taskB_{kind}_test_classification_report.txt").write_text(
                    f"Task B (4-class category) - {kind} - test\n\n{report}", encoding="utf-8")
                matrix = confusion_matrix(df["label"], pred, labels=range(4))
                viz.plot_confusion_matrix(matrix, config.CATEGORY_NAMES,
                                          f"Task B category - {config.MODEL_DISPLAY_NAMES[kind]} (test)",
                                          config.FIGURES_DIR / f"taskB_confusion_matrix_{kind}.png")
        model_dir = config.MODELS_DIR / f"taskB_{kind}"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(features, model_dir / "tfidf_vectorizer.joblib")
        joblib.dump(model, model_dir / "model.joblib")
        save_json(best[kind]["config"], model_dir / "selected_config.json")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    logger.info(f"Task B results -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF baselines.")
    parser.add_argument("--force", action="store_true", help="retrain even if saved results exist")
    parser.add_argument("--task", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()
    ensure_output_dirs()
    if args.task in ("a", "both"):
        run_task_a(args.force)
    if args.task in ("b", "both"):
        run_task_b(args.force)
