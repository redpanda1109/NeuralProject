"""
Train / validation / test split (70 / 15 / 15).

Run from the project root:
    python -m src.data_split                      # default: group by code context
    python -m src.data_split --group-by repo_url  # stricter: whole repositories per split

Why split BEFORE expanding the pairs?
    Every row holds a golden AND a hallucinated completion that share the same code
    context. We split the ORIGINAL rows and only expand them into binary examples
    afterwards, so both completions of a benchmark_id always land in the same split.

Why group by code context and not only by benchmark_id?
    benchmark_id is unique per row, but the benchmark often creates 2-3 items
    (e.g. an `import` AND a `parameter` hallucination) from the SAME place in the SAME
    file. Those items share prompt, suffix and golden completion, so their golden
    examples are byte-for-byte identical. A plain row split put 163 of 293 test items'
    golden examples into training as well - that is leakage. We therefore give every
    distinct (prompt, suffix) a `context_id` and keep each context_id inside one split.
    This is strictly stronger than splitting by benchmark_id: every benchmark_id still
    belongs to exactly one split.

Stratification
    Groups are split with a stratified split on the group's language, so all seven
    languages keep the same proportions. Hallucination categories end up close to
    their overall proportions as well (printed below and saved to
    outputs/metrics/split_category_counts.csv).
"""

import argparse
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.preprocessing import load_raw_data
from src.utils import ensure_output_dirs, save_json, set_seed


def add_context_id(df):
    """Same prompt + same suffix  ->  same context_id (short hash)."""
    df = df.copy()
    text = df["prompt"].fillna("") + "\x00" + df["suffix"].fillna("")
    df["context_id"] = [hashlib.md5(t.encode("utf-8")).hexdigest()[:12] for t in text]
    return df


def grouped_split(df, group_column, seed=config.SEED):
    """
    70/15/15 split where every value of `group_column` stays inside ONE split.
    The groups (not the rows) are split, stratified by language when possible.
    """
    groups = df[group_column].fillna("missing_" + df["benchmark_id"])
    df = df.assign(_group=groups)

    group_table = df.groupby("_group")["language"].agg(lambda s: s.mode()[0]).rename("language").reset_index()
    language_counts = group_table["language"].value_counts()
    stratify = group_table["language"] if language_counts.min() >= 7 else None
    if stratify is None:
        print("  (some languages have too few groups - splitting without stratification)")

    holdout_fraction = config.VAL_FRACTION + config.TEST_FRACTION
    train_groups, holdout_groups = train_test_split(
        group_table, test_size=holdout_fraction, stratify=stratify, random_state=seed)
    holdout_stratify = holdout_groups["language"] if stratify is not None else None
    val_groups, test_groups = train_test_split(
        holdout_groups, test_size=config.TEST_FRACTION / holdout_fraction,
        stratify=holdout_stratify, random_state=seed)

    def rows_of(group_df):
        return df[df["_group"].isin(set(group_df["_group"]))].drop(columns="_group")

    return rows_of(train_groups), rows_of(val_groups), rows_of(test_groups)


def count_overlap(splits, column):
    """How many values of `column` appear in more than one split (pairwise)."""
    names = list(splits)
    result = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = set(splits[names[i]][column].dropna()) & set(splits[names[j]][column].dropna())
            result[f"{names[i]} & {names[j]}"] = len(shared)
    return result


def identical_golden_overlap(splits):
    """Golden examples (prompt, suffix, golden) that appear in training AND another split."""
    def keys(d):
        return set(zip(d["prompt"].fillna(""), d["suffix"].fillna(""), d["golden_completion"]))
    train_keys = keys(splits["train"])
    return {name: len(keys(splits[name]) & train_keys) for name in ["validation", "test"]}


def create_splits(group_by="context_id", force=False):
    set_seed()
    ensure_output_dirs()
    print("=" * 70)
    print(f"TRAIN / VALIDATION / TEST SPLIT (grouped by {group_by})")
    print("=" * 70)
    split_paths = [config.PROCESSED_DIR / f"{name}.csv" for name in ["train", "validation", "test"]]
    if not force and all(p.exists() for p in split_paths):
        # Never silently replace an existing split: saved models were trained on it.
        print("Split files already exist - keeping them (use --force to recreate).")
        print("If you recreate the split, retrain every model with --force afterwards.")
        return None
    df = add_context_id(load_raw_data())
    print(f"{len(df)} benchmark items, {df['context_id'].nunique()} distinct code contexts, "
          f"{df['repo_url'].nunique()} repositories")

    train_df, val_df, test_df = grouped_split(df, group_by)
    splits = {"train": train_df, "validation": val_df, "test": test_df}

    # Safety checks -------------------------------------------------------
    if sum(len(s) for s in splits.values()) != len(df):
        raise RuntimeError("Rows were lost or duplicated during the split!")
    id_overlap = count_overlap(splits, "benchmark_id")
    context_overlap = count_overlap(splits, "context_id")
    golden_overlap = identical_golden_overlap(splits)
    repo_overlap = count_overlap(splits, "repo_url")
    if any(id_overlap.values()) or any(context_overlap.values()) or any(golden_overlap.values()):
        raise RuntimeError(f"Leakage detected! ids={id_overlap} contexts={context_overlap} golden={golden_overlap}")
    print("benchmark_id overlap      :", id_overlap, "(must be 0)")
    print("code-context overlap      :", context_overlap, "(must be 0)")
    print("identical golden examples :", golden_overlap, "(must be 0)")
    print("repository overlap        :", repo_overlap,
          "(allowed unless --group-by repo_url; see README limitations)")

    # Save -------------------------------------------------------------------
    for name, split_df in splits.items():
        path = config.PROCESSED_DIR / f"{name}.csv"
        split_df.to_csv(path, index=False)
        print(f"  {name:<10} {len(split_df):>5} benchmark items -> {2 * len(split_df):>5} binary examples   ({path})")

    summary = pd.concat({name: s["hallucination_type"].value_counts() for name, s in splits.items()},
                        axis=1).fillna(0).astype(int)
    percent = (100 * summary / summary.sum()).round(1).add_suffix(" %")
    summary = pd.concat([summary, percent], axis=1)
    summary.to_csv(config.METRICS_DIR / "split_category_counts.csv")
    language_summary = pd.concat({name: s["language"].value_counts() for name, s in splits.items()},
                                 axis=1).fillna(0).astype(int)
    language_summary.to_csv(config.METRICS_DIR / "split_language_counts.csv")
    print("\nCategory counts per split:\n" + summary.to_string())
    print("\nLanguage counts per split:\n" + language_summary.to_string())

    save_json({"group_by": group_by, "seed": config.SEED,
               "sizes_benchmark_items": {n: len(s) for n, s in splits.items()},
               "benchmark_id_overlap": id_overlap, "context_overlap": context_overlap,
               "identical_golden_overlap": golden_overlap, "repo_url_overlap": repo_overlap},
              config.METRICS_DIR / "split_info.json")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the train/validation/test split.")
    parser.add_argument("--group-by", default="context_id", choices=["context_id", "repo_url"],
                        help="context_id (default: same prompt+suffix stay together) or "
                             "repo_url (stricter: whole repositories stay together)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing split files (then retrain all models with --force)")
    args = parser.parse_args()
    create_splits(args.group_by, args.force)
