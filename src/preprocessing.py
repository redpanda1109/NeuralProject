"""
Preprocessing shared by ALL models.

Main ideas
----------
1. Load the raw / split CSVs and validate their columns.
2. Turn each PAIRED row (golden + hallucinated) into TWO binary examples:
       golden_completion        -> label 0 (correct)
       hallucinated_completion  -> label 1 (hallucinated)
   Both examples keep the same `benchmark_id` (= `pair_id`) so we never forget
   that they come from the same benchmark item.
3. Build the model input from ONLY: language, prompt, suffix, completion.
   (`check_no_leakage` enforces this.)
4. Keep a window of code around the completion: the END of the prompt and the
   START of the suffix, because long files do not fit into a model.
5. A simple code tokenizer (for the RNN) and a token-budget function (shared by
   the RNN and the Transformers) that guarantees the completion is never cut off.
"""

import re

import pandas as pd

from src import config
from src.utils import require_file


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def validate_columns(df, required_columns=config.REQUIRED_COLUMNS):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing required columns: {missing}\n"
                         f"Columns found: {list(df.columns)}")


def load_raw_data(path=config.RAW_DATA_PATH):
    """Load the original delulu.csv (never modified)."""
    require_file(path, "Copy delulu.csv into NeuralProject/data/ (see INSTRUCTIONS.md).")
    df = pd.read_csv(path)
    validate_columns(df)
    return df


def load_split(split_name):
    """Load data/processed/{train,validation,test}.csv created by data_split.py."""
    path = config.PROCESSED_DIR / f"{split_name}.csv"
    require_file(path, "Run the split first:  python -m src.data_split")
    df = pd.read_csv(path)
    validate_columns(df)
    return df


def fill_text_columns(df):
    """Missing prompt/suffix (some files start/end at the completion) -> empty string."""
    df = df.copy()
    for column in ["prompt", "suffix", "golden_completion", "hallucinated_completion"]:
        df[column] = df[column].fillna("").astype(str)
    return df


# ---------------------------------------------------------------------------
# Task A: paired rows -> binary examples
# ---------------------------------------------------------------------------
def expand_to_binary(paired_df):
    """
    One paired row  ->  two binary examples (golden = 0, hallucinated = 1).

    Kept for ANALYSIS only (never model input): pair_id, hallucination_type.
    """
    paired_df = fill_text_columns(paired_df)
    records = []
    for row in paired_df.itertuples(index=False):
        shared = {
            "pair_id": row.benchmark_id,
            "language": row.language,
            "prompt": row.prompt,
            "suffix": row.suffix,
            "hallucination_type": row.hallucination_type,  # analysis only
        }
        records.append({**shared, "completion": row.golden_completion,
                        "completion_type": "golden", "label": 0})
        records.append({**shared, "completion": row.hallucinated_completion,
                        "completion_type": "hallucinated", "label": 1})
    return pd.DataFrame(records)


def hallucinated_only(paired_df):
    """
    Task B data: only the hallucinated completions, labelled with their category
    (import / method / parameter / undefinedvariable).
    """
    paired_df = fill_text_columns(paired_df)
    task_b = pd.DataFrame({
        "pair_id": paired_df["benchmark_id"],
        "language": paired_df["language"],
        "prompt": paired_df["prompt"],
        "suffix": paired_df["suffix"],
        "completion": paired_df["hallucinated_completion"],
        "category": paired_df["hallucination_type"],
    })
    task_b["label"] = task_b["category"].map({name: i for i, name in enumerate(config.CATEGORY_NAMES)})
    if task_b["label"].isna().any():
        unknown = task_b.loc[task_b["label"].isna(), "category"].unique()
        raise ValueError(f"Unknown hallucination categories: {unknown}")
    task_b["label"] = task_b["label"].astype(int)
    return task_b


def check_no_leakage(columns_used):
    """Stop immediately if a model is about to read a label-revealing column."""
    leaked = [c for c in columns_used if c in config.LEAKY_COLUMNS]
    not_allowed = [c for c in columns_used if c not in config.MODEL_INPUT_COLUMNS]
    if leaked or not_allowed:
        raise RuntimeError(
            f"Data leakage guard: model input may only use {config.MODEL_INPUT_COLUMNS}, "
            f"but got {list(columns_used)} (leaky: {leaked})."
        )


def load_binary_splits():
    """Convenience: train/validation/test as binary (Task A) DataFrames."""
    check_no_leakage(config.MODEL_INPUT_COLUMNS)
    return {name: expand_to_binary(load_split(name)) for name in ["train", "validation", "test"]}


# ---------------------------------------------------------------------------
# Context windows
# ---------------------------------------------------------------------------
def prompt_tail(text, n_chars=config.PROMPT_TAIL_CHARS):
    """Last n characters of the prompt (the code right BEFORE the completion)."""
    return text[-n_chars:] if n_chars > 0 else ""


def suffix_head(text, n_chars=config.SUFFIX_HEAD_CHARS):
    """First n characters of the suffix (the code right AFTER the completion)."""
    return text[:n_chars]


# ---------------------------------------------------------------------------
# Code tokenizer (used by the RNN)
# ---------------------------------------------------------------------------
# identifiers | numbers | any single symbol such as ( ) . , = { }
CODE_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")
# Splits camelCase / PascalCase / snake_case into sub-words.
SUBWORD_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_identifier(identifier):
    """'CacheMessageInfo' -> ['cache', 'message', 'info'];  'my_var2' -> ['my', 'var', '2']"""
    parts = SUBWORD_PATTERN.findall(identifier)
    return [p.lower() for p in parts] if parts else [identifier.lower()]


def tokenize_code(text):
    """
    Turn code into a list of tokens.

    Identifiers are split into lowercase sub-words so that a new name such as
    `CacheMessageInfo` is still made of known pieces instead of one <unk> token.
    """
    tokens = []
    for token in CODE_TOKEN_PATTERN.findall(text):
        if token[0].isalpha() or token[0] == "_":
            tokens.extend(split_identifier(token))
        else:
            tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Token budget (shared by RNN and Transformers)
# ---------------------------------------------------------------------------
def fit_to_budget(prompt_tokens, completion_tokens, suffix_tokens, budget,
                  completion_max, prompt_share=config.PROMPT_BUDGET_SHARE):
    """
    Choose how many tokens of each part to keep so that the total <= budget.

    1. The completion is kept first (up to `completion_max` tokens).
    2. The remaining budget is split between prompt (70%) and suffix (30%).
    3. If one side is shorter than its share, the other side gets the spare room.

    Returns (prompt_part, completion_part, suffix_part).
    """
    completion_part = completion_tokens[:completion_max]
    remaining = max(budget - len(completion_part), 0)

    prompt_budget = int(remaining * prompt_share)
    suffix_budget = remaining - prompt_budget
    spare_from_prompt = max(prompt_budget - len(prompt_tokens), 0)
    spare_from_suffix = max(suffix_budget - len(suffix_tokens), 0)
    prompt_budget += spare_from_suffix
    suffix_budget += spare_from_prompt

    # Keep the END of the prompt (closest to the completion) ...
    prompt_part = prompt_tokens[-prompt_budget:] if prompt_budget > 0 else []
    # ... and the START of the suffix.
    suffix_part = suffix_tokens[:suffix_budget]
    return list(prompt_part), list(completion_part), list(suffix_part)


# ---------------------------------------------------------------------------
# Text views for TF-IDF baselines
# ---------------------------------------------------------------------------
def build_tfidf_text(binary_df, view, context_prompt_chars, context_suffix_chars):
    """
    Build one string per example for TF-IDF.

    view = "completion"          -> "lang_python <completion>"
    view = "completion_context"  -> completion + short code window around it
    """
    check_no_leakage(["language", "prompt", "suffix", "completion"])
    language_token = "lang_" + binary_df["language"].astype(str)
    if view == "completion":
        return (language_token + " " + binary_df["completion"]).tolist()
    if view == "completion_context":
        before = binary_df["prompt"].apply(lambda t: prompt_tail(t, context_prompt_chars))
        after = binary_df["suffix"].apply(lambda t: suffix_head(t, context_suffix_chars))
        return (language_token + " " + binary_df["completion"] + " \n " + before + " \n " + after).tolist()
    raise ValueError(f"Unknown TF-IDF view: {view}")
