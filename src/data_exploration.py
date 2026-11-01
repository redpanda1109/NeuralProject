"""
PHASE 1 - Understand the dataset.

Run from the project root:
    python -m src.data_exploration

Produces:
    outputs/metrics/data_exploration_report.md   (readable report)
    outputs/metrics/eda_*.csv                    (analysis tables)
    outputs/figures/eda_*.png                    (graphs)
"""

import json

import pandas as pd

from src import config
from src.preprocessing import load_raw_data
from src.utils import ensure_output_dirs, markdown_table
from src import visualization as viz


def _shorten(text, n=300):
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= n else "..." + text[-n:]


def basic_overview(df):
    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (100 * df.isna().mean()).round(2),
        "unique_values": df.nunique(),
    })
    overview.index.name = "column"
    return overview


def length_statistics(df):
    """Character and line counts for the code text columns."""
    rows = {}
    for column in ["prompt", "suffix", "golden_completion", "hallucinated_completion"]:
        text = df[column].fillna("").astype(str)
        chars = text.str.len().describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        lines = (text.str.count("\n") + 1).describe(percentiles=[0.5, 0.95])
        rows[column + " (chars)"] = chars
        rows[column + " (lines)"] = lines
    return pd.DataFrame(rows).T.round(1)


def find_suspicious_records(df):
    """Look for malformed or unexpected records. Returns a DataFrame of issues."""
    issues = []

    def add(mask, issue):
        for benchmark_id in df.loc[mask, "benchmark_id"]:
            issues.append({"benchmark_id": benchmark_id, "issue": issue})

    golden = df["golden_completion"].fillna("").astype(str)
    hallucinated = df["hallucinated_completion"].fillna("").astype(str)

    add(df["prompt"].isna(), "missing prompt (completion is at the start of the file)")
    add(df["suffix"].isna(), "missing suffix (completion is at the end of the file)")
    add(df["prompt"].fillna("").str.len() < 20, "very short prompt (< 20 chars)")
    add(golden.str.strip() == hallucinated.str.strip(), "golden and hallucinated completion identical")
    add(golden.str.strip() == "", "empty golden completion")
    add(hallucinated.str.strip() == "", "empty hallucinated completion")
    add(df["repo_url"].isna(), "missing repo_url")

    # benchmark_id should look like "<language>-<type>-<hash>"
    expected_prefix = df["language"] + "-" + df["hallucination_type"] + "-"
    matches = [str(bid).startswith(prefix) for bid, prefix in zip(df["benchmark_id"], expected_prefix)]
    add(~pd.Series(matches, index=df.index), "benchmark_id does not match language/hallucination_type")

    # metadata is a JSON string; it should agree with the columns
    for i, row in df.iterrows():
        try:
            meta = json.loads(row["metadata"])
        except (TypeError, ValueError):
            issues.append({"benchmark_id": row["benchmark_id"], "issue": "metadata is not valid JSON"})
            continue
        for key in ["language", "hallucination_type", "benchmark_id", "file_path"]:
            if key in meta and str(meta[key]) != str(row[key]):
                issues.append({"benchmark_id": row["benchmark_id"], "issue": f"metadata.{key} disagrees with column"})

    return pd.DataFrame(issues, columns=["benchmark_id", "issue"])


def run_exploration():
    ensure_output_dirs()
    print("=" * 70)
    print("PHASE 1 - DATASET EXPLORATION")
    print("=" * 70)
    df = load_raw_data()
    print(f"Loaded {config.RAW_DATA_PATH}: {df.shape[0]} rows x {df.shape[1]} columns")

    # --- 1. Overview -------------------------------------------------------
    overview = basic_overview(df)
    overview.to_csv(config.METRICS_DIR / "eda_column_overview.csv")
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["benchmark_id"].duplicated().sum())
    duplicate_pairs = int(df.duplicated(subset=["prompt", "suffix", "golden_completion",
                                                "hallucinated_completion"]).sum())
    duplicate_golden_in_context = int(df.duplicated(subset=["prompt", "suffix", "golden_completion"]).sum())
    n_contexts = df[["prompt", "suffix"]].fillna("").drop_duplicates().shape[0]

    # --- 2. Labels and languages -------------------------------------------
    category_counts = df["hallucination_type"].value_counts()
    category_table = pd.DataFrame({"count": category_counts,
                                   "percent": (100 * category_counts / len(df)).round(2)})
    category_table.to_csv(config.METRICS_DIR / "eda_category_distribution.csv")

    language_counts = df["language"].value_counts()
    language_table = pd.DataFrame({"count": language_counts,
                                   "percent": (100 * language_counts / len(df)).round(2)})
    language_table.to_csv(config.METRICS_DIR / "eda_language_distribution.csv")

    crosstab = pd.crosstab(df["language"], df["hallucination_type"])
    crosstab.to_csv(config.METRICS_DIR / "eda_language_by_category.csv")

    # --- 3. Lengths --------------------------------------------------------
    lengths = length_statistics(df)
    lengths.to_csv(config.METRICS_DIR / "eda_length_statistics.csv")
    golden_len = df["golden_completion"].astype(str).str.len()
    halluc_len = df["hallucinated_completion"].astype(str).str.len()
    longer_share = float((halluc_len > golden_len).mean())

    # --- 4. Repositories / file paths (leakage analysis) --------------------
    repo_counts = df["repo_url"].fillna("(missing)").value_counts()
    repo_counts.rename("benchmark_items").to_csv(config.METRICS_DIR / "eda_repo_counts.csv")

    # --- 5. Suspicious records -------------------------------------------------
    suspicious = find_suspicious_records(df)
    suspicious.to_csv(config.METRICS_DIR / "eda_suspicious_records.csv", index=False)
    issue_summary = suspicious["issue"].value_counts().rename("records")

    # --- 6. Figures --------------------------------------------------------
    print("Creating figures ...")
    viz.plot_counts(category_counts, "Benchmark items per hallucination category",
                    "Number of benchmark items", config.FIGURES_DIR / "eda_hallucination_categories.png")
    viz.plot_counts(language_counts, "Benchmark items per programming language",
                    "Number of benchmark items", config.FIGURES_DIR / "eda_languages.png")
    viz.plot_crosstab(crosstab, "Hallucination categories within each language",
                      config.FIGURES_DIR / "eda_language_by_category.png")
    viz.plot_histograms({"Golden (correct)": golden_len, "Hallucinated": halluc_len},
                        "Completion length", "Characters",
                        config.FIGURES_DIR / "eda_completion_lengths.png")
    viz.plot_histograms({"Prompt (code before)": df["prompt"].fillna("").str.len().clip(lower=1),
                         "Suffix (code after)": df["suffix"].fillna("").str.len().clip(lower=1)},
                        "Prompt and suffix length", "Characters",
                        config.FIGURES_DIR / "eda_prompt_suffix_lengths.png")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        viz.plot_counts(missing, "Missing values per column", "Number of missing values",
                        config.FIGURES_DIR / "eda_missing_values.png", show_percent=False)

    # --- 7. Written report -------------------------------------------------
    examples = []
    for category in sorted(df["hallucination_type"].unique()):
        sample = df[df["hallucination_type"] == category].sample(2, random_state=config.SEED)
        examples.append(f"### `{category}`\n")
        for _, row in sample.iterrows():
            examples.append(
                f"**{row['benchmark_id']}** ({row['language']})\n\n"
                f"```\n# ...end of prompt...\n{_shorten(row['prompt'], 200)}\n```\n"
                f"- golden: `{_shorten(row['golden_completion'], 150).strip()}`\n"
                f"- hallucinated: `{_shorten(row['hallucinated_completion'], 150).strip()}`\n"
                f"- error message (analysis only, never model input): "
                f"`{_shorten(row['error_message'], 150).strip()}`\n"
            )

    report = f"""# Delulu Dataset Exploration Report

Source file: `data/delulu.csv` (copy of the original; the original was not modified).

## 1. Size
- Rows (benchmark items): **{len(df)}**
- Columns: **{df.shape[1]}**
- Each row is a **pair**: one correct (`golden_completion`) and one hallucinated
  (`hallucinated_completion`) completion for the same code context
  (`prompt` = code before the gap, `suffix` = code after the gap).
- After expanding pairs for the binary task: **{2 * len(df)}** examples (50% correct / 50% hallucinated).

## 2. Columns, data types and missing values
{markdown_table(overview)}

## 3. Duplicates
- Fully duplicated rows: **{duplicate_rows}**
- Duplicated `benchmark_id`: **{duplicate_ids}** (every benchmark_id is unique, so splitting by
  benchmark_id is the same as splitting by original row)
- Duplicated (prompt, suffix, golden, hallucinated): **{duplicate_pairs}**
- Duplicated (prompt, suffix, golden): **{duplicate_golden_in_context}**
- Distinct code contexts (prompt + suffix): **{n_contexts}**

**Important:** the benchmark often builds 2-3 items (different hallucination types) from the
SAME location in the same file. Those items share prompt, suffix and golden completion, so their
correct examples are identical. A plain row split would place identical correct examples in
both training and test. `data_split.py` therefore keeps each code context inside one split.

## 4. Programming languages ({df['language'].nunique()})
{markdown_table(language_table)}

## 5. Hallucination categories (Task B labels)
{markdown_table(category_table)}

### Language x category
{markdown_table(crosstab)}

## 6. Length statistics
{markdown_table(lengths)}

- Hallucinated completion is longer than the golden one in **{100 * longer_share:.1f}%** of pairs
  (a length bias that simple models could exploit).

## 7. Repositories and file paths (leakage analysis)
- Distinct repositories: **{df['repo_url'].nunique()}** (missing repo_url: {int(df['repo_url'].isna().sum())})
- Items from the largest repository: **{int(repo_counts.iloc[0])}**
- Distinct file paths: **{df['file_path'].nunique()}**
- Inside a pair, `prompt`, `suffix`, `file_path`, `repo_url` and `license` are **identical** for the
  correct and the hallucinated completion. Therefore `file_path` carries **no information** that can
  separate the two completions; it could only let a model memorise repositories. It is **not** used
  as model input. `hallucination_type`, `error_message`, `benchmark_id`, `image_tag` and `metadata`
  directly reveal the label/category and are also excluded.

## 8. Suspicious / unexpected records
{markdown_table(issue_summary.to_frame()) if len(issue_summary) else "No problems found."}

Full list: `outputs/metrics/eda_suspicious_records.csv`.
Missing prompts/suffixes are legitimate (the completion is at the very start or end of a file); they
are replaced by an empty string during preprocessing. No records were removed.

## 9. Example records per category
{chr(10).join(examples)}

## 10. Figures
- `outputs/figures/eda_hallucination_categories.png`
- `outputs/figures/eda_languages.png`
- `outputs/figures/eda_language_by_category.png`
- `outputs/figures/eda_completion_lengths.png`
- `outputs/figures/eda_prompt_suffix_lengths.png`
- `outputs/figures/eda_missing_values.png`
"""
    report_path = config.METRICS_DIR / "data_exploration_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\nSummary")
    print(f"  rows={len(df)}, languages={df['language'].nunique()}, repos={df['repo_url'].nunique()}")
    print(f"  categories:\n{category_table.to_string()}")
    print(f"  suspicious-record issues:\n{issue_summary.to_string() if len(issue_summary) else '  none'}")
    print(f"\nReport saved -> {report_path}")
    return report_path


if __name__ == "__main__":
    run_exploration()
