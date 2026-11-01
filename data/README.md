# Data

| File | What it is |
|---|---|
| `delulu.csv` | Unmodified copy of the original Delulu benchmark CSV (1,947 rows x 13 columns). SHA-256 `4738d0c9...6c5f37`, identical to `PROJECT CODES/delulu.csv`. |
| `processed/train.csv` | 70% of the original rows (created by `python -m src.data_split`) |
| `processed/validation.csv` | 15% of the original rows - used for model selection / early stopping |
| `processed/test.csv` | 15% of the original rows - used ONLY for the final evaluation |

The processed files keep the original **paired** format (one row = one golden + one
hallucinated completion) plus one extra analysis column, `context_id` (a hash of
prompt + suffix used to group the split). The expansion to binary examples
(golden = 0, hallucinated = 1) happens in memory in `src/preprocessing.py`.

No data augmentation, synthetic examples, or duplicated samples are created anywhere.

## Columns

| Column | Meaning | Used as model input? |
|---|---|---|
| `language` | Programming language (7 languages) | yes |
| `prompt` | Code before the gap | yes (end of it) |
| `suffix` | Code after the gap | yes (start of it) |
| `golden_completion` | Correct completion -> label 0 | yes (as "completion") |
| `hallucinated_completion` | Hallucinated completion -> label 1 | yes (as "completion") |
| `hallucination_type` | import / method / parameter / undefinedvariable | **no** (Task B label, analysis) |
| `error_message` | Compiler/runtime error of the hallucinated code | **no** (reveals the label) |
| `benchmark_id`, `image_tag`, `metadata` | Identifiers; contain the category name | **no** |
| `file_path`, `repo_url`, `license` | Source repository information | **no** (identical within a pair; could only enable memorising repositories) |

## Source and license

**Delulu: A Verified Multi-Lingual Benchmark for Code Hallucination Detection in
Fill-in-the-Middle Tasks** (Erfanian et al., 2026, Microsoft).
Hugging Face: `microsoft/delulu-fim-benchmark` · Code: https://github.com/microsoft/delulu ·
Paper: https://arxiv.org/abs/2605.07024. The `delulu.csv` here is byte-identical
(same SHA-256) to the file published on Hugging Face.

- The dataset card, schema and metadata fields are released under **MIT**.
- The code fields (`prompt`, `suffix`, `golden_completion`, `hallucinated_completion`)
  keep the license of their upstream GitHub repository, recorded per row in the
  `license` column (MIT 818, Apache-2.0 594, UNLICENSED 104, EPL-2.0 93, GPL-3.0 71, ...).
  Respect those licenses when redistributing snippets.

**Intended use (from the dataset card):** Delulu is published as a *test-only*
benchmark and the authors discourage fine-tuning on it, mainly so that it stays a clean
benchmark and is not used to train models that produce more convincing hallucinations.
This course project trains *detectors* on an internal 70/15/15 split, which is a
defensive use. The authors ask for such use to be reported transparently, so the
README says so explicitly. Do not present these numbers as official Delulu benchmark
results, and do not publish models trained on it as general-purpose detectors.
