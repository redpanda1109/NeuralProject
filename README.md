# NeuralProject — Detecting Hallucinations in LLM-Generated Code

A Neural Networks & Deep Learning project that compares traditional machine learning,
recurrent neural networks and pretrained code Transformers on detecting **hallucinated
code completions**: completions that import modules that do not exist, call methods
or functions that do not exist, pass wrong parameters, or use undefined variables.

> Quick start: see **[INSTRUCTIONS.md](INSTRUCTIONS.md)** (Colab notebook:
> `notebooks/NeuralProject_Colab.ipynb`).

---

## 1. Objective and research question

**Objective.** Given the code around a gap (*prompt* before, *suffix* after) and a
completion an LLM proposed for that gap, decide whether the completion is
**correct (0)** or **hallucinated (1)**.

**Research question.** *How well can models of increasing capacity detect code
hallucinations, from TF-IDF + linear models to GRU/LSTM networks trained from
scratch to pretrained code Transformers (CodeBERT, GraphCodeBERT)? And does adapting
the final two Transformer layers help compared with a completely frozen encoder?*

---

## 2. Dataset — Delulu

`data/delulu.csv` is an unmodified copy of the **Delulu** benchmark (Erfanian et al., 2026, Microsoft;
`microsoft/delulu-fim-benchmark`; https://arxiv.org/abs/2605.07024). It has **1,947 rows × 13 columns**,
7 languages (TypeScript 420, Python 370, Go 291, Rust 252, C# 246, Java 243, C++ 125) and 319 repositories.
Every row is a **pair**: a correct (`golden_completion`) and a hallucinated
(`hallucinated_completion`) completion for the same gap. Both are execution-verified:
the golden one compiles/runs and the hallucinated one fails.

| Hallucination type | Items | Share | Example |
|---|---|---|---|
| `undefinedvariable` | 576 | 29.6% | uses a variable/name that is not defined |
| `import` | 476 | 24.4% | imports a module/package that does not exist |
| `method` | 460 | 23.6% | calls a method/function that does not exist on a correct module/object |
| `parameter` | 435 | 22.3% | passes wrong or non-existent arguments |

The full exploration report is in `outputs/metrics/data_exploration_report.md`. Things worth knowing:

- 125 rows have no `prompt` (the gap is at the start of the file), 12 have no `suffix`, and 3 have no `repo_url`. These are legitimate; empty text is used. No rows were removed.
- The hallucinated completion is longer than the golden one in **88%** of pairs, a shortcut simple models can exploit.
- Only **1,077 distinct code contexts** exist for 1,947 items. The benchmark often builds 2–3 items (different hallucination types) from the **same place in the same file**, so those items share an identical golden example (see §5).

Column descriptions and the license are in [`data/README.md`](data/README.md).

### Label definitions

**Task A — binary detection (main task).** Each pair is expanded into two examples:

| Example | Label |
|---|---|
| `golden_completion` + its context | **0 = correct** |
| `hallucinated_completion` + the same context | **1 = hallucinated** (the **positive class** for every metric) |

This gives 3,894 examples (exactly 50/50). Both examples keep the original `benchmark_id`
(`pair_id`), so the pairing is never lost: they are always in the same split, and we also report a
paired metric (§7).

**Task B — hallucination category (optional, 4 classes).** Only hallucinated completions are used
(1,947 examples); the label is `hallucination_type`. This is a **different task**: it asks *which kind*
of hallucination it is, not *whether* the code is hallucinated. It is run with the TF-IDF baselines.

---

## 3. Input representation

Each model only sees **`language`, `prompt`, `suffix` and the completion** being classified. This is
enforced in code (`preprocessing.check_no_leakage`).

Prompts can be up to 22,650 characters, far more than a model can read, so we keep a
**window around the gap**:

```text
<language>  [END of the prompt]  [COMPLETION]  [START of the suffix]
```

- Token budget: CodeBERT/GraphCodeBERT 512 tokens, RNN 400 tokens.
- The completion is **always kept** (up to 160 / 128 tokens). The remaining budget is split 70% prompt / 30% suffix, and unused budget on one side goes to the other (`preprocessing.fit_to_budget`).
- Neural models mark the completion tokens with a mask and **pool only over the completion**. The model reads the whole window, and then the hidden states of the completion tokens, which already "know" their context, are averaged and max-pooled.

**Excluded columns and why:**

| Column | Why it is not input |
|---|---|
| `hallucination_type`, `error_message` | directly reveal the label / category |
| `benchmark_id`, `image_tag`, `metadata` | contain the category name (e.g. `go-method-…`) |
| `file_path`, `repo_url`, `license` | **identical for both completions of a pair**, so they cannot separate correct from hallucinated; they would only let a model memorise repositories |

---

## 4. Project structure

```text
NeuralProject/
├── data/
│   ├── delulu.csv              original dataset (unmodified copy)
│   ├── processed/              train / validation / test CSVs
│   └── README.md               columns, source, license
├── notebooks/
│   └── NeuralProject_Colab.ipynb   run everything on Colab (T4)
├── src/
│   ├── config.py               ALL paths and hyperparameters (one place to change things)
│   ├── utils.py                seeds, device, folders, logging, JSON
│   ├── preprocessing.py        loading, pair -> binary expansion, leakage guard, tokenizer, token budget
│   ├── data_exploration.py     PHASE 1
│   ├── data_split.py           70/15/15 grouped split
│   ├── baseline_models.py      PHASE 2  TF-IDF + Logistic Regression / Linear SVM (+ Task B)
│   ├── training.py             shared PyTorch training loop, early stopping, completion pooling
│   ├── rnn_models.py           PHASE 3  GRU / LSTM
│   ├── pretrained_models.py    PHASE 4  frozen CodeBERT / GraphCodeBERT + head
│   ├── finetuning.py           PHASE 5  unfreeze ONLY the final 2 layers
│   ├── evaluation.py           PHASE 6  metrics, comparison table, answers
│   └── visualization.py        all plots
├── tests/smoke_test.py         tiny end-to-end check (temporary output folder)
├── outputs/  figures/ metrics/ models/ logs/ features/
├── README.md  INSTRUCTIONS.md  requirements.txt  .gitignore
```

Two files were added to the suggested structure for a technical reason: `config.py` keeps every
setting in one place, and `training.py` holds the single training loop that the RNNs, the frozen
head and the fine-tuned model all share, so it is not copied three times.

---

## 5. Data split and leakage prevention

- **70 / 15 / 15** split of the original rows, giving 1,355 / 296 / 296 benchmark items (2,710 / 592 / 592 binary examples). Seed 42.
- The split is done **before** expanding pairs, so both completions of a `benchmark_id` are always in the same split.
- **Grouped by code context.** `benchmark_id` is unique per row, but items built from the same gap share an identical prompt, suffix and golden completion. A plain row split placed **163 of 293** test items' golden examples into training as well, so the model would be tested on examples it had memorised. Every distinct (prompt, suffix) therefore gets a `context_id`, and each context stays in one split. This is strictly stronger than splitting by `benchmark_id`. `data_split.py` checks that the overlap of benchmark_ids, contexts and identical golden examples is **0** and stops otherwise.
- **Stratified** by language at the group level. Category shares stay within ±2.5 percentage points across splits (`outputs/metrics/split_category_counts.csv`).
- TF-IDF vectorizers and the RNN vocabulary are **fit on training data only**.
- The **validation** split is used for every choice: hyperparameters, early stopping and checkpoint selection. The **test** split is predicted once, after selection, and never influences training.
- Repositories can still appear in several splits (different files or locations). `python -m src.data_split --group-by repo_url` creates a stricter repository-level split (see Limitations).

---

## 6. Models

| Phase | Model | What is trained | Key settings |
|---|---|---|---|
| 2 | **Logistic Regression** | all weights | TF-IDF; grid over text view × {word, char, word+char} × C; selected on validation macro-F1 |
| 2 | **Linear SVM** | all weights | same grid; the SVM decision value is used as the score for ROC/PR-AUC |
| 3 | **GRU** | all weights (from scratch) | Embedding 128 → BiGRU 128 → completion mean+max pooling → Dropout 0.3 → Linear → logit |
| 3 | **LSTM** | all weights (from scratch) | same, with BiLSTM |
| 4 | **CodeBERT (frozen)** | head only (~0.4M params) | encoder completely frozen; features cached once; 2-layer MLP head |
| 4 | **GraphCodeBERT (frozen)** | head only | same recipe |
| 5 | **CodeBERT (last 2 layers)** | layers 10–11 + head (~14.6M of 124M params) | AdamW, LR 2e-5 (encoder) / 1e-4 (head), linear warmup, fp16, batch 16 × 2 accumulation |
| 5 | GraphCodeBERT (last 2 layers) | layers 10–11 + head | optional, same recipe |

**RNN details.** Code is tokenized into identifiers, numbers and symbols. Identifiers are split
into lowercase sub-words (`CacheMessageInfo` → `cache message info`), so a hallucinated name made
of known pieces is not just one `<unk>`. The vocabulary is built from training data (min frequency 2,
max 30k), with `<pad>` and `<unk>` tokens. The BiGRU/BiLSTM uses packed sequences so padding is ignored.
Training uses BCEWithLogitsLoss, Adam (LR 1e-3), gradient clipping, and early stopping on validation
loss (patience 4).

**Frozen feature extraction (Phase 4).** The pretrained encoder works as a fixed "code
understanding" function. Every example passes through it **once** (no gradients, fp16 on GPU), and
the pooled completion features are cached in `outputs/features/`. Only a small MLP head
(1536 → 256 → 1) is trained on them. This measures how much hallucination-relevant information
CodeBERT *already* encodes without any adaptation.

**GraphCodeBERT note.** GraphCodeBERT was pretrained with an extra data-flow-graph input that needs
language-specific parsers for all 7 languages. We use it as a **sequence encoder** with the same
input as CodeBERT. This is supported by Hugging Face but does not use the data-flow part, which
the results should keep in mind.

### Fine-tuning: why unfreeze only the final two layers?

A Transformer's lower layers learn general patterns, such as syntax and which tokens belong
together. Its upper layers combine these into task-specific meaning. Unfreezing **only layers 10
and 11** (plus the head) lets the top of the network specialise in *"does this completion fit its
context?"* while the general code knowledge in layers 0–9 and the embeddings stays exactly as
pretrained. Compared with full fine-tuning, this:

- trains ~14.6M instead of ~124M parameters, which is faster and uses less GPU memory (gradients and optimizer state only for 2 layers);
- lowers the risk of **overfitting** and of "forgetting" pretrained knowledge on a small dataset (2.7k training examples);
- isolates one question: is adapting the *top* of the network enough to beat a frozen encoder?

Phase 5 builds directly on Phase 4. It uses the same tokenizer, input, pooling, head architecture
and split, and the head starts from the Phase 4 head. Before training,
`finetuning.verify_trainable_layers` prints the trainable/frozen parameter counts and the
trainable blocks. It **refuses to train** if anything other than `encoder.layer.10`,
`encoder.layer.11` and the head is trainable, or if any of those is frozen. The smoke test checks
this guard by deliberately unfreezing a wrong layer. Full fine-tuning is intentionally not
implemented.

---

## 7. Evaluation

All models are evaluated on the **same untouched test split** (592 examples) with the **same
positive class: 1 = hallucinated**, threshold 0.5 (SVM: decision value 0).

- **Accuracy, Precision, Recall, F1** of the hallucinated class; **ROC-AUC** and **PR-AUC** (from scores/probabilities).
- **Confusion matrix** and **classification report** per model.
- **Pairwise accuracy** (extra, uses the pairing): the share of benchmark items where the model scores the hallucinated completion higher than the golden one. 0.5 = chance.
- **Recall per hallucination category**, showing which kinds of hallucination are caught.
- **95% bootstrap confidence interval for F1**, resampling whole pairs. With ~300 test items, small differences may be noise, and the comparison text says so.

`python -m src.evaluation` reads only the saved prediction files. It writes
`outputs/metrics/model_comparison.csv` and `outputs/metrics/comparison_summary.md`, which answers the
six comparison questions automatically, and draws every comparison chart. A model without
predictions is listed as **unavailable**, never filled with made-up values. A "Yes" verdict requires
F1, accuracy **and** ROC-AUC to all improve, because F1 alone can be inflated by predicting
everything as hallucinated (F1 = 0.67 on balanced data).

---

## 8. Results

<!-- RESULTS-START -->
_Generated automatically by `python -m src.evaluation` from the saved test predictions - do not edit by hand. Positive class = 1 = hallucinated; test split = 592 examples. Models marked unavailable have not been trained yet._

| model | status | accuracy | precision | recall | f1 | roc_auc | pr_auc | pairwise_accuracy | f1_ci_low | f1_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | completed | 0.7213 | 0.6825 | 0.8277 | 0.7481 | 0.7981 | 0.7711 | 0.9155 | 0.7235 | 0.7716 |
| Linear SVM | completed | 0.7196 | 0.6816 | 0.8243 | 0.7462 | 0.8007 | 0.7683 | 0.9223 | 0.7211 | 0.7714 |
| GRU | completed | 0.6486 | 0.6803 | 0.5608 | 0.6148 | 0.6912 | 0.6337 | 0.8226 | 0.5758 | 0.6535 |
| LSTM | completed | 0.6588 | 0.6958 | 0.5642 | 0.6231 | 0.6998 | 0.6393 | 0.8395 | 0.5819 | 0.6617 |
| CodeBERT (frozen) | completed | 0.7365 | 0.7201 | 0.7736 | 0.7459 | 0.8184 | 0.8017 | 0.9139 | 0.7150 | 0.7747 |
| CodeBERT (last 2 layers) | completed | 0.7449 | 0.7636 | 0.7095 | 0.7356 | 0.8259 | 0.8065 | 0.9003 | 0.7011 | 0.7690 |
| GraphCodeBERT (frozen) | completed | 0.7399 | 0.7432 | 0.7331 | 0.7381 | 0.8180 | 0.7892 | 0.9206 | 0.7049 | 0.7685 |
| GraphCodeBERT (last 2 layers) | completed | 0.7466 | 0.7786 | 0.6892 | 0.7312 | 0.8359 | 0.8045 | 0.9307 | 0.6920 | 0.7641 |

**Task B (4-class category, hallucinated completions only, test split):**

| classifier | accuracy | macro_f1 | majority_class_share |
|---|---|---|---|
| logistic_regression | 0.8041 | 0.7977 | 0.2838 |
| linear_svm | 0.8007 | 0.7937 | 0.2838 |

Positive class = 1 = hallucinated. All models are evaluated on the same untouched test split.
Each question compares the model with the best test F1 in each group. The verdict is
'Yes' only if F1, accuracy AND ROC-AUC all improve, 'No' if none improve, otherwise 'Mixed'.
1. **Did the RNN outperform the traditional baseline?** No. Best RNN: LSTM (F1 0.6231, accuracy 0.6588, ROC-AUC 0.6998) vs best baseline: Logistic Regression (F1 0.7481, accuracy 0.7213, ROC-AUC 0.7981). Differences: F1 -0.1250, accuracy -0.0625, ROC-AUC -0.0984. The 95% bootstrap F1 intervals do not overlap.
2. **Did the pretrained model outperform the RNN?** Yes. Best pretrained model: CodeBERT (frozen) (F1 0.7459, accuracy 0.7365, ROC-AUC 0.8184) vs best RNN: LSTM (F1 0.6231, accuracy 0.6588, ROC-AUC 0.6998). Differences: F1 +0.1228, accuracy +0.0777, ROC-AUC +0.1186. The 95% bootstrap F1 intervals do not overlap.
3. **Did unfreezing the final two layers improve over the frozen encoder?**
   - codebert: Mixed (the metrics disagree). Best last-2-layers codebert: CodeBERT (last 2 layers) (F1 0.7356, accuracy 0.7449, ROC-AUC 0.8259) vs best frozen codebert: CodeBERT (frozen) (F1 0.7459, accuracy 0.7365, ROC-AUC 0.8184). Differences: F1 -0.0104, accuracy +0.0084, ROC-AUC +0.0075. The 95% bootstrap F1 intervals overlap, so the difference may not be reliable.
   - graphcodebert: Mixed (the metrics disagree). Best last-2-layers graphcodebert: GraphCodeBERT (last 2 layers) (F1 0.7312, accuracy 0.7466, ROC-AUC 0.8359) vs best frozen graphcodebert: GraphCodeBERT (frozen) (F1 0.7381, accuracy 0.7399, ROC-AUC 0.8180). Differences: F1 -0.0069, accuracy +0.0068, ROC-AUC +0.0180. The 95% bootstrap F1 intervals overlap, so the difference may not be reliable.
4. **Best accuracy:** **GraphCodeBERT (last 2 layers)** (accuracy = 0.7466)
5. **Best recall for hallucinated code:** **Logistic Regression** (recall = 0.8277)
6. **Best F1-score:** **Logistic Regression** (f1 = 0.7481)

Figures: `outputs/figures/comparison_*.png`, `frozen_vs_finetuned_*.png`, `confusion_matrix_*.png`, `loss_curve_*.png`.
<!-- RESULTS-END -->

---

## 9. Limitations

- **The dataset is meant to be test-only.** The Delulu authors publish it as an evaluation benchmark and discourage fine-tuning on it (mainly to keep it clean and to avoid training models that produce better hallucinations). This project trains *detectors* on an internal split, a defensive use the card asks to be reported transparently. These numbers are **not** official Delulu benchmark results, and the trained models should not be published as general detectors.
- **Small data.** 1,355 training items; ~300 test items, so differences of a few points are within noise (see the confidence intervals).
- **Repository overlap.** The default split separates code contexts, but different files or locations from the same repository can be in train and test. Use `--group-by repo_url` for a stricter estimate.
- **Length shortcut.** Hallucinated completions are usually longer, so part of what any model learns may be length, not meaning.
- **Context window.** Models see at most ~512 tokens around the gap. A hallucination that is only detectable from code far away (e.g. an import at the top of a long file) can be missed.
- **GraphCodeBERT** is used without its data-flow-graph input.
- **Threshold fixed at 0.5**, which favours neither precision nor recall. A different operating point could be chosen on validation data.
- Some language × category cells are small (Python × parameter: 22 items).

## 10. Future improvements

- Retrieval of the *whole* file or repository symbols (e.g. "does this identifier exist in the file?") as an extra input or feature.
- Use GraphCodeBERT's data-flow graphs (tree-sitter parsers per language).
- Code-specific encoders with longer context (e.g. UniXcoder, CodeT5+ encoder).
- Tune the decision threshold on validation; calibrate probabilities.
- Repository-level split and cross-validation for tighter confidence intervals.
- Train on separately generated data (as the Delulu authors recommend) and keep Delulu purely as the test set.

## 11. Dataset source and license

Erfanian, M., Troncoso, N., Garg, A., Gale, A., Liu, X., Golnari, P. A., & Fu, S. (2026).
*Delulu: A Verified Multi-Lingual Benchmark for Code Hallucination Detection in Fill-in-the-Middle Tasks.*
https://arxiv.org/abs/2605.07024 · https://github.com/microsoft/delulu ·
Hugging Face `microsoft/delulu-fim-benchmark`.
The dataset card, schema and metadata are MIT-licensed. Each code snippet keeps its upstream
repository's license, recorded per row in the `license` column.

Pretrained models: `microsoft/codebert-base`, `microsoft/graphcodebert-base` (Hugging Face).
