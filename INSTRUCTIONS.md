# How to run NeuralProject — step by step

> **Working directory:** every command below must be run from the **project root**,
> the folder that contains `src/`, `data/` and `README.md`:
>
> - Windows: `C:\Laavya\AMRITA CLG\SEMESTER 5\DL & NN\NeuralProject`
> - Colab: e.g. `/content/drive/MyDrive/NeuralProject` (the notebook sets this for you)
>
> Commands use `python -m src.<module>` so that Python can find the `src` package.
> Running `python src/<module>.py` directly will fail with `ModuleNotFoundError: No module named 'src'`.

---

## A. Recommended: run everything in Google Colab (T4 GPU)

1. **Upload** the whole `NeuralProject` folder to Google Drive, ideally to `MyDrive/NeuralProject`.
   It must contain `data/delulu.csv`.
2. Open `notebooks/NeuralProject_Colab.ipynb` from Drive with Google Colab.
3. `Runtime -> Change runtime type -> T4 GPU -> Save`.
4. Run the cells **in order, top to bottom**:

| Cell | What it does | Approx. time on T4 |
|---|---|---|
| 1 | Mount Google Drive | — |
| 2 | Find / set the project root (edit `PROJECT_ROOT` if needed) | — |
| 3 | `pip install -r requirements.txt` (keeps Colab's PyTorch) | 1 min |
| 4 | Verify the GPU | — |
| 5 | Quick look at the dataset | — |
| 6 | Phase 1 – dataset exploration + figures | < 1 min |
| 7 | Train / validation / test split | < 1 min |
| 8 | Phase 2 – TF-IDF Logistic Regression + Linear SVM (+ Task B) | 3–5 min (CPU) |
| 9 | Phase 3 – GRU and LSTM | 3–8 min |
| 10 | Phase 4 – frozen CodeBERT | 2–4 min |
| 11 | Phase 4 – frozen GraphCodeBERT | 2–4 min |
| 12 | Phase 5 – CodeBERT with the final 2 layers unfrozen | 10–25 min |
| 12b | (optional) same for GraphCodeBERT | 10–25 min |
| 13 | Phase 6 – evaluation and comparison of all models | < 1 min |
| 14 | Show comparison graphs + answers | — |
| 15 | Download `outputs.zip` (only needed if the project is NOT on Drive) | — |

**If Colab disconnects:** run cells 1–4 again, then continue where you stopped.
Finished steps are detected from their saved files and **skipped automatically**.
To retrain one step on purpose, add `--force`, for example `!python -u -m src.rnn_models --force`.

---

## B. Local setup (Windows, CPU is fine for phases 1–3)

```powershell
cd "C:\Laavya\AMRITA CLG\SEMESTER 5\DL & NN\NeuralProject"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(Linux/macOS: `source .venv/bin/activate` instead of `.venv\Scripts\activate`.)

Phases 4–5 also run on a CPU but are slow. Fine-tuning CodeBERT on a CPU takes hours, so use Colab for those.

---

## C. The individual commands, in order

| Step | Command | Main outputs |
|---|---|---|
| 1. Dataset exploration | `python -m src.data_exploration` | `outputs/metrics/data_exploration_report.md`, `outputs/metrics/eda_*.csv`, `outputs/figures/eda_*.png` |
| 2. Split | `python -m src.data_split` | `data/processed/{train,validation,test}.csv`, `outputs/metrics/split_*.csv` |
| 3. Baselines | `python -m src.baseline_models` | `outputs/models/{logistic_regression,linear_svm}/`, `taskB_*` files |
| 4. GRU + LSTM | `python -m src.rnn_models` | `outputs/models/{gru,lstm}/`, `loss_curve_{gru,lstm}.png` |
| 5. Frozen CodeBERT | `python -m src.pretrained_models --model codebert` | `outputs/models/codebert_frozen/`, `outputs/features/codebert_*.pt` |
| 6. Frozen GraphCodeBERT | `python -m src.pretrained_models --model graphcodebert` | `outputs/models/graphcodebert_frozen/` |
| 7. Fine-tune last 2 CodeBERT layers | `python -m src.finetuning --model codebert` | `outputs/models/codebert_finetuned_last2/` |
| 7b. (optional) same for GraphCodeBERT | `python -m src.finetuning --model graphcodebert` | `outputs/models/graphcodebert_finetuned_last2/` |
| 8. Evaluate + all graphs | `python -m src.evaluation` | `outputs/metrics/model_comparison.csv`, `outputs/metrics/comparison_summary.md`, `outputs/figures/comparison_*.png` |

Useful options:

- `--force` for steps 3–7 retrains even if results already exist.
- `python -m src.rnn_models --model gru` (or `lstm`) trains only one RNN.
- `python -m src.baseline_models --task a` runs only the binary task; `--task b` runs only the 4-class task.
- `python -m src.pretrained_models --model both` runs CodeBERT and GraphCodeBERT.
- `python -m src.pretrained_models --save-encoder` also saves a full copy of the unchanged encoder (~500 MB).
- `python -m src.data_split` **keeps an existing split**. The split in `data/processed/` is the one the saved results were trained on. To create a new one, add `--force`.
- `python -m src.data_split --group-by repo_url --force` creates a stricter split where whole repositories stay together. Re-run steps 3–8 with `--force` afterwards. Until you do, `src.evaluation` marks old results as **stale** instead of comparing them.

### Comparing frozen vs fine-tuned CodeBERT

1. Run step 5 (frozen), then step 7 (fine-tuned). Step 7 prints a side-by-side table at the end.
2. Run step 8. It creates `outputs/figures/frozen_vs_finetuned_codebert.png` and answers
   "Did unfreezing the final two layers improve performance?" in
   `outputs/metrics/comparison_summary.md`, including 95% bootstrap confidence intervals.

### Generating the graphs again

All graphs are recreated from saved files by `python -m src.evaluation`. No retraining is needed.
Per-model figures (confusion matrices, loss curves) are created when each model finishes training.

### Smoke test (checks that everything runs, ~2 minutes on CPU)

```bash
python tests/smoke_test.py                 # includes CodeBERT (downloads ~500 MB once)
python tests/smoke_test.py --skip-pretrained
```

It trains on a tiny subset and writes to a temporary folder, so it never touches `outputs/`.

---

## D. Where everything is saved

```text
outputs/
├── figures/        all PNG graphs
├── metrics/        reports, CSV tables, *_metrics.json, classification reports
│   └── predictions/  one CSV per model and split (label, score, prediction)
├── models/         checkpoints, vectorizers, vocabularies, tokenizer files, configs
├── features/       cached frozen-encoder features (safe to delete; recomputed when needed)
└── logs/           one log file per experiment
```

---

## E. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You are not in the project root, or you ran `python src/x.py`. `cd` into the project root and use `python -m src.x`. |
| `Required file not found: .../data/delulu.csv` | Copy `delulu.csv` into `NeuralProject/data/`. |
| `Required file not found: .../data/processed/train.csv` | Run `python -m src.data_split` first. |
| Notebook cell 2: `Could not find the NeuralProject folder` | Set `PROJECT_ROOT = '/content/drive/MyDrive/<your path>/NeuralProject'` manually. |
| `CUDA available: False` on Colab | `Runtime -> Change runtime type -> T4 GPU`, then restart from cell 1. |
| `CUDA out of memory` | `Runtime -> Restart session` and run only the step you need. You can also lower `batch_size` in `src/config.py` (`FINETUNE_CONFIG`: 16 -> 8 and `grad_accumulation_steps`: 2 -> 4 keeps the effective batch size at 32). |
| `Could not load 'microsoft/codebert-base'` | No internet, or Hugging Face is down. Check the connection and retry. |
| `Wrong layers are trainable/frozen - refusing to train` | The layer-check safety guard fired. Make sure `UNFREEZE_LAST_N_LAYERS = 2` in `src/config.py` and that the code in `unfreeze_last_n_layers` was not modified. |
| A step says "already trained - skipping" | That is intended (restart-safe). Add `--force` to retrain. |
| A model shows as `stale` in the comparison | Its predictions were made on a different test split. Retrain it with `--force`. |
| Colab session ended and files disappeared | The project was in `/content`, which is temporary. Keep it on Google Drive, or download `outputs.zip` (cell 15). |
| `pip` tries to reinstall torch on Colab | It should not, because `torch>=2.3` is already satisfied. If it does, skip that cell and run `pip install -q transformers scikit-learn scipy pandas matplotlib joblib` instead. |
