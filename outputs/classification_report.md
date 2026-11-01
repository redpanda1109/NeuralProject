# Classification reports - all models and experiments

_Generated automatically by `python -m src.evaluation` (or `python -m src.evaluation --classification-report-only`) from the saved test predictions in `outputs/metrics/predictions/`. Do not edit by hand - re-run the command after training more models._

- **Task:** binary detection. Label **0 = correct** (golden completion), **1 = hallucinated** (hallucinated completion); the positive class is 1 = hallucinated.
- **Test set:** the saved split `data/processed/test.csv` (296 benchmark items = 592 binary examples, 50% per class). It is never used for training, tuning or early stopping.
- **Predictions:** exactly as saved by each model (neural models: probability >= 0.5; Linear SVM: decision value >= 0).
- A model is **Not yet evaluated** when it has no saved test predictions for the current split; no values are filled in for it.

## Summary - top two models

**Primary metric: test F1-score of the hallucinated class (`f1`), ties broken by ROC-AUC (`roc_auc`).** This is the metric the project's comparison already uses (`comparison_summary.md` picks the best model of each group by test F1). For a detector, F1 of the positive class balances precision (how many flagged completions really are hallucinated) and recall (how many hallucinations are caught). F1 alone can be inflated by predicting everything as hallucinated (F1 = 0.67 on this balanced test set), so accuracy and ROC-AUC are shown next to it.

| rank | model | type | f1 | accuracy | roc_auc | f1_95%_CI |
|---|---|---|---|---|---|---|
| 1 | CodeT5-small (last 2 layers) | Partially fine-tuned | 0.7938 | 0.7973 | 0.8730 | [0.7660, 0.8222] |
| 2 | CodeBERT (last 4 layers) | Partially fine-tuned (last FOUR layers) | 0.7636 | 0.7720 | 0.8745 | [0.7276, 0.7958] |

The two best models by test F1 are **CodeT5-small (last 2 layers)** and **CodeBERT (last 4 layers)**. Their 95% bootstrap F1 intervals overlap, so the order between them is not reliable (test set of 296 items; intervals of all models are in `outputs/metrics/model_comparison.csv`).

## Overview

| model | type | status | rank by F1 | f1 | accuracy |
|---|---|---|---|---|---|
| Logistic Regression | Traditional baseline | Evaluated | 6 | 0.7481 | 0.7213 |
| Linear SVM | Traditional baseline | Evaluated | 7 | 0.7462 | 0.7196 |
| GRU | Trained from scratch | Evaluated | 14 | 0.6148 | 0.6486 |
| LSTM | Trained from scratch | Evaluated | 13 | 0.6231 | 0.6588 |
| CodeBERT (frozen) | Frozen encoder | Evaluated | 8 | 0.7459 | 0.7365 |
| CodeBERT (last 2 layers) | Partially fine-tuned | Evaluated | 10 | 0.7356 | 0.7449 |
| GraphCodeBERT (frozen) | Frozen encoder | Evaluated | 9 | 0.7381 | 0.7399 |
| GraphCodeBERT (last 2 layers) | Partially fine-tuned | Evaluated | 11 | 0.7312 | 0.7466 |
| CodeT5-small (frozen) | Frozen encoder | Evaluated | 3 | 0.7632 | 0.7652 |
| CodeT5-small (last 2 layers) | Partially fine-tuned | Evaluated | 1 | 0.7938 | 0.7973 |
| UniXcoder (frozen) | Frozen encoder | Evaluated | 12 | 0.6922 | 0.6875 |
| UniXcoder (last 2 layers) | Partially fine-tuned | Evaluated | 4 | 0.7532 | 0.7399 |
| CodeBERT (last 4 layers) | Partially fine-tuned (last FOUR layers) | Evaluated | 2 | 0.7636 | 0.7720 |
| CodeBERT (frozen + new Transformer layer) | Added trainable layer | Evaluated | 5 | 0.7522 | 0.7618 |

## Binary task - classification report of every model (test set)

### 1. Logistic Regression (`logistic_regression`)

- **Setup:** Traditional baseline - TF-IDF features + Logistic Regression (no neural network)
- **Status:** Evaluated - 592 test examples, rank 6 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7811    0.6149    0.6881       296
hallucinated     0.6825    0.8277    0.7481       296

    accuracy                         0.7213       592
   macro avg     0.7318    0.7213    0.7181       592
weighted avg     0.7318    0.7213    0.7181       592
```

### 2. Linear SVM (`linear_svm`)

- **Setup:** Traditional baseline - TF-IDF features + Linear SVM (no neural network)
- **Status:** Evaluated - 592 test examples, rank 7 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7778    0.6149    0.6868       296
hallucinated     0.6816    0.8243    0.7462       296

    accuracy                         0.7196       592
   macro avg     0.7297    0.7196    0.7165       592
weighted avg     0.7297    0.7196    0.7165       592
```

### 3. GRU (`gru`)

- **Setup:** Trained from scratch - bidirectional GRU (all weights trained)
- **Status:** Evaluated - 592 test examples, rank 14 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.6264    0.7365    0.6770       296
hallucinated     0.6803    0.5608    0.6148       296

    accuracy                         0.6486       592
   macro avg     0.6534    0.6486    0.6459       592
weighted avg     0.6534    0.6486    0.6459       592
```

### 4. LSTM (`lstm`)

- **Setup:** Trained from scratch - bidirectional LSTM (all weights trained)
- **Status:** Evaluated - 592 test examples, rank 13 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.6335    0.7534    0.6883       296
hallucinated     0.6958    0.5642    0.6231       296

    accuracy                         0.6588       592
   macro avg     0.6647    0.6588    0.6557       592
weighted avg     0.6647    0.6588    0.6557       592
```

### 5. CodeBERT (frozen) (`codebert_frozen`)

- **Setup:** Frozen encoder - all 12 CodeBERT layers frozen, only the classification head trained
- **Status:** Evaluated - 592 test examples, rank 8 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7555    0.6993    0.7263       296
hallucinated     0.7201    0.7736    0.7459       296

    accuracy                         0.7365       592
   macro avg     0.7378    0.7365    0.7361       592
weighted avg     0.7378    0.7365    0.7361       592
```

### 6. CodeBERT (last 2 layers) (`codebert_finetuned_last2`)

- **Setup:** Partially fine-tuned - CodeBERT layers 10-11 + head trained, embeddings and layers 0-9 frozen
- **Status:** Evaluated - 592 test examples, rank 10 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7287    0.7804    0.7537       296
hallucinated     0.7636    0.7095    0.7356       296

    accuracy                         0.7449       592
   macro avg     0.7462    0.7449    0.7446       592
weighted avg     0.7462    0.7449    0.7446       592
```

### 7. GraphCodeBERT (frozen) (`graphcodebert_frozen`)

- **Setup:** Frozen encoder - all 12 GraphCodeBERT layers frozen, only the classification head trained
- **Status:** Evaluated - 592 test examples, rank 9 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7367    0.7466    0.7416       296
hallucinated     0.7432    0.7331    0.7381       296

    accuracy                         0.7399       592
   macro avg     0.7399    0.7399    0.7399       592
weighted avg     0.7399    0.7399    0.7399       592
```

### 8. GraphCodeBERT (last 2 layers) (`graphcodebert_finetuned_last2`)

- **Setup:** Partially fine-tuned - GraphCodeBERT layers 10-11 + head trained, embeddings and layers 0-9 frozen
- **Status:** Evaluated - 592 test examples, rank 11 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7212    0.8041    0.7604       296
hallucinated     0.7786    0.6892    0.7312       296

    accuracy                         0.7466       592
   macro avg     0.7499    0.7466    0.7458       592
weighted avg     0.7499    0.7466    0.7458       592
```

### 9. CodeT5-small (frozen) (`codet5_small_frozen`)

- **Setup:** Frozen encoder - all 6 CodeT5-small encoder blocks frozen, only the classification head trained
- **Status:** Evaluated - 592 test examples, rank 3 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7608    0.7736    0.7672       296
hallucinated     0.7698    0.7568    0.7632       296

    accuracy                         0.7652       592
   macro avg     0.7653    0.7652    0.7652       592
weighted avg     0.7653    0.7652    0.7652       592
```

### 10. CodeT5-small (last 2 layers) (`codet5_small_finetuned_last2`)

- **Setup:** Partially fine-tuned - CodeT5-small encoder blocks 4-5 + final encoder LayerNorm + head trained, embeddings and blocks 0-3 frozen
- **Status:** Evaluated - 592 test examples, rank 1 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7876    0.8142    0.8007       296
hallucinated     0.8077    0.7804    0.7938       296

    accuracy                         0.7973       592
   macro avg     0.7976    0.7973    0.7972       592
weighted avg     0.7976    0.7973    0.7972       592
```

### 11. UniXcoder (frozen) (`unixcoder_frozen`)

- **Setup:** Frozen encoder - all 12 UniXcoder layers frozen, only the classification head trained
- **Status:** Evaluated - 592 test examples, rank 12 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.6934    0.6723    0.6827       296
hallucinated     0.6820    0.7027    0.6922       296

    accuracy                         0.6875       592
   macro avg     0.6877    0.6875    0.6874       592
weighted avg     0.6877    0.6875    0.6874       592
```

### 12. UniXcoder (last 2 layers) (`unixcoder_finetuned_last2`)

- **Setup:** Partially fine-tuned - UniXcoder layers 10-11 + head trained, embeddings and layers 0-9 frozen
- **Status:** Evaluated - 592 test examples, rank 4 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7689    0.6858    0.7250       296
hallucinated     0.7165    0.7939    0.7532       296

    accuracy                         0.7399       592
   macro avg     0.7427    0.7399    0.7391       592
weighted avg     0.7427    0.7399    0.7391       592
```

### 13. CodeBERT (last 4 layers) (`codebert_finetuned_last4`)

- **Setup:** Partially fine-tuned (last FOUR layers) - CodeBERT layers 8-11 + head trained, embeddings and layers 0-7 frozen
- **Status:** Evaluated - 592 test examples, rank 2 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7539    0.8074    0.7798       296
hallucinated     0.7927    0.7365    0.7636       296

    accuracy                         0.7720       592
   macro avg     0.7733    0.7720    0.7717       592
weighted avg     0.7733    0.7720    0.7717       592
```

### 14. CodeBERT (frozen + new Transformer layer) (`codebert_extra_transformer_layer`)

- **Setup:** Added trainable layer - all of CodeBERT frozen; one NEW randomly initialised Transformer layer + head trained
- **Status:** Evaluated - 592 test examples, rank 5 of 14 by test F1.

```text
              precision    recall  f1-score   support

     correct     0.7429    0.8007    0.7707       296
hallucinated     0.7839    0.7230    0.7522       296

    accuracy                         0.7618       592
   macro avg     0.7634    0.7618    0.7615       592
weighted avg     0.7634    0.7618    0.7615       592
```

## Task B (4-class hallucination category) - separate task, not ranked above

Only hallucinated completions; labels = the four hallucination types. Shown for completeness because these baselines are also part of the project.

### Task B - Logistic Regression (`taskB_logistic_regression`)

- **Status:** Evaluated - 296 hallucinated test completions.

```text
                   precision    recall  f1-score   support

           import     0.9753    1.0000    0.9875        79
           method     0.6984    0.6567    0.6769        67
        parameter     0.7571    0.8030    0.7794        66
undefinedvariable     0.7561    0.7381    0.7470        84

         accuracy                         0.8041       296
        macro avg     0.7967    0.7995    0.7977       296
     weighted avg     0.8018    0.8041    0.8025       296
```

### Task B - Linear SVM (`taskB_linear_svm`)

- **Status:** Evaluated - 296 hallucinated test completions.

```text
                   precision    recall  f1-score   support

           import     0.9753    1.0000    0.9875        79
           method     0.6825    0.6418    0.6615        67
        parameter     0.7308    0.8636    0.7917        66
undefinedvariable     0.7838    0.6905    0.7342        84

         accuracy                         0.8007       296
        macro avg     0.7931    0.7990    0.7937       296
     weighted avg     0.8002    0.8007    0.7982       296
```
