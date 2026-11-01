# Final comparison - answers generated from saved test predictions

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

## Results table

| display_name | status | accuracy | precision | recall | f1 | roc_auc | pr_auc | pairwise_accuracy | f1_ci_low | f1_ci_high | n_test_examples |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | completed | 0.7213 | 0.6825 | 0.8277 | 0.7481 | 0.7981 | 0.7711 | 0.9155 | 0.7235 | 0.7716 | 592 |
| Linear SVM | completed | 0.7196 | 0.6816 | 0.8243 | 0.7462 | 0.8007 | 0.7683 | 0.9223 | 0.7211 | 0.7714 | 592 |
| GRU | completed | 0.6486 | 0.6803 | 0.5608 | 0.6148 | 0.6912 | 0.6337 | 0.8226 | 0.5758 | 0.6535 | 592 |
| LSTM | completed | 0.6588 | 0.6958 | 0.5642 | 0.6231 | 0.6998 | 0.6393 | 0.8395 | 0.5819 | 0.6617 | 592 |
| CodeBERT (frozen) | completed | 0.7365 | 0.7201 | 0.7736 | 0.7459 | 0.8184 | 0.8017 | 0.9139 | 0.7150 | 0.7747 | 592 |
| CodeBERT (last 2 layers) | completed | 0.7449 | 0.7636 | 0.7095 | 0.7356 | 0.8259 | 0.8065 | 0.9003 | 0.7011 | 0.7690 | 592 |
| GraphCodeBERT (frozen) | completed | 0.7399 | 0.7432 | 0.7331 | 0.7381 | 0.8180 | 0.7892 | 0.9206 | 0.7049 | 0.7685 | 592 |
| GraphCodeBERT (last 2 layers) | completed | 0.7466 | 0.7786 | 0.6892 | 0.7312 | 0.8359 | 0.8045 | 0.9307 | 0.6920 | 0.7641 | 592 |