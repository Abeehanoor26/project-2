# Project 2: Supervised Learning (Fraud Detection Pipeline)
DecodeLabs | Data Science | Batch 2026

A leak-free fraud detection pipeline for a highly imbalanced dataset (~0.17% fraud).

## Setup
```bash
pip install -r requirements.txt
```
Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it next to the script.

## Run
```bash
python fraud_detection_pipeline.py --data creditcard.csv            # full run (5-fold, full grids)
python fraud_detection_pipeline.py --data creditcard.csv --fast     # small grids, 3-fold, 50 trees
python fraud_detection_pipeline.py --data creditcard.csv --max-rows 100000   # stratified subsample
python fraud_detection_pipeline.py --fast                           # no CSV -> synthetic demo data
```
The full run on all 284,807 rows is slow (SMOTE + 5-fold grid search on Random Forest).
Start with `--fast` or `--max-rows`, then do the full run.

## Design (mirrors the deck)
| Step | Choice | Why |
|---|---|---|
| Split | Stratified 80/20, done **first** | Test set keeps real-world imbalance, no leakage |
| Linear model | `StandardScaler -> SMOTE -> LogisticRegression` | LR is scale-sensitive; scaler fit on train fold only |
| Ensemble | `SMOTE -> RandomForest` | Trees are scale-invariant, no scaler needed |
| Pipeline class | `imblearn.pipeline.Pipeline` | Resamples only during `fit`, never on validation folds |
| Tuning | `GridSearchCV`, 5-fold stratified | Tunes `smote__k_neighbors` and model params together |
| Metrics | Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix | Accuracy is discarded |

## Outputs (`./outputs`)
`test_metrics.csv`, `best_params.json`, `cv_results_*.csv`, `confusion_matrices.png`,
`roc_pr_curves.png`, `best_model.joblib`

## Notes for your write-up
- The baseline "always legitimate" row shows ~0 recall despite ~99.8% "accuracy" (Trap #1).
- Discuss the Precision/Recall trade-off: SMOTE raises recall but lowers precision (more false declines).
  Try tuning the decision threshold or `sampling_strategy` to move along that curve.
- Results from the synthetic fallback data are only a smoke test. Report numbers from the real dataset.
