#!/usr/bin/env python3
"""
DecodeLabs | Data Science | Project 2
Supervised Learning: Leak-Free Fraud Detection Pipeline
=======================================================

What this script does (mirrors the project deck):
  1. Loads the credit-card fraud data (Kaggle "creditcard.csv": Time, V1..V28,
     Amount, Class). If the file is not found it generates a synthetic dataset
     with the same schema and the same extreme imbalance so the script always runs.
  2. Stratified 80/20 train/test split BEFORE any scaling or resampling.
  3. Builds two imblearn Pipelines:
        Logistic Regression : StandardScaler -> SMOTE -> LogisticRegression
        Random Forest       : SMOTE -> RandomForestClassifier   (no scaler needed)
  4. Tunes preprocessing + model hyperparameters together with GridSearchCV
     (5-fold StratifiedKFold). SMOTE is re-fit inside every training fold only,
     so the validation folds keep the real-world imbalance -> zero leakage.
  5. Evaluates on the untouched test set with Precision, Recall, F1, ROC-AUC
     (plus PR-AUC) and the confusion matrix. Accuracy is deliberately NOT used.
  6. Saves metrics, plots, best hyperparameters and the best model to ./outputs.

Usage:
    python fraud_detection_pipeline.py --data creditcard.csv
    python fraud_detection_pipeline.py --fast                  # quick demo run
    python fraud_detection_pipeline.py --data creditcard.csv --max-rows 100000

Requirements:  pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # NOT sklearn.pipeline.Pipeline
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Class"
FRAUD_RATE = 0.0017  # 0.17% as in the deck
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def make_synthetic(n_samples: int, seed: int) -> pd.DataFrame:
    """Synthetic stand-in for creditcard.csv (same columns, same imbalance)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[1 - FRAUD_RATE, FRAUD_RATE],
        class_sep=1.2,
        flip_y=0,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n_samples)))
    # Amount has a huge variance -> this is why Logistic Regression needs a scaler
    df["Amount"] = np.where(
        y == 1, rng.lognormal(4.0, 1.6, n_samples), rng.lognormal(3.5, 1.4, n_samples)
    ).round(2)
    df[TARGET] = y
    return df


def load_data(path: str | None, n_synthetic: int, max_rows: int | None, seed: int):
    if path and Path(path).exists():
        df = pd.read_csv(path)
        source = f"real dataset ({path})"
    else:
        if path:
            print(f"[warn] '{path}' not found -> falling back to synthetic data.")
        df = make_synthetic(n_synthetic, seed)
        source = "synthetic dataset (same schema as creditcard.csv)"

    if max_rows and len(df) > max_rows:  # stratified subsample for quick runs
        df, _ = train_test_split(
            df, train_size=max_rows, stratify=df[TARGET], random_state=seed
        )
        df = df.reset_index(drop=True)
    return df, source


# --------------------------------------------------------------------------- #
# 2. Pipelines + hyperparameter grids
# --------------------------------------------------------------------------- #
def build_candidates(seed: int, fast: bool):
    """Return {name: (imblearn Pipeline, param_grid)}."""
    lr_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # inside the pipeline -> no leakage
            ("smote", SMOTE(random_state=seed)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    rf_pipe = Pipeline(
        steps=[
            ("smote", SMOTE(random_state=seed)),  # trees are scale-invariant
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=50 if fast else 100, random_state=seed, n_jobs=1
                ),
            ),
        ]
    )

    if fast:
        lr_grid = {"smote__k_neighbors": [3, 5], "classifier__C": [0.1, 1.0]}
        rf_grid = {"smote__k_neighbors": [3, 5], "classifier__max_depth": [10, 20]}
    else:
        lr_grid = {
            "smote__k_neighbors": [3, 5, 7],
            "classifier__C": [0.01, 0.1, 1.0],
        }
        rf_grid = {
            "smote__k_neighbors": [3, 5, 7],
            "classifier__max_depth": [10, 20, None],
        }
    return {
        "Logistic Regression": (lr_pipe, lr_grid),
        "Random Forest": (rf_pipe, rf_grid),
    }


# --------------------------------------------------------------------------- #
# 3. Evaluation (no accuracy!)
# --------------------------------------------------------------------------- #
def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def make_plots(fitted: dict, X_test, y_test, out_dir: Path) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    # ROC + Precision-Recall curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, model in fitted.items():
        score = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, score, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, score, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set_title("ROC curve (test set)")
    axes[0].legend(loc="lower right")
    axes[1].set_title("Precision-Recall curve (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, len(fitted), figsize=(5.5 * len(fitted), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (name, model) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, model.predict(X_test),
            display_labels=["Legit", "Fraud"], cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Leak-free fraud detection pipeline")
    ap.add_argument("--data", default="creditcard.csv", help="path to creditcard.csv")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--synthetic-n", type=int, default=100_000,
                    help="rows for synthetic data if the CSV is missing")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stratified subsample size (speeds up tuning)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for GridSearchCV")
    ap.add_argument("--fast", action="store_true", help="small grids, 3-fold CV, 50 trees")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_folds = 3 if args.fast else args.cv

    # ---- Load ------------------------------------------------------------- #
    df, source = load_data(args.data, args.synthetic_n, args.max_rows, args.seed)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"\nData source : {source}")
    print(f"Transactions: {len(df):,} | Fraud: {int(y.sum()):,} ({y.mean():.2%})")
    print(f"An 'always legitimate' model would look ~{1 - y.mean():.2%} 'accurate' "
          "yet catch ZERO fraud -> accuracy is discarded.\n")

    # ---- Split FIRST (stratified) ---------------------------------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.seed
    )
    print(f"Train: {len(X_train):,} rows ({y_train.mean():.3%} fraud) | "
          f"Test: {len(X_test):,} rows ({y_test.mean():.3%} fraud, untouched)\n")

    # ---- Baseline: shows why Recall / ROC-AUC matter ---------------------- #
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    results = [evaluate("Baseline (always legit)", baseline, X_test, y_test)]

    # ---- Tune + fit each candidate ---------------------------------------- #
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
    scoring = {"roc_auc": "roc_auc", "recall": "recall",
               "precision": "precision", "f1": "f1"}
    fitted, best_params = {}, {}

    for name, (pipe, grid) in build_candidates(args.seed, args.fast).items():
        n_fits = np.prod([len(v) for v in grid.values()]) * cv_folds
        print(f">>> Tuning {name}: {n_fits} fits ({cv_folds}-fold CV)")
        t0 = time.time()
        search = GridSearchCV(
            pipe, grid, scoring=scoring, refit="roc_auc",
            cv=cv, n_jobs=args.n_jobs, verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"    done in {time.time() - t0:,.0f}s | best CV ROC-AUC = "
              f"{search.best_score_:.4f}")
        print(f"    best params: {search.best_params_}")

        fitted[name] = search.best_estimator_
        best_params[name] = {k: (None if v is None else v)
                             for k, v in search.best_params_.items()}
        results.append(evaluate(name, search.best_estimator_, X_test, y_test))

        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(out_dir / f"cv_results_{name.replace(' ', '_').lower()}.csv",
                     index=False)

    # ---- Report ----------------------------------------------------------- #
    res_df = pd.DataFrame(results).set_index("model")
    metric_cols = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    print("\n================ TEST-SET RESULTS (accuracy intentionally omitted) "
          "================")
    print(res_df[metric_cols].round(4).to_string())
    print("\nConfusion-matrix counts:")
    print(res_df[["TN", "FP", "FN", "TP"]].to_string())

    # Fraud-class report only: sklearn's default report prints an "accuracy"
    # row, which this project deliberately discards.
    for name, model in fitted.items():
        rep = classification_report(y_test, model.predict(X_test), labels=[0, 1],
                                    target_names=["Legit", "Fraud"], output_dict=True,
                                    zero_division=0)
        print(f"\n--- Fraud-class report: {name} ---")
        print(pd.DataFrame(rep).T.loc[["Legit", "Fraud"],
              ["precision", "recall", "f1-score", "support"]].round(4).to_string())

    # Random Forest feature importances
    if "Random Forest" in fitted:
        rf = fitted["Random Forest"].named_steps["classifier"]
        imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
        print("Top-10 Random Forest feature importances:")
        print(imp.round(4).to_string())

    # ---- Save artefacts --------------------------------------------------- #
    res_df.to_csv(out_dir / "test_metrics.csv")
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    make_plots(fitted, X_test, y_test, out_dir)

    best_name = max(fitted, key=lambda n: res_df.loc[n, "roc_auc"])
    joblib.dump(fitted[best_name], out_dir / "best_model.joblib")
    print(f"\nBest model by test ROC-AUC: {best_name} -> saved to "
          f"{out_dir / 'best_model.joblib'}")
    print(f"Artefacts written to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
