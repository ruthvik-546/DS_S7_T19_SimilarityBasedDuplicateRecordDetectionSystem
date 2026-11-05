"""
models.py
---------
Implements the "Classification / Decision" and "Hyperparameter Tuning"
blocks: trains Logistic Regression, Random Forest and XGBoost on the
pair-similarity features, tunes them with cross-validated grid search,
and reports accuracy / precision / recall / F1-score for each.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from xgboost import XGBClassifier

FEATURE_COLS = [
    "name_levenshtein", "name_jaro_winkler", "email_levenshtein",
    "address_levenshtein", "address_jaro_winkler", "name_jaccard",
    "address_jaccard", "phone_exact_match", "email_exact_match",
    "tfidf_cosine", "combined_score",
]


def split_data(pair_df, test_size=0.3, random_state=42):
    X = pair_df[FEATURE_COLS]
    y = pair_df["is_duplicate"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Grid-search + cross-validate three models, return metrics + fitted best estimators."""
    results = {}
    fitted_models = {}

    # --- Logistic Regression ---
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid={"C": [0.1, 1.0, 10.0]},
        cv=5, scoring="f1",
    )
    lr_grid.fit(X_train, y_train)
    fitted_models["Logistic Regression"] = lr_grid.best_estimator_

    # --- Random Forest ---
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [4, 8, None]},
        cv=5, scoring="f1",
    )
    rf_grid.fit(X_train, y_train)
    fitted_models["Random Forest"] = rf_grid.best_estimator_

    # --- XGBoost ---
    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid={"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        cv=5, scoring="f1",
    )
    xgb_grid.fit(X_train, y_train)
    fitted_models["XGBoost"] = xgb_grid.best_estimator_

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "best_params": (lr_grid.best_params_ if name == "Logistic Regression"
                             else rf_grid.best_params_ if name == "Random Forest"
                             else xgb_grid.best_params_),
        }

    return results, fitted_models


def pick_best_model(results, fitted_models, metric="f1_score"):
    best_name = max(results, key=lambda k: results[k][metric])
    return best_name, fitted_models[best_name], results[best_name]
