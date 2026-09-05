"""
evaluation.py
------------------
Reconstruit le même split train/test (80/20, random_state=42, stratifié) que celui
utilisé dans construction_modele.ipynb, afin de pouvoir recalculer les métriques
de performance (Accuracy, Precision, Recall, F1, AUC-ROC, matrice de confusion)
et faire varier le seuil de décision dynamiquement dans la page "Performances du
Modèle", sans avoir à ré-entraîner le modèle.
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from utils.preprocessing import build_model_ready_frame
from utils.model_helper import predict_proba

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Processed_augmented_5000.csv"


@lru_cache(maxsize=1)
def get_test_set():
    data = pd.read_csv(DATA_PATH)
    df = data.drop(columns=["Dropout_Risk_Score"])
    target_col = ["Target_Dropout_Binary", "Target_Dropout_Class"]
    X = df.drop(columns=target_col)
    y = df["Target_Dropout_Binary"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_ready = build_model_ready_frame(X_test, allow_missing_columns=False)
    return X_test_ready, y_test.reset_index(drop=True)


@lru_cache(maxsize=1)
def get_test_predictions():
    X_test_ready, y_test = get_test_set()
    y_proba = predict_proba(X_test_ready)
    return y_test.values, y_proba


def metrics_at_threshold(threshold: float) -> dict:
    y_test, y_proba = get_test_predictions()
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def roc_curve_data():
    y_test, y_proba = get_test_predictions()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr, roc_auc_score(y_test, y_proba)
