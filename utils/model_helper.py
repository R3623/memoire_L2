"""
model_helper.py
------------------
Chargement du modèle XGBoost (.pkl), prédictions de probabilité de décrochage,
explicabilité SHAP (TreeExplainer), et logique métier (niveau de risque,
recommandations d'accompagnement).
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple

import pickle
import numpy as np
import pandas as pd
import shap

from utils.preprocessing import FEATURE_ORDER, MENTAL_HEALTH_COLS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_dropout_model.pkl"


@lru_cache(maxsize=1)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@lru_cache(maxsize=1)
def get_explainer():
    """TreeExplainer adapté à XGBoost. Mis en cache : coûteux à instancier."""
    model = load_model()
    return shap.TreeExplainer(model)


def predict_proba(df_ready: pd.DataFrame) -> np.ndarray:
    """df_ready doit déjà être passé par build_model_ready_frame (43 colonnes, ordre exact)."""
    model = load_model()
    df_ready = df_ready[FEATURE_ORDER]
    return model.predict_proba(df_ready)[:, 1]


def predict_class(df_ready: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    proba = predict_proba(df_ready)
    return (proba >= threshold).astype(int)


def shap_values_for(df_ready: pd.DataFrame):
    """Renvoie (shap_values, base_value) pour un batch de lignes déjà prêtes pour le modèle."""
    explainer = get_explainer()
    df_ready = df_ready[FEATURE_ORDER]
    sv = explainer.shap_values(df_ready)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[-1] if np.ndim(base_value) > 0 else base_value
    return np.array(sv), base_value


def risk_level(proba: float) -> str:
    if proba < 0.33:
        return "Faible"
    if proba < 0.66:
        return "Moyen"
    return "Élevé"


RISK_COLORS = {"Faible": "#2ECC71", "Moyen": "#F39C12", "Élevé": "#E74C3C"}


def top_shap_drivers(shap_row: np.ndarray, feature_row: pd.Series, n: int = 5) -> List[Tuple[str, float, float]]:
    """Renvoie les n variables ayant le plus contribué (en valeur absolue) à la
    prédiction pour une observation donnée : (nom_variable, valeur_shap, valeur_brute)."""
    order = np.argsort(-np.abs(shap_row))[:n]
    result = []
    for idx in order:
        col = FEATURE_ORDER[idx]
        result.append((col, float(shap_row[idx]), feature_row.get(col, np.nan)))
    return result


# ---------------------------------------------------------------------------
# Recommandations d'accompagnement ciblées
# ---------------------------------------------------------------------------
_RECOMMENDATIONS: Dict[str, str] = {
    "Stress Value": "Score de stress académique élevé : orienter vers le service d'accompagnement psychologique et revoir la charge de travail / le planning d'études avec un tuteur.",
    "Anxiety Value": "Score d'anxiété élevé : proposer un suivi avec un(e) psychologue universitaire et des techniques de gestion du stress (respiration, méthode Pomodoro pour les révisions).",
    "Depression Value": "Score de dépression élevé : alerter le service de santé mentale de l'université ; un contact humain rapide et bienveillant est recommandé en priorité.",
    "Current_CGPA": "CGPA en baisse ou faible : proposer un tutorat académique ciblé et un plan de rattrapage avec l'enseignant référent.",
    "Financial_Debtor_proxy": "Situation d'endettement détectée : orienter vers le service des bourses / aides financières d'urgence de l'université.",
    "Tuition_UpToDate_proxy": "Frais de scolarité non à jour : proposer un échéancier de paiement et vérifier l'éligibilité à une bourse.",
    "Parental_SES_proxy": "Contexte socio-économique familial vulnérable : proposer un accompagnement social et des dispositifs de bourse au mérite/social.",
    "Economic_Vulnerability_Index": "Vulnérabilité économique globale élevée : orienter vers le guichet social de l'université pour un bilan complet des aides disponibles.",
    "waiver_or_scholarship": "Absence de bourse ou d'exonération : vérifier l'éligibilité de l'étudiant à un dispositif de bourse.",
    "Academic_Year": "Année académique à risque (souvent 1ère année) : renforcer le mentorat par les pairs et l'intégration.",
}

def generate_recommendations(top_drivers: List[Tuple[str, float, float]], proba: float) -> List[str]:
    """Construit une liste de recommandations personnalisées à partir des facteurs
    SHAP les plus influents (uniquement ceux qui augmentent le risque, shap > 0)."""
    recs = []
    for col, shap_val, _raw_val in top_drivers:
        if shap_val <= 0:
            continue
        base_col = col
        if base_col in _RECOMMENDATIONS:
            recs.append(_RECOMMENDATIONS[base_col])
        elif base_col.startswith("PSS"):
            recs.append(_RECOMMENDATIONS["Stress Value"])
        elif base_col.startswith("GAD"):
            recs.append(_RECOMMENDATIONS["Anxiety Value"])
        elif base_col.startswith("PHQ"):
            recs.append(_RECOMMENDATIONS["Depression Value"])
    # dédoublonnage en conservant l'ordre
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            unique_recs.append(r)
            seen.add(r)
    if not unique_recs:
        if proba >= 0.66:
            unique_recs.append("Profil à risque élevé sans facteur dominant unique : recommander un entretien individuel de suivi global avec un conseiller académique.")
        else:
            unique_recs.append("Aucun facteur de risque majeur détecté : maintenir un suivi de routine.")
    return unique_recs[:5]
