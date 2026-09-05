"""
preprocessing.py
------------------
Reproduit EXACTEMENT le pipeline de nettoyage / encodage appliqué dans
construction_modele.ipynb, à partir des artefacts figés dans
models/preprocessing_artifacts.json (générés une seule fois en rejouant le
train_test_split(random_state=42) du notebook, afin de garantir que les
LabelEncoders utilisés ici sont identiques à ceux vus par le modèle pendant
l'entraînement).

Ce module est utilisé aussi bien par :
  - la page Auto-évaluation (une seule ligne construite depuis un formulaire),
  - la page Tableau de bord Admin (upload CSV en lot, colonnes partielles ou
    nommées différemment).
"""

from __future__ import annotations
import json
import re
import difflib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ARTIFACTS_PATH = Path(__file__).resolve().parent.parent / "models" / "preprocessing_artifacts.json"

with open(ARTIFACTS_PATH, "r", encoding="utf-8") as f:
    _ARTIFACTS = json.load(f)

FEATURE_ORDER: List[str] = _ARTIFACTS["feature_order"]              # 43 colonnes, ordre exact du modèle
CATEGORICAL_COLUMNS: List[str] = _ARTIFACTS["categorical_columns"]
LABEL_ENCODERS: Dict[str, Dict[str, int]] = _ARTIFACTS["label_encoders"]
MENTAL_HEALTH_SHIFT: Dict[str, float] = _ARTIFACTS["mental_health_shift"]
MEDIAN_IMPUTATION: Dict[str, float] = _ARTIFACTS["median_imputation"]

MENTAL_HEALTH_COLS = ["Stress Value", "Anxiety Value", "Depression Value"]

# ---------------------------------------------------------------------------
# Synonymes de colonnes : un institut partenaire n'utilisera pas forcément les
# mêmes intitulés exacts. On normalise (minuscule, sans espace/underscore) et on
# propose une table de correspondance + un fallback par similarité de texte.
# ---------------------------------------------------------------------------
COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "Age": ["age", "age_group", "tranche_age", "student_age"],
    "Gender": ["gender", "sexe", "sex"],
    "University": ["university", "universite", "institution", "établissement", "etablissement", "school"],
    "Department": ["department", "departement", "filiere", "filière", "major", "program", "programme"],
    "Academic_Year": ["academic_year", "annee_academique", "year", "niveau", "grade_level", "study_year"],
    "Current_CGPA": ["current_cgpa", "cgpa", "gpa", "moyenne", "note_moyenne", "grade_average"],
    "waiver_or_scholarship": ["waiver_or_scholarship", "scholarship", "bourse", "waiver"],
    "Stress Value": ["stress_value", "stress score", "pss_total", "stress total"],
    "Stress Label": ["stress_label", "stress_category", "niveau_stress"],
    "Anxiety Value": ["anxiety_value", "anxiety score", "gad_total", "anxiete_total"],
    "Anxiety Label": ["anxiety_label", "anxiety_category", "niveau_anxiete"],
    "Depression Value": ["depression_value", "depression score", "phq_total", "depression_total"],
    "Depression Label": ["depression_label", "depression_category", "niveau_depression"],
    "Financial_Debtor_proxy": ["financial_debtor_proxy", "financial_debtor", "endettement", "debiteur", "is_debtor"],
    "Tuition_UpToDate_proxy": ["tuition_uptodate_proxy", "tuition_paid", "frais_a_jour", "tuition_status"],
    "Parental_SES_proxy": ["parental_ses_proxy", "parental_ses", "ses_parent", "categorie_socioprofessionnelle", "csp"],
    "Economic_Vulnerability_Index": ["economic_vulnerability_index", "vulnerability_index", "indice_vulnerabilite"],
}
for i in range(1, 11):
    COLUMN_SYNONYMS[f"PSS{i}"] = [f"pss{i}", f"pss_{i}", f"stress_item_{i}", f"stress{i}"]
for i in range(1, 8):
    COLUMN_SYNONYMS[f"GAD{i}"] = [f"gad{i}", f"gad_{i}", f"anxiety_item_{i}", f"anxiety{i}"]
for i in range(1, 10):
    COLUMN_SYNONYMS[f"PHQ{i}"] = [f"phq{i}", f"phq_{i}", f"depression_item_{i}", f"depression{i}"]


def _normalize(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def map_columns_to_schema(df: pd.DataFrame, fuzzy_cutoff: float = 0.82) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Tente de faire correspondre les colonnes d'un CSV importé (upload admin) aux 43
    colonnes canoniques attendues par le modèle, en acceptant des synonymes ou des
    noms de colonnes proches (accents, majuscules, underscores, etc.).

    Retourne :
        - df renommé (colonnes reconnues renommées vers le nom canonique)
        - le mapping {colonne_originale: colonne_canonique} effectivement appliqué
        - la liste des colonnes canoniques toujours manquantes après le mapping
    """
    normalized_to_canonical = {_normalize(c): c for c in FEATURE_ORDER}
    for canonical, synonyms in COLUMN_SYNONYMS.items():
        for syn in synonyms:
            normalized_to_canonical.setdefault(_normalize(syn), canonical)

    rename_map: Dict[str, str] = {}
    already_used_canonical = set()

    for col in df.columns:
        norm = _normalize(col)
        if norm in normalized_to_canonical:
            canonical = normalized_to_canonical[norm]
            if canonical not in already_used_canonical:
                rename_map[col] = canonical
                already_used_canonical.add(canonical)
                continue
        # Repli : correspondance approximative (fautes de frappe, variantes mineures)
        close = difflib.get_close_matches(norm, normalized_to_canonical.keys(), n=1, cutoff=fuzzy_cutoff)
        if close:
            canonical = normalized_to_canonical[close[0]]
            if canonical not in already_used_canonical:
                rename_map[col] = canonical
                already_used_canonical.add(canonical)

    df_renamed = df.rename(columns=rename_map)
    missing = [c for c in FEATURE_ORDER if c not in df_renamed.columns]
    return df_renamed, rename_map, missing


# ---------------------------------------------------------------------------
# Nettoyage déterministe (identique au notebook)
# ---------------------------------------------------------------------------
def clean_cgpa(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    try:
        # Déjà numérique (ex: upload admin avec CGPA exact)
        return float(val)
    except ValueError:
        pass
    if " - " in val:
        low, high = map(float, val.split(" - "))
        return (low + high) / 2
    if "Bellow" in val or "Below" in val or "<" in val:
        nums = re.findall(r"\d+\.\d+|\d+", val)
        return float(nums[0]) - 0.25 if nums else np.nan
    if "Above" in val or ">" in val:
        nums = re.findall(r"\d+\.\d+|\d+", val)
        return float(nums[0]) + 0.25 if nums else np.nan
    return np.nan


def clean_age(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    try:
        return float(val)
    except ValueError:
        pass
    if "-" in val:
        try:
            low, high = map(int, val.split("-"))
            return (low + high) / 2
        except ValueError:
            pass
    if "Below" in val or "Bellow" in val or "<" in val:
        nums = re.findall(r"\d+", val)
        return float(nums[0]) - 2 if nums else np.nan
    if "Above" in val or ">" in val:
        nums = re.findall(r"\d+", val)
        return float(nums[0]) + 2 if nums else np.nan
    return np.nan


def apply_mental_health_shift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, shift in MENTAL_HEALTH_SHIFT.items():
        if col in df.columns and shift:
            df[col] = pd.to_numeric(df[col], errors="coerce") + shift
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes catégorielles avec les LabelEncoders figés à l'entraînement.
    Toute catégorie jamais vue à l'entraînement -> -1 (comme dans le notebook)."""
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        mapping = LABEL_ENCODERS[col]
        df[col] = df[col].astype(str).map(mapping)
        df[col] = df[col].fillna(-1).astype(int)
    return df


def impute_medians(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, median in MEDIAN_IMPUTATION.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)
    return df


def build_model_ready_frame(df: pd.DataFrame, allow_missing_columns: bool = False) -> pd.DataFrame:
    """
    Prend un DataFrame déjà mappé sur les noms canoniques (via map_columns_to_schema)
    et renvoie un DataFrame prêt pour model.predict(), dans l'ordre exact FEATURE_ORDER.

    Si allow_missing_columns=True (upload admin avec < 43 colonnes) : les colonnes
    canoniques absentes sont ajoutées en NaN. Le modèle XGBoost sous-jacent a été
    entraîné avec missing=nan et gère nativement les valeurs manquantes (il apprend,
    pour chaque split d'arbre, la direction par défaut à suivre en cas de valeur
    manquante) : la prédiction reste donc possible, avec une confiance dégradée à
    proportion du nombre de colonnes manquantes.
    """
    df = df.copy()

    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(clean_age)
    if "Current_CGPA" in df.columns:
        df["Current_CGPA"] = df["Current_CGPA"].apply(clean_cgpa)

    df = apply_mental_health_shift(df)
    df = encode_categoricals(df)
    df = impute_medians(df)

    for col in FEATURE_ORDER:
        if col not in df.columns:
            if not allow_missing_columns:
                raise ValueError(f"Colonne obligatoire manquante : {col}")
            df[col] = np.nan

    return df[FEATURE_ORDER]


def coverage_ratio(missing_columns: List[str]) -> float:
    return 1.0 - (len(missing_columns) / len(FEATURE_ORDER))
