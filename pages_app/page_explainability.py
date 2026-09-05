"""Page 3 — Explicabilité (SHAP) & Analyse What-If"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils.preprocessing import FEATURE_ORDER, build_model_ready_frame
from utils.model_helper import predict_proba, get_explainer, risk_level

DATA_PATH_SAMPLE_ROWS = 200  # nombre de lignes chargées pour la démo si aucun lot admin n'est disponible


@st.cache_data(show_spinner=False)
def _load_demo_sample():
    df = pd.read_csv("data/Processed_augmented_5000.csv").drop(
        columns=["Dropout_Risk_Score", "Target_Dropout_Binary", "Target_Dropout_Class"]
    )
    return df.head(DATA_PATH_SAMPLE_ROWS).reset_index(drop=True)


def _get_source_data():
    """Renvoie (raw_df, ready_df) : priorité au lot importé dans le Tableau de bord Admin,
    sinon un échantillon de démonstration du dataset fourni."""
    if "admin_result_df" in st.session_state and "admin_ready_df" in st.session_state:
        return st.session_state["admin_result_df"].reset_index(drop=True), st.session_state["admin_ready_df"].reset_index(drop=True)
    raw = _load_demo_sample()
    ready = build_model_ready_frame(raw, allow_missing_columns=False)
    return raw, ready


def render():
    st.header("🔬 Explicabilité & Analyse What-If")
    raw_df, ready_df = _get_source_data()

    st.caption(
        "Les explications ci-dessous portent soit sur le dernier lot importé dans le "
        "Tableau de bord Administrateur, soit sur un échantillon de démonstration "
        "issu du jeu de données fourni."
    )

    idx = st.number_input("Index de l'étudiant à analyser", min_value=0, max_value=len(ready_df) - 1, value=0, step=1)
    student_ready = ready_df.iloc[[idx]]
    proba = float(predict_proba(student_ready)[0])
    level = risk_level(proba)

    st.metric("Probabilité de décrochage prédite", f"{proba:.1%}", level)

    explainer = get_explainer()
    shap_vals = explainer.shap_values(student_ready[FEATURE_ORDER])
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[-1] if np.ndim(base_value) > 0 else base_value

    st.divider()
    plot_type = st.radio("Type de graphique SHAP", ["Waterfall", "Bar plot", "Force plot"], horizontal=True)

    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=base_value,
        data=student_ready[FEATURE_ORDER].iloc[0].values,
        feature_names=FEATURE_ORDER,
    )

    if plot_type == "Waterfall":
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    elif plot_type == "Bar plot":
        order = np.argsort(-np.abs(shap_vals[0]))[:15]
        bar_df = pd.DataFrame({
            "Variable": [FEATURE_ORDER[i] for i in order],
            "Contribution SHAP": [shap_vals[0][i] for i in order],
        }).sort_values("Contribution SHAP")
        fig = go.Figure(go.Bar(
            x=bar_df["Contribution SHAP"], y=bar_df["Variable"], orientation="h",
            marker_color=["#E74C3C" if v > 0 else "#2ECC71" for v in bar_df["Contribution SHAP"]],
        ))
        fig.update_layout(title="Impact de chaque variable sur la prédiction", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.components.v1.html(
            shap.force_plot(base_value, shap_vals[0], student_ready[FEATURE_ORDER].iloc[0], matplotlib=False).html()
            + f"<script>{shap.getjs()}</script>",
            height=150,
        )

    st.divider()
    st.subheader("🧪 Simulateur What-If")
    st.caption("Modifiez une variable pour observer son impact sur la probabilité de décrochage prédite, toutes choses égales par ailleurs.")

    simulable_numeric = ["Current_CGPA", "Stress Value", "Anxiety Value", "Depression Value", "Economic_Vulnerability_Index", "Age"]
    ranges = {
        "Current_CGPA": (0.0, 4.0, 0.05),
        "Stress Value": (0, 40, 1),
        "Anxiety Value": (0, 21, 1),
        "Depression Value": (0, 27, 1),
        "Economic_Vulnerability_Index": (0, 6, 1),
        "Age": (16, 45, 1),
    }
    feature_to_vary = st.selectbox("Variable à simuler", simulable_numeric)
    lo, hi, step = ranges[feature_to_vary]
    current_val = float(student_ready.iloc[0][feature_to_vary])
    new_val = st.slider(f"Nouvelle valeur pour « {feature_to_vary} »", float(lo), float(hi), float(np.clip(current_val, lo, hi)), float(step))

    simulated = student_ready.copy()
    simulated.iloc[0, simulated.columns.get_loc(feature_to_vary)] = new_val
    new_proba = float(predict_proba(simulated)[0])

    c1, c2 = st.columns(2)
    c1.metric("Probabilité actuelle", f"{proba:.1%}")
    c2.metric("Probabilité simulée", f"{new_proba:.1%}", f"{(new_proba - proba):+.1%}")

    # Courbe de sensibilité : probabilité prédite pour un balayage complet de la variable
    sweep_values = np.linspace(lo, hi, 25)
    sweep_probas = []
    for v in sweep_values:
        tmp = student_ready.copy()
        tmp.iloc[0, tmp.columns.get_loc(feature_to_vary)] = v
        sweep_probas.append(float(predict_proba(tmp)[0]))

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=sweep_values, y=sweep_probas, mode="lines+markers", name="Probabilité prédite"))
    fig_sweep.add_vline(x=current_val, line_dash="dash", line_color="gray", annotation_text="Valeur actuelle")
    fig_sweep.add_vline(x=new_val, line_dash="dot", line_color="#E74C3C", annotation_text="Valeur simulée")
    fig_sweep.update_layout(
        title=f"Sensibilité de la probabilité de décrochage à « {feature_to_vary} »",
        xaxis_title=feature_to_vary, yaxis_title="Probabilité de décrochage", height=400,
    )
    st.plotly_chart(fig_sweep, use_container_width=True)
