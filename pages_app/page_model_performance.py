"""Page 4 — Performances du Modèle"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.evaluation import metrics_at_threshold, roc_curve_data


def render():
    st.header("📐 Performances du Modèle")
    st.caption(
        "Les métriques ci-dessous sont recalculées sur le même jeu de test "
        "(20% des données, `random_state=42`, stratifié) que celui utilisé dans le "
        "notebook `construction_modele.ipynb`, afin de rester rigoureusement "
        "cohérentes avec l'entraînement."
    )

    threshold = st.slider("Seuil de décision (probabilité à partir de laquelle un étudiant est classé « décrochage »)", 0.0, 1.0, 0.5, 0.01)
    metrics = metrics_at_threshold(threshold)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['Precision']:.1%}")
    c3.metric("Recall", f"{metrics['Recall']:.1%}")
    c4.metric("F1-Score", f"{metrics['F1-Score']:.1%}")
    c5.metric("AUC-ROC", f"{metrics['AUC-ROC']:.3f}")

    st.divider()
    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm = metrics["confusion_matrix"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Prédit : Non-décrochage", "Prédit : Décrochage"],
            y=["Réel : Non-décrochage", "Réel : Décrochage"],
            colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False,
        ))
        fig_cm.update_layout(title=f"Matrice de confusion (seuil = {threshold:.2f})", height=420)
        fig_cm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        fpr, tpr, auc = roc_curve_data()
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Modèle (AUC = {auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard (AUC = 0.5)", line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title="Courbe ROC (jeu de test)", xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs", height=420)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()
    st.subheader("À propos du modèle")
    st.markdown(
        "- **Algorithme** : `XGBoostClassifier` (sélectionné dans le notebook comme "
        "meilleur modèle selon l'AUC-ROC, parmi Régression Logistique, KNN, Random "
        "Forest et XGBoost).\n"
        "- **Variables** : 43 variables (démographiques, académiques, socio-économiques, "
        "et scores de santé mentale PSS / GAD-7 / PHQ-9).\n"
        "- **Fuite de données** : la variable `Dropout_Risk_Score` (utilisée pour "
        "construire la cible) a été explicitement exclue de l'entraînement.\n"
        "- **Seuil par défaut** : 0.5. Abaisser le seuil augmente le rappel (moins de "
        "décrocheurs manqués) au prix de plus de faux positifs ; l'augmenter fait "
        "l'inverse."
    )
