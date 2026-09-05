"""Page 2 — Tableau de bord Administrateur (analyse globale par lot)"""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from utils.preprocessing import map_columns_to_schema, build_model_ready_frame, coverage_ratio, FEATURE_ORDER
from utils.model_helper import predict_proba, risk_level


def render():
    st.header("🏛️ Tableau de bord Administrateur")
    st.caption(
        "Importez un fichier CSV d'étudiants (un ou plusieurs instituts). Les noms de "
        "colonnes proches ou synonymes des colonnes attendues sont automatiquement "
        "reconnus. Si moins de 43 colonnes sont fournies, une prédiction reste "
        "possible à partir des colonnes disponibles (le modèle gère nativement les "
        "valeurs manquantes), avec un indice de confiance dégradé en conséquence."
    )

    uploaded = st.file_uploader("Importer un fichier CSV d'étudiants", type=["csv"])
    if uploaded is None:
        st.info("En attente d'un fichier. Vous pouvez tester avec `data/Processed_augmented_5000.csv`.")
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Impossible de lire le fichier CSV : {e}")
        return

    mapped_df, rename_map, missing_cols = map_columns_to_schema(raw_df)
    coverage = coverage_ratio(missing_cols)

    with st.expander("🔎 Détail de la correspondance des colonnes", expanded=coverage < 1.0):
        st.write(f"**Colonnes reconnues :** {len(rename_map)} / {len(raw_df.columns)}")
        if rename_map:
            st.dataframe(
                pd.DataFrame({"Colonne du fichier": list(rename_map.keys()), "Colonne du modèle": list(rename_map.values())}),
                use_container_width=True, hide_index=True,
            )
        if missing_cols:
            st.warning(f"Colonnes du modèle non trouvées dans le fichier ({len(missing_cols)}) : {', '.join(missing_cols)}")

    st.metric("Couverture des variables du modèle", f"{coverage:.0%}")
    if coverage < 0.30:
        st.error(
            "Moins de 30% des variables du modèle sont disponibles : les prédictions "
            "seraient trop peu fiables. Merci de fournir un fichier plus complet "
            "(au minimum le CGPA, la situation économique, et si possible les scores "
            "de stress / anxiété / dépression)."
        )
        return
    if coverage < 1.0:
        st.warning(
            f"Fichier partiel ({coverage:.0%} des variables). Les prédictions restent "
            "calculées mais avec une confiance réduite pour les lignes concernées."
        )

    try:
        ready_df = build_model_ready_frame(mapped_df, allow_missing_columns=(coverage < 1.0))
    except Exception as e:
        st.error(f"Erreur de préparation des données : {e}")
        return

    proba = predict_proba(ready_df)
    result_df = raw_df.copy()
    result_df["Probabilite_Decrochage"] = proba
    result_df["Niveau_Risque"] = [risk_level(p) for p in proba]

    st.divider()
    st.subheader("📊 Indicateurs clés (KPIs)")
    n_students = len(result_df)
    n_high_risk = int((result_df["Niveau_Risque"] == "Élevé").sum())
    taux_risque = n_high_risk / n_students if n_students else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Étudiants analysés", f"{n_students:,}".replace(",", " "))
    k2.metric("Taux de risque élevé", f"{taux_risque:.1%}")
    k3.metric("Probabilité moyenne de décrochage", f"{proba.mean():.1%}")
    k4.metric("Risque moyen (Moyen+Élevé)", f"{(result_df['Niveau_Risque'] != 'Faible').mean():.1%}")

    st.divider()
    st.subheader("📈 Répartition et facteurs globaux")
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(result_df, names="Niveau_Risque", title="Répartition des niveaux de risque",
                          color="Niveau_Risque",
                          color_discrete_map={"Faible": "#2ECC71", "Moyen": "#F39C12", "Élevé": "#E74C3C"})
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_hist = px.histogram(result_df, x="Probabilite_Decrochage", nbins=30,
                                 title="Distribution des probabilités de décrochage")
        st.plotly_chart(fig_hist, use_container_width=True)

    if "Department" in mapped_df.columns or "Department" in result_df.columns:
        dep_col = "Department" if "Department" in result_df.columns else None
        if dep_col:
            fig_dep = px.bar(
                result_df.groupby(dep_col)["Probabilite_Decrochage"].mean().sort_values(ascending=False).reset_index(),
                x=dep_col, y="Probabilite_Decrochage", title="Probabilité moyenne de décrochage par département",
            )
            st.plotly_chart(fig_dep, use_container_width=True)

    st.divider()
    st.subheader("🚨 Étudiants prioritaires")
    min_proba = st.slider("Seuil minimal de probabilité à afficher", 0.0, 1.0, 0.5, 0.05)
    priority_df = result_df[result_df["Probabilite_Decrochage"] >= min_proba].sort_values(
        "Probabilite_Decrochage", ascending=False
    )
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Exporter la liste priorisée (CSV)",
        priority_df.to_csv(index=False).encode("utf-8"),
        file_name="etudiants_prioritaires.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.session_state["admin_result_df"] = result_df
    st.session_state["admin_ready_df"] = ready_df
