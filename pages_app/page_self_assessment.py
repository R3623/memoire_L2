"""Page 1 — Auto-évaluation (Étudiant)"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.preprocessing import build_model_ready_frame, LABEL_ENCODERS
from utils.model_helper import predict_proba, risk_level, RISK_COLORS, shap_values_for, top_shap_drivers, generate_recommendations
from utils.questionnaires import PSS_SCALE, GAD_SCALE, PHQ_SCALE, score_to_label, PHQ9_ITEM_ID


def _options(col: str) -> list[str]:
    return sorted(LABEL_ENCODERS[col].keys())


def _render_likert_block(scale, key_prefix: str) -> dict:
    st.markdown(f"**{scale.name}**")
    answers = {}
    for item in scale.items:
        val = st.select_slider(
            item.question,
            options=list(range(0, item.scale_max + 1)),
            value=0,
            format_func=lambda v, opts=scale.options: opts[v] if v < len(opts) else str(v),
            key=f"{key_prefix}_{item.item_id}",
        )
        answers[item.item_id] = val
    return answers


def render():
    st.header("🎓 Auto-évaluation étudiante")
    st.caption(
        "Ce questionnaire est strictement confidentiel. Il permet d'estimer un niveau "
        "de risque de décrochage et de proposer un accompagnement adapté — il ne "
        "remplace en aucun cas un avis médical ou psychologique professionnel."
    )

    with st.form("self_assessment_form"):
        st.subheader("1. Profil général")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge", min_value=16, max_value=60, value=21)
            gender = st.selectbox("Genre", _options("Gender"))
        with c2:
            department = st.selectbox("Département / Filière", _options("Department"))
        with c3:
            academic_year = st.selectbox("Année académique", _options("Academic_Year"))
            cgpa = st.slider("CGPA actuel", 0.0, 4.0, 3.0, 0.01)

        university = _options("University")[0]

        waiver = st.radio("Bénéficiez-vous d'une bourse ou d'une exonération de frais ?", _options("waiver_or_scholarship"), horizontal=True)

        st.divider()
        st.subheader("2. Situation socio-économique")
        c4, c5, c6 = st.columns(3)
        with c4:
            financial_debtor = st.radio("Avez-vous des dettes ou emprunts en cours ?", _options("Financial_Debtor_proxy"), horizontal=True)
        with c5:
            tuition_up_to_date = st.radio("Vos frais de scolarité sont-ils à jour ?", _options("Tuition_UpToDate_proxy"), horizontal=True)
        with c6:
            parental_ses = st.selectbox("Catégorie socio-professionnelle des parents", _options("Parental_SES_proxy"))

        st.divider()
        pss_answers = _render_likert_block(PSS_SCALE, "pss")
        st.divider()
        gad_answers = _render_likert_block(GAD_SCALE, "gad")
        st.divider()
        phq_answers = _render_likert_block(PHQ_SCALE, "phq")

        submitted = st.form_submit_button("Évaluer mon risque de décrochage", type="primary", use_container_width=True)

    if not submitted:
        return

    # Alerte de soutien immédiate si l'item PHQ-9 (idées suicidaires / auto-agression) > 0,
    # indépendamment du score global de décrochage.
    if phq_answers.get(PHQ9_ITEM_ID, 0) > 0:
        st.error(
            "💛 **Vous n'êtes pas seul(e).** Votre réponse indique que vous avez pu avoir des pensées "
            "difficiles ces derniers temps. Nous vous encourageons vivement à en parler rapidement à "
            "quelqu'un de confiance : le service de santé mentale / psychologique de votre université, "
            "un médecin, ou une ligne d'écoute locale. Si vous êtes en danger immédiat, contactez les "
            "services d'urgence de votre pays dès maintenant."
        )

    stress_score = sum(pss_answers.values())
    anxiety_score = sum(gad_answers.values())
    depression_score = sum(phq_answers.values())
    stress_label = score_to_label(stress_score, PSS_SCALE.thresholds)
    anxiety_label = score_to_label(anxiety_score, GAD_SCALE.thresholds)
    depression_label = score_to_label(depression_score, PHQ_SCALE.thresholds)

    # Indice de vulnérabilité économique (0-6) : formule simplifiée et documentée,
    # combinant endettement, retard de paiement, et catégorie socio-professionnelle des parents.
    ses_penalty = {
        "Cadre_intellectuel_superieur": 0,
        "Employe_technicien": 1,
        "Ouvrier_metier_manuel": 2,
        "Non_specifie_ou_sans_emploi": 2,
    }
    econ_index = (2 if financial_debtor == "Yes" else 0) + (2 if tuition_up_to_date == "No" else 0) + ses_penalty.get(parental_ses, 1)
    econ_index = min(econ_index, 6)

    row = {
        "Age": age, "Gender": gender, "University": university, "Department": department,
        "Academic_Year": academic_year, "Current_CGPA": cgpa, "waiver_or_scholarship": waiver,
        **pss_answers, "Stress Value": stress_score, "Stress Label": stress_label,
        **gad_answers, "Anxiety Value": anxiety_score, "Anxiety Label": anxiety_label,
        **phq_answers, "Depression Value": depression_score, "Depression Label": depression_label,
        "Financial_Debtor_proxy": financial_debtor, "Tuition_UpToDate_proxy": tuition_up_to_date,
        "Parental_SES_proxy": parental_ses, "Economic_Vulnerability_Index": econ_index,
    }
    df_raw = pd.DataFrame([row])
    df_ready = build_model_ready_frame(df_raw, allow_missing_columns=False)
    proba = float(predict_proba(df_ready)[0])
    level = risk_level(proba)

    st.divider()
    st.subheader("Résultat de l'évaluation")

    col_gauge, col_scores = st.columns([1.2, 1])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            title={"text": f"Probabilité de décrochage — Risque {level}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[level]},
                "steps": [
                    {"range": [0, 33], "color": "#E9F9EF"},
                    {"range": [33, 66], "color": "#FDF2E3"},
                    {"range": [66, 100], "color": "#FBE7E6"},
                ],
            },
        ))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_scores:
        st.metric("Score de stress (PSS)", f"{stress_score}/40", stress_label)
        st.metric("Score d'anxiété (GAD-7)", f"{anxiety_score}/21", anxiety_label)
        st.metric("Score de dépression (PHQ-9)", f"{depression_score}/27", depression_label)
        st.metric("Indice de vulnérabilité économique", f"{econ_index}/6")

    st.subheader("🎯 Facteurs déterminants de ce résultat")
    shap_vals, _base = shap_values_for(df_ready)
    drivers = top_shap_drivers(shap_vals[0], df_ready.iloc[0], n=5)
    driver_df = pd.DataFrame(drivers, columns=["Variable", "Contribution SHAP", "Valeur observée"])
    st.dataframe(driver_df, use_container_width=True, hide_index=True)

    st.subheader("📋 Recommandations d'accompagnement")
    for rec in generate_recommendations(drivers, proba):
        st.info(rec)
