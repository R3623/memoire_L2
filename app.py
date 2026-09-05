"""
UniGlobal.Predict — Plateforme d'aide à la décision pour la prédiction et la
prévention de l'abandon universitaire.

Point d'entrée principal de l'application Streamlit.
"""
import streamlit as st

from pages_app import (
    page_self_assessment,
    page_admin_dashboard,
    page_explainability,
    page_model_performance,
)

st.set_page_config(
    page_title="UniGlobal.Predict",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "🎓 Auto-évaluation (Étudiant)": page_self_assessment,
    "🏛️ Tableau de Bord Administrateur": page_admin_dashboard,
    "🔬 Explicabilité & What-If": page_explainability,
    "📐 Performances du Modèle": page_model_performance,
}

with st.sidebar:
    st.title("UniGlobal.Predict")
    st.caption("Aide à la décision pour la prévention du décrochage universitaire")
    choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.markdown(
        "**À propos**\n\n"
        "UniGlobal.Predict combine un modèle XGBoost et l'explicabilité SHAP pour "
        "aider les instituts universitaires à identifier et accompagner les "
        "étudiants à risque de décrochage, à l'échelle mondiale."
    )
    st.caption("⚠️ Outil d'aide à la décision — ne remplace pas un avis professionnel.")

PAGES[choice].render()
