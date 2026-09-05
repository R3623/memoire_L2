# 🎓 UniGlobal.Predict

Plateforme d'aide à la décision pour la **prédiction et la prévention de l'abandon
universitaire**, à destination des instituts universitaires du monde entier.

L'application combine :
- un modèle **XGBoost** entraîné sur 43 variables (démographiques, académiques,
  socio-économiques, et scores de santé mentale PSS-10 / GAD-7 / PHQ-9) ;
- l'explicabilité **SHAP** (TreeExplainer) pour comprendre *pourquoi* un étudiant
  est jugé à risque ;
- une interface **Streamlit** à 4 espaces : auto-évaluation étudiante, tableau de
  bord administrateur, explicabilité & simulation What-If, performances du modèle.

---

## 1. Structure du projet

```
uniglobal_app/
│
├── .streamlit/
│   └── config.toml                # Thème (couleurs, mode clair)
├── models/
│   ├── best_dropout_model.pkl     # Modèle XGBoost entraîné (fourni)
│   └── preprocessing_artifacts.json  # Encodeurs/médianes figés à l'entraînement
├── data/
│   ├── Processed_augmented_5000.csv  # Dataset d'entraînement (pour la démo/perf.)
│   ├── Stress.csv                 # Source des vraies questions PSS-10
│   ├── Anxiety.csv                # Source des vraies questions GAD-7
│   └── Depression.csv             # Source des vraies questions PHQ-9
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py           # Nettoyage, encodage, mapping colonnes/synonymes
│   ├── model_helper.py            # Chargement modèle, prédiction, SHAP, recommandations
│   ├── evaluation.py              # Reconstruction du jeu de test + métriques
│   └── questionnaires.py          # Vraies questions PSS/GAD/PHQ + barèmes cliniques
├── pages_app/
│   ├── __init__.py
│   ├── page_self_assessment.py    # Onglet 1 — Auto-évaluation étudiante
│   ├── page_admin_dashboard.py    # Onglet 2 — Tableau de bord admin
│   ├── page_explainability.py     # Onglet 3 — Explicabilité & What-If
│   └── page_model_performance.py  # Onglet 4 — Performances du modèle
├── app.py                         # Point d'entrée principal Streamlit
├── requirements.txt               # Dépendances
├── README.md                      # Ce fichier
└── guide.txt                      # Guide d'exécution et de déploiement détaillé
```

> **Note sur l'arborescence** : par rapport à la structure initialement demandée,
> j'ai ajouté deux fichiers dans `utils/` (`preprocessing.py`, `evaluation.py`,
> `questionnaires.py`) et un dossier `pages_app/` en plus de `app.py`. C'est une
> évolution *volontaire* : mettre les 4 onglets dans 4 modules séparés, importés
> par `app.py`, plutôt que 4 sections dans un seul fichier de 1000+ lignes.
> `app.py` reste bien le point d'entrée unique (`streamlit run app.py`) — voir
> **section 5 de `guide.txt`** pour le détail complet de la relation entre
> fichiers et comment recréer le projet manuellement.

---

## 2. Étape 1 — Environnement virtuel et packages

### Créer et activer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Contenu de `requirements.txt`

```
streamlit>=1.32,<2.0
pandas>=2.0,<2.3
numpy>=1.24,<2.1
xgboost>=1.7,<2.2
shap>=0.44,<0.46
plotly>=5.18,<6.0
scikit-learn>=1.3,<1.6
matplotlib>=3.7,<3.10
joblib>=1.3,<1.6
```

### Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ Au chargement du modèle, XGBoost peut afficher un `UserWarning` invitant à
> re-sauvegarder le modèle avec `Booster.save_model()` si sa version diffère de
> celle utilisée à l'entraînement. C'est un avertissement inoffensif (pas une
> erreur) : le modèle se charge et prédit normalement via `pickle`.

---

## 3. Étape 2 — Configuration, données et intégration du modèle

- **Thème** : `.streamlit/config.toml` définit un thème clair avec une couleur
  primaire bleu institutionnel (`#2E5EAA`), adapté à un contexte universitaire.
- **Modèle** : placez (ou laissez, il y est déjà) `best_dropout_model.pkl` dans
  `models/best_dropout_model.pkl`. Le fichier `models/preprocessing_artifacts.json`
  **doit rester à côté** : il contient les `LabelEncoder` et médianes exacts appris
  sur le jeu d'entraînement du notebook (même `train_test_split(random_state=42)`),
  indispensables pour que les nouvelles données soient encodées exactement comme
  à l'entraînement.
- **Données** : `data/Processed_augmented_5000.csv` sert de jeu de référence pour
  la page "Performances du Modèle" (reconstruction du jeu de test) et pour la
  démo de la page "Explicabilité". `Stress.csv`, `Anxiety.csv`, `Depression.csv`
  fournissent les intitulés réels des questions PSS-10 / GAD-7 / PHQ-9.

---

## 4. Étape 3 — Modules utilitaires (`utils/`)

| Fichier | Rôle |
|---|---|
| `preprocessing.py` | Reproduit le nettoyage du notebook (`clean_age`, `clean_cgpa`, shift santé mentale, encodage). Fournit `map_columns_to_schema()` pour reconnaître des colonnes synonymes lors d'un upload CSV admin, et `build_model_ready_frame()` qui produit un DataFrame prêt pour `model.predict()`, dans l'ordre exact des 43 features. |
| `model_helper.py` | Charge le `.pkl` (mis en cache), calcule les probabilités, instancie le `shap.TreeExplainer`, calcule le niveau de risque (Faible/Moyen/Élevé) et génère des recommandations d'accompagnement à partir des variables SHAP dominantes. |
| `evaluation.py` | Rejoue le `train_test_split` du notebook pour reconstruire le même jeu de test, et calcule les métriques (Accuracy, Precision, Recall, F1, AUC-ROC, matrice de confusion) à un seuil de décision donné. |
| `questionnaires.py` | Contient les vraies questions PSS-10 / GAD-7 / PHQ-9 (extraites de `Stress.csv` / `Anxiety.csv` / `Depression.csv`) et les barèmes cliniques permettant de dériver `Stress/Anxiety/Depression Value` et `Label` à partir des réponses brutes. |

---

## 5. Étape 4 — Application principale (`app.py` + `pages_app/`)

`app.py` construit la barre latérale de navigation (`st.sidebar.radio`) entre les
4 espaces, puis délègue le rendu à la page correspondante :

1. **🎓 Auto-évaluation (Étudiant)** — formulaire complet (profil, situation
   socio-économique, puis les 26 vraies questions PSS/GAD/PHQ), jauge de risque
   Plotly, variables SHAP dominantes, recommandations ciblées. Une alerte de
   soutien avec invitation à contacter un professionnel s'affiche si l'item PHQ-9
   (idées suicidaires) indique une réponse positive, indépendamment du score de
   décrochage.
2. **🏛️ Tableau de Bord Administrateur** — upload CSV avec reconnaissance
   automatique des colonnes synonymes, tolérance aux fichiers partiels (<43
   colonnes, en s'appuyant sur la gestion native des valeurs manquantes de
   XGBoost), KPIs, graphiques Plotly, table filtrable et export CSV des étudiants
   prioritaires.
3. **🔬 Explicabilité & What-If** — Waterfall / Bar plot / Force plot SHAP pour un
   étudiant donné (issu du dernier lot admin importé, ou d'un échantillon de
   démonstration), et un simulateur qui recalcule la probabilité en temps réel
   quand on modifie une variable (CGPA, stress, anxiété, dépression, vulnérabilité
   économique, âge).
4. **📐 Performances du Modèle** — métriques et matrice de confusion recalculées
   sur le vrai jeu de test du notebook, avec un slider de seuil de décision de 0
   à 1 qui met à jour les métriques en direct.

---

## 6. Lancer l'application

```bash
streamlit run app.py
```

Pour tout le détail (déploiement, création manuelle du projet, dépannage), voir
**`guide.txt`**.
