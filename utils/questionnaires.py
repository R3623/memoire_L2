"""
questionnaires.py
------------------
Contient les VRAIES questions des échelles psychométriques utilisées pour construire
le dataset (extraites de Stress.csv, Anxiety.csv, Depression.csv), ainsi que les
barèmes officiels permettant de transformer les réponses brutes en :
    - un score total (Stress Value / Anxiety Value / Depression Value)
    - une catégorie (Stress Label / Anxiety Label / Depression Label)

Ces barèmes ont été retrouvés en croisant, dans les fichiers CSV sources
(non augmentés), le score total avec le label associé (cf. section 1.6 du README).

Important : le dataset d'entraînement final (Processed_augmented_5000.csv) contient
du bruit d'augmentation (valeurs parfois négatives ou décimales) qui casse cette
correspondance propre score -> label. Pour le formulaire utilisateur, on applique
donc les barèmes cliniques "propres" retrouvés sur les données sources, ce qui est
plus cohérent pour une vraie auto-évaluation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LikertItem:
    item_id: str          # ex: "PSS1" -> nom de colonne attendu par le modèle
    question: str          # texte réel de la question (source CSV)
    scale_max: int          # valeur max de l'item (4 pour PSS/PHQ, 3 pour GAD)


@dataclass
class LikertScale:
    name: str
    items: List[LikertItem]
    options: List[str]      # libellés affichés pour chaque valeur 0..scale_max
    thresholds: List[tuple]  # [(min, max, label), ...] pour dériver le label depuis le score


# ---------------------------------------------------------------------------
# 1) PSS - Perceived (Academic) Stress Scale — 10 items, échelle 0-4
#    Options officielles PSS : Never / Almost never / Sometimes / Fairly often / Very often
# ---------------------------------------------------------------------------
PSS_OPTIONS = ["Jamais (0)", "Presque jamais (1)", "Parfois (2)", "Assez souvent (3)", "Très souvent (4)"]

PSS_ITEMS = [
    LikertItem("PSS1", "Au cours d'un semestre, à quelle fréquence vous êtes-vous senti(e) contrarié(e) à cause de quelque chose lié à vos affaires académiques ?", 4),
    LikertItem("PSS2", "À quelle fréquence avez-vous eu l'impression d'être incapable de contrôler les choses importantes de votre vie académique ?", 4),
    LikertItem("PSS3", "À quelle fréquence vous êtes-vous senti(e) nerveux(se) et stressé(e) à cause de la pression académique ?", 4),
    LikertItem("PSS4", "À quelle fréquence avez-vous eu l'impression de ne pas pouvoir faire face à toutes les activités académiques obligatoires (devoirs, quiz, examens) ?", 4),
    LikertItem("PSS5", "À quelle fréquence vous êtes-vous senti(e) confiant(e) dans votre capacité à gérer vos problèmes académiques / universitaires ?", 4),
    LikertItem("PSS6", "À quelle fréquence avez-vous eu l'impression que les choses allaient dans le bon sens dans votre vie académique ?", 4),
    LikertItem("PSS7", "À quelle fréquence avez-vous été capable de contrôler les irritations liées à vos affaires académiques / universitaires ?", 4),
    LikertItem("PSS8", "À quelle fréquence avez-vous eu l'impression que votre performance académique était au top ?", 4),
    LikertItem("PSS9", "À quelle fréquence vous êtes-vous mis(e) en colère à cause de mauvaises performances ou notes échappant à votre contrôle ?", 4),
    LikertItem("PSS10", "À quelle fréquence avez-vous eu l'impression que les difficultés académiques s'accumulaient au point de ne plus pouvoir les surmonter ?", 4),
]

PSS_SCALE = LikertScale(
    name="PSS (Stress académique perçu)",
    items=PSS_ITEMS,
    options=PSS_OPTIONS,
    thresholds=[(0, 13, "Low Stress"), (14, 26, "Moderate Stress"), (27, 40, "High Perceived Stress")],
)

# ---------------------------------------------------------------------------
# 2) GAD-7 - Generalized Anxiety Disorder scale — 7 items, échelle 0-3
# ---------------------------------------------------------------------------
GAD_OPTIONS = ["Jamais (0)", "Plusieurs jours (1)", "Plus de la moitié des jours (2)", "Presque tous les jours (3)"]

GAD_ITEMS = [
    LikertItem("GAD1", "Au cours d'un semestre, à quelle fréquence vous êtes-vous senti(e) nerveux(se), anxieux(se) ou à bout à cause de la pression académique ?", 3),
    LikertItem("GAD2", "À quelle fréquence avez-vous été incapable d'arrêter de vous inquiéter à propos de vos affaires académiques ?", 3),
    LikertItem("GAD3", "À quelle fréquence avez-vous eu du mal à vous détendre à cause de la pression académique ?", 3),
    LikertItem("GAD4", "À quelle fréquence avez-vous été facilement agacé(e) ou irrité(e) à cause de la pression académique ?", 3),
    LikertItem("GAD5", "À quelle fréquence vous êtes-vous trop inquiété(e) de vos affaires académiques ?", 3),
    LikertItem("GAD6", "À quelle fréquence avez-vous été si agité(e) à cause de la pression académique qu'il était difficile de rester assis(e) ?", 3),
    LikertItem("GAD7", "À quelle fréquence vous êtes-vous senti(e) effrayé(e), comme si quelque chose de terrible pouvait arriver ?", 3),
]

GAD_SCALE = LikertScale(
    name="GAD-7 (Anxiété généralisée)",
    items=GAD_ITEMS,
    options=GAD_OPTIONS,
    thresholds=[(0, 4, "Minimal Anxiety"), (5, 9, "Mild Anxiety"), (10, 14, "Moderate Anxiety"), (15, 21, "Severe Anxiety")],
)

# ---------------------------------------------------------------------------
# 3) PHQ-9 - Patient Health Questionnaire (dépression) — 9 items, échelle 0-3
#    ATTENTION : l'item 9 porte sur les idées suicidaires / d'auto-agression.
#    Il est conservé tel quel (c'est un item clinique standard du PHQ-9), mais
#    l'application déclenche un message de soutien + ressources d'aide dès que
#    la réponse est différente de 0 (voir pages_app/page_self_assessment.py).
# ---------------------------------------------------------------------------
PHQ_OPTIONS = ["Jamais (0)", "Plusieurs jours (1)", "Plus de la moitié des jours (2)", "Presque tous les jours (3)"]

PHQ_ITEMS = [
    LikertItem("PHQ1", "Au cours d'un semestre, à quelle fréquence avez-vous eu peu d'intérêt ou de plaisir à faire les choses ?", 3),
    LikertItem("PHQ2", "À quelle fréquence vous êtes-vous senti(e) triste, déprimé(e) ou désespéré(e) ?", 3),
    LikertItem("PHQ3", "À quelle fréquence avez-vous eu du mal à vous endormir, à rester endormi(e), ou au contraire trop dormi ?", 3),
    LikertItem("PHQ4", "À quelle fréquence vous êtes-vous senti(e) fatigué(e) ou avec peu d'énergie ?", 3),
    LikertItem("PHQ5", "À quelle fréquence avez-vous eu peu d'appétit ou au contraire mangé excessivement ?", 3),
    LikertItem("PHQ6", "À quelle fréquence vous êtes-vous senti(e) mal dans votre peau, ou comme un échec, ou avez-vous eu l'impression d'avoir déçu votre famille ?", 3),
    LikertItem("PHQ7", "À quelle fréquence avez-vous eu du mal à vous concentrer (lecture, télévision, etc.) ?", 3),
    LikertItem("PHQ8", "À quelle fréquence avez-vous parlé ou bougé si lentement que d'autres l'ont remarqué, ou au contraire été si agité(e) que vous bougiez beaucoup plus que d'habitude ?", 3),
    LikertItem("PHQ9", "À quelle fréquence avez-vous eu des pensées comme quoi vous seriez mieux mort(e), ou des pensées de vous faire du mal ?", 3),
]

PHQ_SCALE = LikertScale(
    name="PHQ-9 (Dépression)",
    items=PHQ_ITEMS,
    options=PHQ_OPTIONS,
    thresholds=[
        (0, 0, "No Depression"),
        (1, 4, "Minimal Depression"),
        (5, 9, "Mild Depression"),
        (10, 14, "Moderate Depression"),
        (15, 19, "Moderately Severe Depression"),
        (20, 27, "Severe Depression"),
    ],
)


def score_to_label(score: int, thresholds) -> str:
    """Renvoie le label correspondant à un score, selon une liste de bornes (min, max, label)."""
    for lo, hi, label in thresholds:
        if lo <= score <= hi:
            return label
    # Sécurité : si le score dépasse les bornes connues, on renvoie l'extrême le plus proche
    return thresholds[-1][2] if score > thresholds[-1][1] else thresholds[0][2]


PHQ9_ITEM_ID = "PHQ9"  # utilisé par la page d'auto-évaluation pour détecter l'item sensible
