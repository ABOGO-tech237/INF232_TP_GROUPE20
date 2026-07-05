"""
Question 1 — Analyse statistique univariée des notes d'évaluation.

Répond au proviseur :
  - Comment se répartissent les résultats ?
  - Y a-t-il des élèves atypiques (outliers) ?
  - Visualisation simple pour le conseil pédagogique (box plot).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Permet d'importer src.* depuis n'importe quel point d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.chargement import charger_donnees_brutes, chemin_figures


def calculer_statistiques_descriptives(notes: pd.Series) -> pd.Series:
    """
    Calcule les indicateurs demandés dans le TP (Section 4.1).

    On s'appuie sur describe() puis on complète avec l'IQR
    pour repérer les valeurs aberrantes selon la règle de Tukey.
    """
    stats = notes.describe()
    q1, q3 = stats["25%"], stats["75%"]
    iqr = q3 - q1

    # Seuils Tukey : au-delà → outlier potentiel à signaler au conseil
    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr

    stats["IQR"] = iqr
    stats["borne_basse_tukey"] = borne_basse
    stats["borne_haute_tukey"] = borne_haute
    stats["nb_outliers_bas"] = int((notes < borne_basse).sum())
    stats["nb_outliers_haut"] = int((notes > borne_haute).sum())

    return stats


def identifier_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Liste les élèves dont la note sort de l'intervalle interquartile étendu."""
    notes = df["Notes"]
    q1, q3 = notes.quantile(0.25), notes.quantile(0.75)
    iqr = q3 - q1
    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr

    masque = (notes < borne_basse) | (notes > borne_haute)
    return df.loc[masque].sort_values("Notes")


def tracer_boxplot(notes: pd.Series, chemin_sortie: Path) -> None:
    """
    Boîte à moustaches : médiane, quartiles et outliers en un coup d'œil.

    Format volontairement épuré pour une présentation au conseil pédagogique.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        notes,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "#4C72B0", "alpha": 0.6},
        medianprops={"color": "darkred", "linewidth": 2},
    )
    ax.set_ylabel("Note à l'évaluation (/20)")
    ax.set_title(
        "Répartition des notes — évaluation interne de terminale\n"
        "(médiane en rouge, points atypiques affichés individuellement)"
    )
    ax.set_xticklabels(["Notes"])
    ax.grid(axis="y", alpha=0.3)

    # Annotation lisible pour un public non statisticien
    mediane = notes.median()
    ax.annotate(
        f"Médiane : {mediane:.1f}/20",
        xy=(1.05, mediane),
        fontsize=10,
        color="darkred",
    )

    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def tracer_histogramme(notes: pd.Series, chemin_sortie: Path) -> None:
    """Histogramme complémentaire pour visualiser la forme de la distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(notes, bins=20, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(notes.mean(), color="navy", linestyle="--", label=f"Moyenne : {notes.mean():.1f}")
    ax.axvline(notes.median(), color="darkred", linestyle="-", label=f"Médiane : {notes.median():.1f}")
    ax.set_xlabel("Note (/20)")
    ax.set_ylabel("Nombre d'élèves")
    ax.set_title("Distribution des notes à l'évaluation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def executer_analyse() -> dict:
    """Pipeline complet Question 1 : stats + figures + liste des outliers."""
    df = charger_donnees_brutes()
    notes = df["Notes"]

    stats = calculer_statistiques_descriptives(notes)
    outliers = identifier_outliers(df)

    dossier_fig = chemin_figures()
    tracer_boxplot(notes, dossier_fig / "boxplot_notes_evaluation.png")
    tracer_histogramme(notes, dossier_fig / "histogramme_notes_evaluation.png")

    # Export intermédiaire pour traçabilité (data/interim/)
    chemin_interim = Path(__file__).resolve().parents[2] / "data" / "interim"
    chemin_interim.mkdir(parents=True, exist_ok=True)
    stats.to_csv(chemin_interim / "stats_descriptives_notes.csv")

    return {
        "statistiques": stats,
        "outliers": outliers,
        "figures": [
            dossier_fig / "boxplot_notes_evaluation.png",
            dossier_fig / "histogramme_notes_evaluation.png",
        ],
    }


if __name__ == "__main__":
    resultats = executer_analyse()
    stats = resultats["statistiques"]

    print("=== Question 1 — Statistiques descriptives (Notes) ===")
    print(f"Moyenne      : {stats['mean']:.2f}/20")
    print(f"Écart-type   : {stats['std']:.2f}")
    print(f"Q1 / Médiane / Q3 : {stats['25%']:.1f} / {stats['50%']:.1f} / {stats['75%']:.1f}")
    print(f"Min / Max    : {stats['min']:.1f} / {stats['max']:.1f}")
    print(f"Outliers (bas / haut) : {int(stats['nb_outliers_bas'])} / {int(stats['nb_outliers_haut'])}")

    outliers = resultats["outliers"]
    if len(outliers) > 0:
        print(f"\nÉlèves atypiques détectés ({len(outliers)}) :")
        print(outliers[["Notes", "Assiduité", "target"]].head(10).to_string(index=False))
    else:
        print("\nAucun outlier détecté selon la règle de Tukey.")

    print("\nFigures sauvegardées dans reports/figures/")
