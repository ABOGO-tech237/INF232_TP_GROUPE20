"""
Question 2 — Volet 1 : analyse bivariée Assiduité ↔ Notes.

Corrélation, covariance et visualisations pour vérifier le lien entre
l'assiduité et la note avant toute modélisation prédictive.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.chargement import RACINE_PROJET, charger_donnees_brutes, chemin_figures


def analyse_bivariate(df: pd.DataFrame) -> dict:
    """
    Caractérise le lien Assiduité ↔ Notes.

    Calcule Pearson, Spearman, covariance et exporte les graphiques
    d'exploration bivariée dans reports/figures/.
    """
    notes = df["Notes"]
    assiduite = df["Assiduité"]

    pearson = assiduite.corr(notes)
    spearman = assiduite.corr(notes, method="spearman")
    covariance = np.cov(assiduite, notes)[0, 1]

    stats_bivariees = pd.DataFrame({"Notes": notes.describe(), "Assiduité": assiduite.describe()})
    matrice_corr = df[["Notes", "Assiduité"]].corr()

    dossier_fig = chemin_figures()
    tracer_heatmap_correlation(matrice_corr, dossier_fig / "heatmap_correlation_bivariee.png")
    tracer_nuage_points(df, dossier_fig / "nuage_points_assiduite_notes.png", pearson)
    tracer_jointplot(df, dossier_fig / "jointplot_assiduite_notes.png")

    chemin_interim = RACINE_PROJET / "data" / "interim"
    chemin_interim.mkdir(parents=True, exist_ok=True)
    stats_bivariees.to_csv(chemin_interim / "stats_bivariees.csv")
    matrice_corr.to_csv(chemin_interim / "matrice_correlation.csv")

    return {
        "pearson": pearson,
        "spearman": spearman,
        "covariance": covariance,
        "stats_bivariees": stats_bivariees,
        "matrice_correlation": matrice_corr,
        "lien_verifie": abs(pearson) >= 0.5,
    }


def interpreter_lien_bivariate(pearson: float) -> str:
    """Message lisible pour le proviseur sur la force du lien."""
    r = abs(pearson)
    if r >= 0.8:
        force = "très forte"
    elif r >= 0.5:
        force = "modérée à forte"
    elif r >= 0.3:
        force = "modérée"
    else:
        force = "faible"

    verification = (
        "Le lien est bien vérifié dans les données."
        if r >= 0.5
        else "Le lien reste faible : l'assiduité seule ne suffit pas à anticiper la note."
    )
    return f"Corrélation r = {pearson:.3f} → relation {force}. {verification}"


def tracer_heatmap_correlation(matrice: pd.DataFrame, chemin_sortie: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrice, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Matrice de corrélation — Notes et Assiduité")
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def tracer_nuage_points(df: pd.DataFrame, chemin_sortie: Path, correlation: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=df, x="Assiduité", y="Notes",
        scatter_kws={"alpha": 0.5, "s": 25}, line_kws={"color": "red"}, ax=ax,
    )
    ax.set_title(f"Analyse bivariée (r = {correlation:.3f})")
    ax.set_xlabel("Assiduité (%)")
    ax.set_ylabel("Note (/20)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def tracer_jointplot(df: pd.DataFrame, chemin_sortie: Path) -> None:
    g = sns.jointplot(data=df, x="Assiduité", y="Notes", kind="scatter", alpha=0.4, height=7)
    g.figure.suptitle("Distribution conjointe Assiduité / Notes", y=1.02)
    g.figure.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def executer_analyse_bivariate() -> dict:
    """Point d'entrée volet 1 de la Question 2."""
    df = charger_donnees_brutes()
    bivarie = analyse_bivariate(df)
    return {
        "bivariate": bivarie,
        "message_lien": interpreter_lien_bivariate(bivarie["pearson"]),
        "correlation": bivarie["pearson"],
    }


if __name__ == "__main__":
    resultats = executer_analyse_bivariate()
    b = resultats["bivariate"]
    print("=== Question 2 — Volet 1 : Analyse bivariée ===")
    print(f"Pearson  : {b['pearson']:.4f}")
    print(f"Spearman : {b['spearman']:.4f}")
    print(f"Covariance : {b['covariance']:.4f}")
    print(resultats["message_lien"])
