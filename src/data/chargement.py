"""
Chargement centralisé des jeux de données du projet.
"""

from pathlib import Path

import pandas as pd

# Racine du dépôt : remonte de src/data/ vers la racine
RACINE_PROJET = Path(__file__).resolve().parents[2]


def chemin_donnees_brutes() -> Path:
    return RACINE_PROJET / "data" / "raw" / "donnees.csv"


def chemin_donnees_traitees() -> Path:
    return RACINE_PROJET / "data" / "processed" / "donnees_eleves_avec_profil.csv"


def charger_donnees_brutes() -> pd.DataFrame:
    """
    Charge le CSV brut et harmonise les noms de colonnes.

    Gère l'ancien format (X, Y) et le format actuel (Notes, Assiduité).
    """
    df = pd.read_csv(chemin_donnees_brutes())

    if "X" in df.columns and "Y" in df.columns:
        df = df.rename(columns={"X": "Notes", "Y": "Assiduité"})

    return df


def chemin_figures() -> Path:
    """Dossier de sortie des graphiques pour le rapport."""
    dossier = RACINE_PROJET / "reports" / "figures"
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier
