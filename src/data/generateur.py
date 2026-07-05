"""
Génération du jeu de données synthétiques du TP.

Variables :
  - Notes (X)     : note à l'évaluation interne, sur 20
  - Assiduité (Y) : taux de présence aux cours de maths, en %
  - target        : orientation recommandée (0 = littéraire, 1 = scientifique)

Relation imposée : Y ≈ 5 * X + bruit gaussien, pour simuler le lien
assiduité / performance tout en gardant des valeurs atypiques réalistes.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def data_genertor(graine: int, nb_echantillons: int = 1000, bruit: float = 5.0):
    """
    Génère les vecteurs X (notes) et Y (assiduité).

    Le bruit gaussien évite une corrélation parfaite et introduit
    des profils atypiques (forte assiduité / note faible, etc.).
    """
    rng = np.random.default_rng(graine)

    # Notes arrondies à 0,1 pour rester lisibles dans les rapports
    X = np.round(rng.uniform(0, 20, size=nb_echantillons), 1)
    y = np.round(5 * X + rng.normal(0, bruit, size=nb_echantillons), 0)
    y = np.clip(y, 0, 100)

    return X, y


def vers_jeu_de_donnees(X, y, seuil: float = 50.0) -> pd.DataFrame:
    """
    Assemble le DataFrame final avec la cible binaire.

    Le seuil à 50 % d'assiduité sert de proxy pour l'orientation :
    au-dessus → filière scientifique (1), en dessous → littéraire (0).
    Ce choix est cohérent avec Y ≈ 5*X : un élève assidu tend à avoir
    une meilleure note, donc une orientation scientifique plus probable.
    """
    df = pd.DataFrame({"Notes": X, "Assiduité": y})
    df["target"] = (df["Assiduité"] >= seuil).astype(int)
    return df


def exporter_csv(df: pd.DataFrame, chemin: str | Path = "data/raw/donnees.csv") -> Path:
    """Persiste le jeu de données brut ; ce fichier ne doit pas être modifié ensuite."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(chemin, index=False)
    return chemin
