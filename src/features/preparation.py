"""
Transformations des variables pour la modélisation.

La standardisation est indispensable avant le clustering K-means :
sans elle, l'assiduité (0-100) dominerait les distances et masquerait
la contribution des notes (0-20).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def extraire_features_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement les deux variables explicatives du TP."""
    return df[["Notes", "Assiduité"]].copy()


def standardiser_features(features: pd.DataFrame):
    """
    Standardise Notes et Assiduité (moyenne 0, écart-type 1).

    Retourne le tableau transformé et le scaler entraîné
    (nécessaire pour revenir à l'échelle originale des centroïdes).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, scaler
