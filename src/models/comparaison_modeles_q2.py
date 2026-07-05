"""
Question 2 — Volet 2 : régression linéaire.

Prédit la note à partir de l'assiduité avec une régression linéaire
(conformément au sujet du TP, Section 5).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.chargement import RACINE_PROJET, charger_donnees_brutes, chemin_figures


def preparer_jeux(df: pd.DataFrame, random_state: int = 42):
    X = df[["Assiduité"]]
    y = df["Notes"]
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def evaluer_regression_lineaire(modele, X_test, y_test) -> dict:
    y_pred = modele.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def equation_regression_lineaire(modele) -> dict:
    a = float(modele.coef_[0])
    b = float(modele.intercept_)
    return {
        "coefficient": a,
        "intercept": b,
        "equation": f"Notes ≈ {a:.3f} × Assiduité + {b:.3f}",
    }


def seuil_incertitude_acceptable(rmse: float, tolerance: float = 2.0) -> str:
    if rmse <= tolerance:
        return (
            f"RMSE = {rmse:.2f} pts (< {tolerance}) : anticipation utilisable avec prudence."
        )
    return (
        f"RMSE = {rmse:.2f} pts (> {tolerance}) : anticipation trop incertaine pour une décision seule."
    )


def tracer_regression(modele, X_train, y_train, X_test, y_test, chemin_sortie: Path, titre: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_train, y_train, alpha=0.5, label="Entraînement", color="steelblue")
    ax.scatter(X_test, y_test, alpha=0.7, label="Test", color="seagreen", marker="s")
    x_ligne = pd.DataFrame({"Assiduité": np.linspace(0, 100, 200)})
    ax.plot(x_ligne, modele.predict(x_ligne), color="red", linewidth=2, label="Droite de régression")
    ax.set_xlabel("Assiduité (%)")
    ax.set_ylabel("Note (/20)")
    ax.set_title(titre)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 22)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def rapport_ols(df: pd.DataFrame, random_state: int = 42):
    """Rapport statsmodels pour la régression linéaire."""
    X_train, _, y_train, _ = preparer_jeux(df, random_state)
    return sm.OLS(y_train, sm.add_constant(X_train)).fit()


def executer_regression_lineaire() -> dict:
    """Point d'entrée volet 2 de la Question 2 — régression linéaire uniquement."""
    df = charger_donnees_brutes()
    X_train, X_test, y_train, y_test = preparer_jeux(df)

    modele = LinearRegression()
    modele.fit(X_train, y_train)
    metriques = evaluer_regression_lineaire(modele, X_test, y_test)
    eq = equation_regression_lineaire(modele)

    dossier_fig = chemin_figures()
    tracer_regression(
        modele, X_train, y_train, X_test, y_test,
        dossier_fig / "regression_lineaire_notes_assiduite.png",
        titre="Régression linéaire : Notes ~ Assiduité",
    )

    chemin_interim = RACINE_PROJET / "data" / "interim"
    chemin_interim.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "modele": "Régression linéaire",
        "rmse_test": metriques["rmse"],
        "mae_test": metriques["mae"],
        "r2_test": metriques["r2"],
        "coefficient_assiduite": eq["coefficient"],
        "intercept": eq["intercept"],
        "equation": eq["equation"],
    }]).to_csv(chemin_interim / "regression_lineaire_q2.csv", index=False)

    modele_retenu = {
        "nom": "Régression linéaire",
        "modele": modele,
        "rmse": metriques["rmse"],
        "mae": metriques["mae"],
        "r2": metriques["r2"],
        "justification": (
            f"Modèle retenu : Régression linéaire "
            f"(RMSE = {metriques['rmse']:.3f} pts, R² = {metriques['r2']:.4f})"
        ),
    }

    return {
        "modele_retenu": modele_retenu,
        "meilleur_modele": modele_retenu,
        "regression_lineaire": eq,
        "metriques": metriques,
        "rapport_statsmodels": rapport_ols(df).summary(),
        "message_incertitude": seuil_incertitude_acceptable(metriques["rmse"]),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# Alias pour compatibilité avec main.py et LinearRegression.py
executer_comparaison_modeles = executer_regression_lineaire


if __name__ == "__main__":
    resultats = executer_regression_lineaire()
    m = resultats["metriques"]
    print("=== Question 2 — Régression linéaire ===")
    print(f"RMSE = {m['rmse']:.3f} | MAE = {m['mae']:.3f} | R² = {m['r2']:.4f}")
    print(f"Équation : {resultats['regression_lineaire']['equation']}")
    print(f"\n>>> {resultats['modele_retenu']['justification']}")
    print(resultats["message_incertitude"])
