"""
Question 2 — Orchestrateur (délègue aux deux volets séparés).

  Volet 1 : src/models/analyse_bivariate.py
  Volet 2 : src/models/comparaison_modeles_q2.py (régression linéaire)
"""

from src.models.analyse_bivariate import executer_analyse_bivariate
from src.models.comparaison_modeles_q2 import executer_regression_lineaire

# Ré-exports pour compatibilité notebook / imports existants
from src.models.analyse_bivariate import (  # noqa: F401
    analyse_bivariate,
    interpreter_lien_bivariate,
    tracer_nuage_points,
)
from src.models.comparaison_modeles_q2 import (  # noqa: F401
    executer_comparaison_modeles,
    preparer_jeux,
    rapport_ols,
    seuil_incertitude_acceptable,
    tracer_regression,
)


def analyser_correlation(df):
    return df["Assiduité"].corr(df["Notes"])


def entrainer_modele(df, random_state=42):
    """Compatibilité : régression linéaire + statsmodels OLS."""
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = preparer_jeux(df, random_state)
    modele = LinearRegression().fit(X_train, y_train)
    modele_sm = rapport_ols(df, random_state)
    return modele, modele_sm, X_train, X_test, y_train, y_test


def executer_analyse() -> dict:
    """Enchaîne volet 1 (bivariée) puis volet 2 (régression linéaire) — Question 2 complète."""
    q2_v1 = executer_analyse_bivariate()
    q2_v2 = executer_regression_lineaire()
    return {**q2_v1, **q2_v2}


if __name__ == "__main__":
    r = executer_analyse()
    b = r["bivariate"]
    print("=== Question 2 — Volet 1 : Analyse bivariée ===")
    print(f"Pearson : {b['pearson']:.4f} | Spearman : {b['spearman']:.4f}")
    print(r["message_lien"])
    print("\n=== Question 2 — Volet 2 : Régression linéaire ===")
    m = r["metriques"]
    print(f"RMSE = {m['rmse']:.3f} | R² = {m['r2']:.4f}")
    print(f"Équation : {r['regression_lineaire']['equation']}")
    print(f"\n>>> {r['modele_retenu']['justification']}")
    print(r["message_incertitude"])
