"""
Question 4 — Classification supervisée pour suggérer une orientation.

Compare régression logistique, arbre de décision et KNN.
Produit : métriques détaillées, matrice de confusion, courbes ROC,
frontière de décision et validation croisée.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.chargement import RACINE_PROJET, charger_donnees_brutes, chemin_figures


def preparer_jeux(df: pd.DataFrame, random_state: int = 42):
    X = df[["Notes", "Assiduité"]]
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


def construire_classifieurs() -> dict:
    """Trois classifieurs comparés dans le rapport (Q4)."""
    return {
        "Régression logistique": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Arbre de décision": DecisionTreeClassifier(max_depth=4, random_state=42),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


def _scores_proba(modele, X):
    if hasattr(modele, "predict_proba"):
        return modele.predict_proba(X)[:, 1]
    if hasattr(modele, "decision_function"):
        scores = modele.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return modele.predict(X).astype(float)


def evaluer_classifieur(modele, X_test, y_test) -> dict:
    y_pred = modele.predict(X_test)
    y_score = _scores_proba(modele, X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_score),
        "matrice_confusion": confusion_matrix(y_test, y_pred),
        "rapport": classification_report(
            y_test, y_pred, target_names=["Littéraire", "Scientifique"], zero_division=0
        ),
        "y_pred": y_pred,
        "y_score": y_score,
    }


def comparer_classifieurs(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    resultats = []
    for nom, modele in construire_classifieurs().items():
        modele.fit(X_train, y_train)
        metriques = evaluer_classifieur(modele, X_test, y_test)
        resultats.append({
            "modele": nom,
            "accuracy_test": metriques["accuracy"],
            "precision_test": metriques["precision"],
            "recall_test": metriques["recall"],
            "f1_test": metriques["f1"],
            "auc_test": metriques["auc"],
            "modele_entraine": modele,
            "matrice_confusion": metriques["matrice_confusion"],
            "rapport": metriques["rapport"],
            "y_score": metriques["y_score"],
        })
    comparaison = pd.DataFrame(resultats).sort_values("f1_test", ascending=False).reset_index(drop=True)
    comparaison["rang"] = comparaison.index + 1
    return comparaison


def validation_croisee_logistique(X, y, cv: int = 5) -> dict:
    modele = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    scores = cross_val_score(modele, X, y, cv=cv, scoring="f1")
    return {"f1_moyen": scores.mean(), "f1_ecart": scores.std(), "scores": scores}


def interpreter_confiance(accuracy: float) -> str:
    pct = accuracy * 100
    if accuracy >= 0.95:
        return (
            f"Précision globale = {pct:.1f}% : scores élevés sur nos données, "
            f"mais à relativiser (cible construite par un seuil d'assiduité)."
        )
    if accuracy >= 0.80:
        return (
            f"Précision globale = {pct:.1f}% : utilisable comme aide à la décision, "
            f"jamais comme décision finale sans le conseil de classe."
        )
    return (
        f"Précision globale = {pct:.1f}% : trop faible pour automatiser l'orientation "
        f"sans supervision humaine."
    )


def risques_pedagogiques() -> list[str]:
    return [
        "Performances élevées liées au seuil artificiel target = (Assiduité >= 50%).",
        "L'orientation réelle repose sur bien plus que deux variables.",
        "Risque d'automatisation excessive (automation bias) si le score est pris pour une certitude.",
        "Le modèle peut reproduire des biais historiques du conseil de classe.",
    ]


def tracer_matrice_confusion(matrice: np.ndarray, chemin_sortie: Path, titre: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Littéraire", "Scientifique"]
    im = ax.imshow(matrice, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité (conseil de classe)")
    ax.set_title(titre)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrice[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def tracer_courbes_roc(comparaison: pd.DataFrame, y_test, chemin_sortie: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in comparaison.iterrows():
        fpr, tpr, _ = roc_curve(y_test, row["y_score"])
        ax.plot(fpr, tpr, label=f"{row['modele']} (AUC={row['auc_test']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC — Question 4")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def tracer_frontiere_decision(modele, df: pd.DataFrame, chemin_sortie: Path) -> None:
    """Frontière apprise par la régression logistique (modèle du rapport)."""
    x_min, x_max = df["Notes"].min() - 1, df["Notes"].max() + 1
    y_min, y_max = df["Assiduité"].min() - 5, df["Assiduité"].max() + 5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grille = pd.DataFrame({"Notes": xx.ravel(), "Assiduité": yy.ravel()})
    Z = modele.predict(grille).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    scatter = ax.scatter(
        df["Notes"], df["Assiduité"], c=df["target"], cmap="coolwarm", alpha=0.5, s=15, edgecolors="none"
    )
    ax.set_xlabel("Note à l'évaluation (/20)")
    ax.set_ylabel("Assiduité (%)")
    ax.set_title("Frontière de décision — Régression logistique")
    plt.colorbar(scatter, ax=ax, label="Orientation (0=litt., 1=sci.)")
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def executer_classification() -> dict:
    df = charger_donnees_brutes()
    X = df[["Notes", "Assiduité"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = preparer_jeux(df)
    comparaison = comparer_classifieurs(X_train, X_test, y_train, y_test)
    cv = validation_croisee_logistique(X, y)

    meilleur_f1 = comparaison.iloc[0]
    modele_arbre = comparaison.loc[comparaison["modele"] == "Arbre de décision"].iloc[0]
    modele_log = comparaison.loc[comparaison["modele"] == "Régression logistique"].iloc[0]

    dossier_fig = chemin_figures()
    tracer_matrice_confusion(
        modele_arbre["matrice_confusion"],
        dossier_fig / "matrice_confusion_q4.png",
        titre="Matrice de confusion — Arbre de décision",
    )
    tracer_courbes_roc(comparaison, y_test, dossier_fig / "courbe_roc_q4.png")
    tracer_frontiere_decision(modele_log["modele_entraine"], df, dossier_fig / "frontiere_decision_q4.png")

    chemin_interim = RACINE_PROJET / "data" / "interim"
    comparaison[
        ["modele", "accuracy_test", "precision_test", "recall_test", "f1_test", "auc_test", "rang"]
    ].to_csv(chemin_interim / "comparaison_classifieurs_q4.csv", index=False)

    return {
        "comparaison_classifieurs": comparaison,
        "meilleur_modele": {
            "nom": meilleur_f1["modele"],
            "accuracy": meilleur_f1["accuracy_test"],
            "f1": meilleur_f1["f1_test"],
            "matrice_confusion": meilleur_f1["matrice_confusion"],
        },
        "matrice_arbre": modele_arbre["matrice_confusion"],
        "validation_croisee": cv,
        "message_confiance": interpreter_confiance(meilleur_f1["accuracy_test"]),
        "risques_pedagogiques": risques_pedagogiques(),
        "effectifs": {
            "litteraire": int((df["target"] == 0).sum()),
            "scientifique": int((df["target"] == 1).sum()),
        },
    }


if __name__ == "__main__":
    resultats = executer_classification()
    print("=== Question 4 — Classification de l'orientation ===")
    print(resultats["comparaison_classifieurs"][
        ["rang", "modele", "accuracy_test", "precision_test", "recall_test", "f1_test", "auc_test"]
    ].to_string(index=False))
    print(f"\n>>> Meilleur modèle (F1) : {resultats['meilleur_modele']['nom']}")
    cv = resultats["validation_croisee"]
    print(f"Validation croisée (log. régression, 5 blocs) : F1 = {cv['f1_moyen']:.3f} ± {cv['f1_ecart']:.3f}")
    print(f"\nMatrice de confusion (arbre) :\n{resultats['matrice_arbre']}")
    print(f"\n{resultats['message_confiance']}")
