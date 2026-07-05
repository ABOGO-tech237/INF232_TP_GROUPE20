"""
Question 3 — Classification non supervisée (clustering K-Means).

Identifie des profils d'élèves à partir de Notes et Assiduité uniquement,
sans utiliser l'orientation déjà recommandée (target).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.chargement import (
    RACINE_PROJET,
    charger_donnees_brutes,
    chemin_figures,
)
from src.features.preparation import extraire_features_numeriques, standardiser_features


def evaluer_nombre_clusters(X_scaled, k_max: int = 8) -> tuple[list, list]:
    """
    Méthode du coude + score de silhouette pour choisir K.

    L'inertie diminue quand K augmente ; le « coude » indique le compromis.
    La silhouette mesure la cohésion / séparation des groupes (proche de 1 = net).
    """
    inerties = []
    silhouettes = []
    k_range = range(1, k_max + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km.fit_predict(X_scaled)
        inerties.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X_scaled, labels_k))
        else:
            silhouettes.append(None)

    return list(k_range), inerties, silhouettes


def tracer_choix_k(k_range, inerties, silhouettes, chemin_sortie: Path) -> None:
    """Graphiques d'aide au choix du nombre de clusters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_range, inerties, marker="o")
    ax1.set_xlabel("Nombre de clusters (K)")
    ax1.set_ylabel("Inertie intra-cluster")
    ax1.set_title("Méthode du coude")
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range[1:], silhouettes[1:], marker="o", color="darkorange")
    ax2.set_xlabel("Nombre de clusters (K)")
    ax2.set_ylabel("Score de silhouette")
    ax2.set_title("Score de silhouette selon K")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150)
    plt.close()


def decrire_clusters(centroides_reels: pd.DataFrame) -> pd.DataFrame:
    """Attribue fragile / intermédiaire / solide selon la note moyenne du cluster."""
    centroides_reels = centroides_reels.copy().sort_values("Note_moyenne").reset_index(drop=True)
    libelles = ["Profil fragile : notes basses et/ou faible assiduité — accompagnement renforcé",
                "Profil intermédiaire : résultats et assiduité modérés",
                "Profil solide : bonnes notes et forte assiduité"]
    if len(centroides_reels) == 3:
        centroides_reels["description_pedagogique"] = libelles
    else:
        descriptions = []
        for _, row in centroides_reels.iterrows():
            note, assid = row["Note_moyenne"], row["Assiduite_moyenne"]
            if note >= 14 and assid >= 70:
                descriptions.append(libelles[2])
            elif note >= 8 and assid >= 40:
                descriptions.append(libelles[1])
            else:
                descriptions.append(libelles[0])
        centroides_reels["description_pedagogique"] = descriptions
    return centroides_reels


def executer_clustering(k_optimal: int = 3) -> dict:
    """Pipeline complet Question 3 — K-Means uniquement."""
    df = charger_donnees_brutes()
    features = extraire_features_numeriques(df)
    X_scaled, scaler = standardiser_features(features)

    k_range, inerties, silhouettes = evaluer_nombre_clusters(X_scaled)
    dossier_fig = chemin_figures()

    tracer_choix_k(
        k_range,
        inerties,
        silhouettes,
        dossier_fig / "graphique_choix_nombre_clusters.png",
    )

    # K=3 : compromis visible sur le coude et bon score de silhouette
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    centroides_reels = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=["Note_moyenne", "Assiduite_moyenne"],
    )
    centroides_reels["effectif"] = df["cluster"].value_counts().sort_index().values
    resume = decrire_clusters(centroides_reels)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["Notes"],
        df["Assiduité"],
        c=df["cluster"],
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    ax.scatter(
        resume["Note_moyenne"],
        resume["Assiduite_moyenne"],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        label="Centroïdes",
    )
    ax.set_xlabel("Note à l'évaluation (/20)")
    ax.set_ylabel("Assiduité (%)")
    ax.set_title(f"Clustering K-Means des élèves (K={k_optimal})")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(dossier_fig / "visualisation_clusters_eleves.png", dpi=150)
    plt.close()

    chemin_interim = RACINE_PROJET / "data" / "interim"
    chemin_interim.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "K": k_range,
        "inertie": inerties,
        "silhouette": silhouettes,
    }).to_csv(chemin_interim / "choix_clusters_q3.csv", index=False)

    chemin_processed = RACINE_PROJET / "data" / "processed"
    chemin_processed.mkdir(parents=True, exist_ok=True)
    df.to_csv(chemin_processed / "donnees_eleves_avec_profil.csv", index=False)
    resume.to_csv(dossier_fig / "profils_clusters_resume.csv", index=False)

    return {
        "k_optimal": k_optimal,
        "inerties": dict(zip(k_range, inerties)),
        "silhouettes": dict(zip(k_range, silhouettes)),
        "resume_clusters": resume,
        "donnees_avec_cluster": df,
    }


if __name__ == "__main__":
    resultats = executer_clustering()

    print("=== Question 3 — Clustering K-Means ===")
    print(f"Nombre de profils retenus : K = {resultats['k_optimal']}")
    print("\nRésumé des clusters :")
    print(resultats["resume_clusters"].to_string(index=False))
    print("\nFigures sauvegardées dans reports/figures/ et data/processed/")
