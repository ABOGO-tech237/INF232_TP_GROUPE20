import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# on charge depuis data/raw, ce script tourne depuis src/models/
df = pd.read_csv("../../data/raw/donnees.csv")

# le fichier peut avoir les colonnes X/Y (ancienne version) ou déjà
# Notes/Assiduité (nouvelle version) selon qui l'a régénéré, donc on gère les deux
if "X" in df.columns and "Y" in df.columns:
    df = df.rename(columns={"X": "Notes", "Y": "Assiduité"})

# on ne garde que Notes et Assiduité, pas target
# le but c'est justement de voir si des groupes ressortent sans regarder l'orientation déjà donnée
X_features = df[["Notes", "Assiduité"]]

# la note va de 0 à 20 et l'assiduité de 0 à 100, donc si on standardise pas
# l'assiduité va écraser la note dans le calcul des distances
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# on sait pas encore combien de groupes existent, donc on teste plusieurs K
# et on regarde ce qui donne les groupes les plus nets (coude + silhouette)
inerties = []
silhouettes = []
K_range = range(1, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    inerties.append(km.inertia_)
    if k > 1:  # le score de silhouette n'est défini que pour k >= 2
        silhouettes.append(silhouette_score(X_scaled, labels_k))
    else:
        silhouettes.append(None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(list(K_range), inerties, marker="o")
ax1.set_xlabel("Nombre de clusters (K)")
ax1.set_ylabel("Inertie intra-cluster")
ax1.set_title("Méthode du coude")
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_range)[1:], silhouettes[1:], marker="o", color="darkorange")
ax2.set_xlabel("Nombre de clusters (K)")
ax2.set_ylabel("Score de silhouette")
ax2.set_title("Score de silhouette selon K")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../../reports/figures/graphique_choix_nombre_clusters.png", dpi=150)
plt.close()

print("Inerties par K :", dict(zip(K_range, inerties)))
print("Silhouettes par K :", dict(zip(K_range, silhouettes)))

# d'après le graphique du coude, 3 semble être le bon compromis
K_OPTIMAL = 3

kmeans_final = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df["cluster"] = kmeans_final.fit_predict(X_scaled)

# on repasse les centroïdes à l'échelle réelle (note/20, assiduité en %)
# pour pouvoir les lire normalement au lieu des valeurs standardisées
centroides_reels = scaler.inverse_transform(kmeans_final.cluster_centers_)
resume = pd.DataFrame(
    centroides_reels, columns=["Note_moyenne", "Assiduite_moyenne"]
)
resume["effectif"] = df["cluster"].value_counts().sort_index().values
print("\nRésumé des clusters :\n", resume)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["Notes"], df["Assiduité"], c=df["cluster"], cmap="viridis", alpha=0.6, s=20
)
centres_x = centroides_reels[:, 0]
centres_y = centroides_reels[:, 1]
plt.scatter(
    centres_x, centres_y, c="red", marker="X", s=200,
    edgecolors="black", label="Centroïdes"
)
plt.xlabel("Note à l'évaluation (/20)")
plt.ylabel("Assiduité (%)")
plt.title(f"Clustering K-means des élèves (K={K_OPTIMAL})")
plt.legend()
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("../../reports/figures/visualisation_clusters_eleves.png", dpi=150)
plt.close()

df.to_csv("../../data/processed/donnees_eleves_avec_profil.csv", index=False)
resume.to_csv("../../reports/figures/profils_clusters_resume.csv", index=False)

print("\nFichiers générés : reports/figures/graphique_choix_nombre_clusters.png, reports/figures/visualisation_clusters_eleves.png,")
print("data/processed/donnees_eleves_avec_profil.csv, reports/figures/profils_clusters_resume.csv")
