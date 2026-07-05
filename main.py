"""
Point d'entrée principal du TP INF232 — Groupe 20.

Enchaîne les étapes du flux de données :
  1. Génération des données synthétiques (graine SHA-256)
  2. Question 1 — Analyse univariée des notes
  3. Question 2 — Régression linéaire (note ~ assiduité)
  4. Question 3 — Clustering K-means des profils d'élèves
  5. Question 4 — Classification supervisée de l'orientation
"""

from pathlib import Path
import sys

# Ajoute la racine du projet au PYTHONPATH pour les imports src.*
RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE))

from src.utils.graine import generer_chaine
from src.data.generateur import data_genertor, vers_jeu_de_donnees, exporter_csv
from src.models.analyse_univariee import executer_analyse as analyse_q1
from src.models.LinearRegression import executer_analyse as analyse_q2
from src.models.clustering_profils_eleves import executer_clustering as analyse_q3
from src.models.classification_orientation_q4 import executer_classification as analyse_q4

# Nom du chef de groupe — sert à produire la graine reproductible du TP
NOM_CHEF_GROUPE = "Atangana Abogo Emmanuel"


def generer_donnees() -> Path:
    """Étape 1 : création du fichier data/raw/donnees.csv (données jamais modifiées ensuite)."""
    graine = generer_chaine(NOM_CHEF_GROUPE)
    X, y = data_genertor(graine=graine)
    jeu = vers_jeu_de_donnees(X, y)
    chemin = exporter_csv(jeu, RACINE / "data" / "raw" / "donnees.csv")

    print(f"Graine utilisée : {graine}")
    print(f"Fichier exporté : {chemin}")
    print(f"Nombre d'élèves : {len(jeu)}")
    print(f"Répartition orientation (0=littéraire, 1=scientifique) : {jeu['target'].value_counts().to_dict()}\n")

    return chemin


def main():
    print("=" * 60)
    print("TP INF232 — Analyse des performances et profils d'élèves")
    print("=" * 60 + "\n")

    print("--- Étape 1 : Génération des données ---")
    generer_donnees()

    print("--- Question 1 : Distribution des notes ---")
    q1 = analyse_q1()
    stats = q1["statistiques"]
    print(f"  Moyenne {stats['mean']:.2f}/20 | Médiane {stats['50%']:.1f} | "
          f"Outliers {int(stats['nb_outliers_bas'])+int(stats['nb_outliers_haut'])}")

    print("\n--- Question 2 : Lien assiduité / notes ---")
    q2 = analyse_q2()
    b = q2["bivariate"]
    print(f"  Bivariée : Pearson r = {b['pearson']:.3f} | Spearman = {b['spearman']:.3f}")
    print(f"  {q2['message_lien']}")
    m = q2["metriques"]
    print(f"  Régression linéaire : RMSE {m['rmse']:.3f} | R² {m['r2']:.3f}")
    print(f"  Équation : {q2['regression_lineaire']['equation']}")
    print(f"  {q2['message_incertitude']}")

    print("\n--- Question 3 : Profils d'élèves (K-Means) ---")
    q3 = analyse_q3()
    print(f"  {q3['k_optimal']} profils distincts identifiés")
    print(q3["resume_clusters"][["Note_moyenne", "Assiduite_moyenne", "effectif", "description_pedagogique"]].to_string(index=False))

    print("\n--- Question 4 : Suggestion automatique d'orientation ---")
    q4 = analyse_q4()
    print(f"  Meilleur classifieur (F1) : {q4['meilleur_modele']['nom']}")
    print(f"  Précision = {q4['meilleur_modele']['accuracy']:.1%} | F1 = {q4['meilleur_modele']['f1']:.3f}")
    cv = q4["validation_croisee"]
    print(f"  Validation croisée (log. régression) : F1 = {cv['f1_moyen']:.3f} ± {cv['f1_ecart']:.3f}")
    print(f"  Matrice de confusion (arbre) :\n{q4['matrice_arbre']}")
    print(f"  {q4['message_confiance']}")

    print("\n" + "=" * 60)
    print("Pipeline terminé. Consultez reports/figures/ pour les graphiques.")
    print("=" * 60)


if __name__ == "__main__":
    main()
