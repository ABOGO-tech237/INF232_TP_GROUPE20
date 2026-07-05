# Index des rapports — INF232 TP Groupe 20

Ce dossier contient la documentation détaillée du projet : explication du code, méthodes statistiques et interprétation des résultats.

---

## Rapports disponibles

| Rapport | Contenu | Script(s) associé(s) |
|---------|---------|----------------------|
| [Introduction](rapport_introduction.md) | Contexte, génération des données, graine SHA-256 | `main.py`, `src/data/generateur.py` |
| [Question 1 — Analyse univariée](rapport_question1_analyse_univariee.md) | Distribution des notes, outliers, boîte à moustaches | `src/models/analyse_univariee.py` |
| [Question 2 — Bivariée et régression](rapport_question2_regression.md) | Corrélation, 3 modèles prédictifs, RMSE | `analyse_bivariate.py`, `comparaison_modeles_q2.py` |
| [Question 3 — Clustering K-Means](rapport_question3_clustering.md) | 3 profils d'élèves, choix de K | `clustering_profils_eleves.py` |
| [Synthèse proviseur](rapport_synthese_proviseur.md) | Réponses condensées aux 3 questions | — |
| **[Rapport global LaTeX](rapport_global.tex)** | **Document PDF complet** (toutes sections + figures) | — |

### Compiler le rapport PDF

```bash
cd reports
pdflatex rapport_global.tex
pdflatex rapport_global.tex   # 2e passe pour la table des matières
```

---

## Figures

Toutes les figures référencées dans les rapports se trouvent dans [`figures/`](figures/) :

```
figures/
├── boxplot_notes_evaluation.png          # Q1
├── histogramme_notes_evaluation.png      # Q1
├── heatmap_correlation_bivariee.png      # Q2 volet 1
├── nuage_points_assiduite_notes.png        # Q2 volet 1
├── jointplot_assiduite_notes.png           # Q2 volet 1
├── comparaison_trois_modeles_q2.png        # Q2 volet 2
├── regression_meilleur_modele_q2.png       # Q2 volet 2
├── graphique_choix_nombre_clusters.png     # Q3
├── visualisation_clusters_eleves.png       # Q3
└── profils_clusters_resume.csv             # Q3 — tableau des profils
```

---

## Données intermédiaires

Les CSV de traçabilité sont dans `data/interim/` :

- `stats_descriptives_notes.csv` — Q1
- `stats_bivariees.csv`, `matrice_correlation.csv` — Q2 volet 1
- `comparaison_modeles_q2.csv` — Q2 volet 2

Le jeu enrichi avec clusters : `data/processed/donnees_eleves_avec_profil.csv`
