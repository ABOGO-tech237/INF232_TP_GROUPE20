# INF232 — TP Groupe 20 : Analyse des performances et profils d'élèves

Projet de travaux pratiques en statistique et apprentissage automatique appliqué à un contexte scolaire : comprendre la répartition des notes, le lien entre assiduité et performance, et identifier des profils d'élèves de terminale.

**Chef de groupe :** Atangana abog Emmanue  
**Graine reproductible :** `1545` (dérivée du nom via SHA-256)

---

## Contexte pédagogique

Un établissement secondaire dispose pour chaque élève de terminale :

| Variable | Description | Échelle |
|----------|-------------|---------|
| **Notes** | Résultat à l'évaluation interne de mathématiques | /20 |
| **Assiduité** | Taux d'assiduité aux cours de maths | 0–100 % |
| **target** | Orientation recommandée (0 = littéraire, 1 = scientifique) | binaire |

Le proviseur pose trois questions statistiques, traitées dans l'ordre :

1. **Question 1** — Comment se répartissent les notes ? Y a-t-il des élèves atypiques ?
2. **Question 2** — L'assiduité est-elle liée à la note ? Peut-on anticiper la note ?
3. **Question 3** — Des groupes de profils ressortent-ils naturellement des données ?

---

## Structure du projet

```
INF232_TP_GROUPE20/
├── data/
│   ├── raw/              # Données brutes (donnees.csv — ne pas modifier)
│   ├── interim/          # Statistiques intermédiaires (CSV)
│   └── processed/        # Données avec clusters assignés
├── notebooks/            # Analyses interactives (Jupyter)
│   ├── question1_analyse_univariee.ipynb
│   ├── question2_regression_lineaire.ipynb
│   └── question3_clustering_profils.ipynb
├── src/
│   ├── utils/graine.py           # Graine SHA-256 reproductible
│   ├── data/
│   │   ├── generateur.py         # Génération synthétique X, Y, target
│   │   └── chargement.py         # Chargement centralisé des CSV
│   ├── features/preparation.py   # Standardisation (clustering)
│   └── models/
│       ├── analyse_univariee.py       # Question 1
│       ├── analyse_bivariate.py       # Question 2 — volet 1
│       ├── comparaison_modeles_q2.py  # Question 2 — volet 2
│       ├── LinearRegression.py        # Orchestrateur Q2
│       └── clustering_profils_eleves.py  # Question 3 (K-Means)
├── reports/
│   ├── figures/          # Graphiques PNG
│   ├── rapport_introduction.md
│   ├── rapport_question1_analyse_univariee.md
│   ├── rapport_question2_regression.md
│   └── rapport_question3_clustering.md
└── main.py               # Pipeline complet
```

---

## Installation et exécution

### Prérequis

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

### Lancer tout le pipeline

```bash
python main.py
```

### Lancer une question individuellement

```bash
# Question 1
python src/models/analyse_univariee.py

# Question 2 — volet 1 (bivariée)
python src/models/analyse_bivariate.py

# Question 2 — volet 2 (modèles prédictifs)
python src/models/comparaison_modeles_q2.py

# Question 2 — complète
python src/models/LinearRegression.py

# Question 3
python src/models/clustering_profils_eleves.py
```

### Notebooks Jupyter

Ouvrir les fichiers dans `notebooks/` depuis la racine du projet (le kernel doit voir le dossier `src/`).

---

## Flux de données

```
main.py
  │
  ├─► src/data/generateur.py  ──► data/raw/donnees.csv
  │
  ├─► analyse_univariee.py    ──► reports/figures/boxplot_*.png
  │                               data/interim/stats_descriptives_notes.csv
  │
  ├─► analyse_bivariate.py    ──► reports/figures/heatmap_*.png, nuage_*.png
  │   comparaison_modeles_q2.py   data/interim/comparaison_modeles_q2.csv
  │
  └─► clustering_profils_eleves.py ──► data/processed/donnees_eleves_avec_profil.csv
                                       reports/figures/visualisation_clusters_eleves.png
```

---

## Résultats principaux (synthèse)

| Question | Résultat clé |
|----------|--------------|
| **Q1** | Moyenne 10,21/20, médiane 10,6 — distribution symétrique, 0 outlier (Tukey) |
| **Q2** | Corrélation r = 0,987 — lien très fort ; meilleur modèle : SVR (RMSE = 0,95 pt) |
| **Q3** | 3 profils K-Means : fragile (347), intermédiaire (317), solide (336) |

---

## Rapports détaillés

Consultez le dossier [`reports/`](reports/) pour les rapports complets :

- [Index des rapports](reports/README.md)
- [Introduction et génération des données](reports/rapport_introduction.md)
- [Question 1 — Analyse univariée](reports/rapport_question1_analyse_univariee.md)
- [Question 2 — Analyse bivariée et régression](reports/rapport_question2_regression.md)
- [Question 3 — Clustering K-Means](reports/rapport_question3_clustering.md)
- [Synthèse pour le proviseur](reports/rapport_synthese_proviseur.md)
- **[Rapport global LaTeX (PDF)](reports/rapport_global.pdf)** — document complet compilé

---

## Équipe

INF232 — Groupe 20
