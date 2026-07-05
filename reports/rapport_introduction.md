# Rapport — Introduction et génération des données

**Projet :** INF232 TP Groupe 20  
**Contexte :** Analyse des performances d'élèves de terminale

---

## 1. Contexte du proviseur

La direction d'un établissement scolaire souhaite mieux comprendre les performances de ses élèves de terminale à partir :

- d'une **note** à une évaluation interne de mathématiques (/20) ;
- d'un **taux d'assiduité** aux cours de maths (%) ;
- d'une **orientation** déjà recommandée par le conseil de classe (littéraire ou scientifique).

Ce TP répond aux trois premières questions du proviseur par des analyses statistiques reproductibles.

---

## 2. Choix des variables

| Variable | Symbole | Justification scolaire |
|----------|---------|------------------------|
| Note d'évaluation | **Notes (X)** | Mesure directe de la performance en mathématiques |
| Taux de présence | **Assiduité (Y)** | Indicateur de régularité et de travail personnel |
| Orientation | **target** | Décision du conseil de classe (0 = littéraire, 1 = scientifique) |

La relation imposée lors de la génération (`Y ≈ 5 × X`) reflète l'intuition pédagogique : un élève assidu tend à obtenir une meilleure note.

---

## 3. Génération des données synthétiques

**Fichiers concernés :**
- `src/utils/graine.py` — calcul de la graine
- `src/data/generateur.py` — génération des variables
- `main.py` — orchestration

### 3.1 Graine reproductible (SHA-256)

```python
# src/utils/graine.py
def generer_chaine(texte: str) -> int:
    texte_normalise = nettoyer_texte(texte)  # majuscules, sans accents
    hash_hex = hashlib.sha256(texte_normalise.encode()).hexdigest()
    return int(hash_hex, 16) % (2**12)
```

Pour le chef de groupe **« Atangana abog Emmanue »**, la graine obtenue est **`1545`**.  
Toute exécution avec ce nom produit le même jeu de 1000 élèves.

### 3.2 Génération de X et Y

```python
# src/data/generateur.py
X = uniforme(0, 20)           # Notes arrondies à 0,1
Y = clip(5 * X + bruit, 0, 100) # Assiduité avec bruit gaussien (σ=5)
```

Le bruit évite une corrélation parfaite et introduit des profils atypiques réalistes (ex. : bonne assiduité mais note faible).

### 3.3 Variable cible (orientation)

```python
target = 1 si Assiduité >= 50 % else 0
```

Seuil à 50 % cohérent avec `Y ≈ 5X` : une assiduité moyenne correspond à une note d'environ 10/20.

### 3.4 Jeu de données produit

| Indicateur | Valeur |
|------------|--------|
| Nombre d'élèves | 1000 |
| Orientation scientifique (1) | 531 (53,1 %) |
| Orientation littéraire (0) | 469 (46,9 %) |
| Fichier | `data/raw/donnees.csv` |

> **Règle du projet :** le fichier `data/raw/donnees.csv` ne doit pas être modifié après génération.

---

## 4. Organisation du code

Le projet suit une architecture modulaire :

| Dossier | Rôle |
|---------|------|
| `src/utils/` | Fonctions transverses (graine) |
| `src/data/` | Génération et chargement |
| `src/features/` | Transformations (standardisation) |
| `src/models/` | Une question = un ou plusieurs scripts |
| `notebooks/` | Explorations interactives |
| `reports/figures/` | Graphiques pour les rapports |

---

## 5. Questions traitées

| # | Question du proviseur | Script principal | Rapport |
|---|----------------------|------------------|---------|
| 1 | Répartition des notes, outliers | `analyse_univariee.py` | [rapport_question1](rapport_question1_analyse_univariee.md) |
| 2 | Lien assiduité/note, prédiction | `analyse_bivariate.py` + `comparaison_modeles_q2.py` | [rapport_question2](rapport_question2_regression.md) |
| 3 | Profils d'élèves | `clustering_profils_eleves.py` | [rapport_question3](rapport_question3_clustering.md) |

---

## 6. Reproductibilité

Pour reproduire l'intégralité des analyses :

```bash
python main.py
```

Les figures sont régénérées dans `reports/figures/` et les CSV intermédiaires dans `data/interim/`.
