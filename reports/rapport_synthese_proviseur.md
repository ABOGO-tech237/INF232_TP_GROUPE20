# Synthèse — Réponses au proviseur

Document de synthèse à destination du conseil pédagogique.  
Les détails techniques et le code sont dans les rapports par question.

---

## Question 1 — Répartition des notes

**Question :** *Comment se répartissent les résultats ? Y a-t-il des élèves atypiques ?*

| Élément | Résultat |
|---------|----------|
| Moyenne | 10,21 /20 |
| Médiane | 10,6 /20 |
| Dispersion (écart-type) | 5,86 |
| Outliers détectés | **Aucun** (règle de Tukey) |

**Réponse :** Les notes couvrent l'échelle 0–20 de façon relativement uniforme. La moitié des élèves sont au-dessus de 10,6/20. Aucun profil statistiquement extrême ne ressort, mais des notes très basses (0) ou très hautes (20) existent et méritent un suivi individualisé.

**Support de présentation recommandé :** boîte à moustaches (`figures/boxplot_notes_evaluation.png`)

→ [Rapport détaillé Q1](rapport_question1_analyse_univariee.md)

---

## Question 2 — Lien assiduité / note et prédiction

**Question :** *Le résultat est-il lié à l'assiduité ? Peut-on anticiper la note ? Quand devient-ce trop incertain ?*

### Le lien est-il vérifié ?

**Oui.** Corrélation de Pearson **r = 0,987** → relation **très forte** entre assiduité et note.

### Peut-on anticiper la note ?

**Oui, avec prudence.** Trois modèles ont été comparés :

| Modèle | Erreur moyenne (RMSE) |
|--------|----------------------|
| SVR (RBF) — retenu | **0,95 point** /20 |
| Régression linéaire | 0,96 point /20 |
| KNN (k=5) | 1,04 points /20 |

Le modèle se trompe en moyenne de **moins d'1 point** sur 20 (97 % de variance expliquée).

### Seuil d'incertitude

| RMSE | Interprétation |
|------|----------------|
| < 2 pts | Anticipation **utilisable** avec prudence |
| ≥ 2 pts | Trop incertain pour une décision seule |

**Notre RMSE = 0,95 pt** → en dessous du seuil, mais l'estimation ne remplace jamais l'évaluation réelle.

→ [Rapport détaillé Q2](rapport_question2_regression.md)

---

## Question 3 — Profils d'élèves

**Question :** *Combien de profils distincts et comment les décrire ?*

**3 profils** identifiés par K-Means (sans utiliser l'orientation du conseil de classe) :

| Profil | Effectif | Note moy. | Assiduité moy. | Accompagnement suggéré |
|--------|----------|-----------|----------------|------------------------|
| **Fragile** | 347 | 3,5/20 | 17 % | Soutien renforcé, suivi absentéisme |
| **Intermédiaire** | 317 | 10,5/20 | 53 % | Consolidation, objectifs progressifs |
| **Solide** | 336 | 16,9/20 | 84 % | Maintien, approfondissement |

Ces profils **décrivent** des comportements scolaires ; ils ne **prescrivent** pas une filière.

→ [Rapport détaillé Q3](rapport_question3_clustering.md)

---

## Recommandations générales

1. **Ne pas automatiser les décisions d'orientation** — les modèles estiment et décrivent, ils ne remplacent pas le conseil de classe.
2. **Croiser les sources** — notes, assiduité, contrôles continus, entretiens.
3. **Utiliser les visualisations** — boîte à moustaches (Q1), nuage de points (Q2), carte des clusters (Q3) pour les réunions pédagogiques.
4. **Reproductibilité** — toutes les analyses sont rejouables via `python main.py`.

---

## Accès rapide au code

```bash
python main.py                              # Tout le TP
python src/models/analyse_univariee.py      # Q1 seule
python src/models/LinearRegression.py       # Q2 complète
python src/models/clustering_profils_eleves.py  # Q3 seule
```

Voir aussi le [README principal](../README.md) et l'[index des rapports](README.md).
