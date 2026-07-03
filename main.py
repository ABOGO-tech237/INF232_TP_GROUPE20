import unicodedata
import hashlib

import numpy as np
import pandas as pd


def nettoyer_texte(texte):
    # 1. Normaliser et supprimer les accents
    nfkd_form = unicodedata.normalize('NFKD', texte)
    texte_sans_accent = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    texte_sans_accent = texte_sans_accent.upper()
    # 2. Supprimer les espaces
    return texte_sans_accent.replace(" ", "")


def generer_chaine(texte: str) -> int:
    """
    Calcule le hash SHA-256 d'une chaîne de caractères et le convertit
    en un entier valide utilisable comme graine (seed) par Scikit-learn.
    """
    texte = nettoyer_texte(texte)
    # 1. Hachage SHA-256 de la chaîne encodée en bytes
    hash_objet = hashlib.sha256(texte.encode('utf-8'))
    hash_hex = hash_objet.hexdigest()

    # 2. Conversion du code hexadécimal en entier
    graine_entiere = int(hash_hex, 16)

    # 3. Réduction de la taille pour Scikit-learn (qui attend un entier 32-bit signé/non-signé)
    # On utilise un modulo pour rester dans l'intervalle classique [0, 2**32 - 1]
    graine_valide = graine_entiere % (2 ** 12)

    return graine_valide

# Exemple d'utilisation
texte_original = "Atangana abog Emmanue"


def data_genertor(graine, nb_echantillons=1000, bruit=5.0):
    """
    Génère un jeu de données : Y est proportionnel à X avec un peu de bruit.
    X ∈ [0, 20], Y ∈ [0, 100], relation Y ≈ 5 * X.
    """
    rng = np.random.default_rng(graine)

    X = rng.uniform(0, 20, size=nb_echantillons)
    y = 5 * X + rng.normal(0, bruit, size=nb_echantillons)
    y = np.clip(y, 0, 100)

    return X, y


def vers_jeu_de_donnees(X, y, seuil=50.0):
    """
    Transforme les données en jeu de données tabulaire avec une cible binaire.
    target = 1 si Y >= seuil, sinon 0.
    """
    df = pd.DataFrame({"X": X, "Y": y})
    df["target"] = (df["Y"] >= seuil).astype(int)
    return df


def exporter_csv(df, chemin="donnees.csv"):
    """Exporte le jeu de données au format CSV."""
    df.to_csv(chemin, index=False)
    return chemin


graine = generer_chaine(texte_original)
X, y = data_genertor(graine=graine)
jeu = vers_jeu_de_donnees(X, y)
fichier = exporter_csv(jeu)

print(f"Graine : {graine}")
print(f"Fichier exporté : {fichier}")
print(f"Nombre de lignes : {len(jeu)}")
print(f"Répartition target : {jeu['target'].value_counts().to_dict()}")
print(jeu.head(10))