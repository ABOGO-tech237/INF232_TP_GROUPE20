"""
Utilitaires de génération de graine (seed) pour la reproductibilité.

La graine est dérivée du nom du chef de groupe via SHA-256,
conformément aux consignes du TP (Section 3.1).
"""

import hashlib
import unicodedata


def nettoyer_texte(texte: str) -> str:
    """
    Normalise un nom pour le hachage : suppression des accents,
    passage en majuscules (espaces conservés, conformément au rapport).
    """
    texte = texte.strip()
    nfkd_form = unicodedata.normalize("NFKD", texte)
    texte_sans_accent = "".join(
        c for c in nfkd_form if not unicodedata.combining(c)
    )
    return texte_sans_accent.upper()


def generer_chaine(texte: str) -> int:
    """
    Calcule le hash SHA-256 d'une chaîne et le convertit en entier
    utilisable comme graine par NumPy / scikit-learn.

    On réduit le hash sur 12 bits pour rester dans un intervalle
    raisonnable tout en conservant la reproductibilité du TP.
    """
    texte_normalise = nettoyer_texte(texte)

    hash_objet = hashlib.sha256(texte_normalise.encode("utf-8"))
    hash_hex = hash_objet.hexdigest()

    graine_entiere = int(hash_hex, 16)
    # Modulo pour obtenir une graine compacte (consigne TP : entier 16 bits)
    return graine_entiere % (2**12)
