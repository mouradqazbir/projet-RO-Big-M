# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:26:06 2024

@author: delll
"""

import numpy as np

def methode_grand_m(FOB, MC, VSMC, M=1e7):
    """
    Résout un problème de programmation linéaire en appliquant la méthode du Grand M.

    Paramètres :
    FOB : Coefficients de la fonction objectif (liste ou numpy array)
    MC : Matrice représentant les contraintes (numpy array)
    VSMC : Valeurs du second membre des contraintes (liste ou numpy array)
    M : Pénalité associée aux variables artificielles (par défaut 1e7)

    Retourne :
    - La solution optimale
    - La valeur optimale de la fonction objectif
    """
    m, n = MC.shape  # On récupère les dimensions de la matrice des contraintes

    # Étape 1 : Construction du tableau initial
    tableau = np.hstack((MC, np.eye(m), VSMC.reshape(-1, 1)))  # Ajout des variables artificielles et du vecteur VSMC
    FOB = np.concatenate((FOB, [M] * m, [0]))  # Mise à jour de la fonction objectif avec les pénalités et le terme constant

    # Définition de la base initiale (indices des variables artificielles)
    base = list(range(n, n + m))

    while True:
        # Étape 2 : Identifier la colonne pivot (colonne avec le plus petit coefficient dans FOB)
        col_pivot = np.argmin(FOB[:-1])
        if FOB[col_pivot] >= 0:
            break  # La solution optimale est atteinte si tous les coefficients sont positifs ou nuls

        # Étape 3 : Calcul des rapports pour déterminer la ligne pivot
        rapports = np.divide(
            tableau[:, -1], tableau[:, col_pivot], out=np.full_like(tableau[:, -1], np.inf), where=tableau[:, col_pivot] > 0
        )
        ligne_pivot = np.argmin(rapports)

        # Étape 4 : Réalisation du pivot
        tableau[ligne_pivot] /= tableau[ligne_pivot, col_pivot]  # Normalisation de la ligne pivot
        for i in range(len(tableau)):
            if i != ligne_pivot:
                tableau[i] -= tableau[ligne_pivot] * tableau[i, col_pivot]

        # Mise à jour de la fonction objectif
        FOB -= FOB[col_pivot] * tableau[ligne_pivot]

        # Mise à jour de la base
        base[ligne_pivot] = col_pivot

    # Extraction de la solution optimale
    solution = np.zeros(n)
    for i, var in enumerate(base):
        if var < n:  # Ignorer les variables artificielles
            solution[var] = tableau[i, -1]

    # Calcul de la valeur optimale de la fonction objectif
    valeur_optimale = FOB[-1]
    return solution, valeur_optimale


# Exemple d'utilisation
FOB = np.array([-2, -3])  # Maximiser Z = 3x1 + 5x2 (coefficients négatifs pour la maximisation)
MC = np.array([[1, 2], [2, 1]])  # Matrice des contraintes
VSMC = np.array([8, 6])  # Second membre des contraintes

solution, valeur = methode_grand_m(FOB, MC, VSMC)
print("Solution optimale :", solution)
print("Valeur optimale :", valeur)
