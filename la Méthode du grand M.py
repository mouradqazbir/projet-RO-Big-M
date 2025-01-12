# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:26:06 2024

@author: delll
"""

import numpy as np

def solve_linear_program_mixed(c, A=None, b=None, A_eq=None, b_eq=None):
    """
    Résout un problème d'optimisation linéaire général avec NumPy.
    max z = c @ x
    sous les contraintes :
        - A @ x <= b (inégalités)
        - A_eq @ x = b_eq (égalités)
        - x >= 0
    
    Arguments :
    - c : Coefficients de la fonction objectif (1D array)
    - A : Matrice des contraintes d'inégalité (2D array, optionnel)
    - b : Limites supérieures des inégalités (1D array, optionnel)
    - A_eq : Matrice des contraintes d'égalité (2D array, optionnel)
    - b_eq : Limites des égalités (1D array, optionnel)
    
    Retourne :
    - Solution optimale x et valeur optimale z, ou None si aucune solution.
    """
    # Vérifier les dimensions
    num_vars = len(c)
    ineq_constraints = A is not None and b is not None
    eq_constraints = A_eq is not None and b_eq is not None

    # Construire la matrice étendue pour le tableau simplex
    num_constraints = (len(b) if ineq_constraints else 0) + (len(b_eq) if eq_constraints else 0)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

    # Ligne de la fonction objectif
    tableau[0, :num_vars] = -c  # Maximiser en mettant le négatif de c

    # Ajouter les contraintes d'inégalité
    if ineq_constraints:
        num_ineq = len(b)
        tableau[1:num_ineq + 1, :num_vars] = A
        tableau[1:num_ineq + 1, num_vars:num_vars + num_ineq] = np.eye(num_ineq)
        tableau[1:num_ineq + 1, -1] = b

    # Ajouter les contraintes d'égalité
    if eq_constraints:
        num_eq = len(b_eq)
        eq_start = 1 + (len(b) if ineq_constraints else 0)
        tableau[eq_start:eq_start + num_eq, :num_vars] = A_eq
        tableau[eq_start:eq_start + num_eq, -1] = b_eq

    # Indices des variables de base
    basis = list(range(num_vars, num_vars + (len(b) if ineq_constraints else 0)))

    # Méthode Simplex
    while True:
        # Identifier la colonne pivot (le coefficient le plus négatif de la ligne 0)
        col_pivot = np.argmin(tableau[0, :-1])
        if tableau[0, col_pivot] >= 0:
            # Optimalité atteinte (aucun coefficient négatif)
            break

        # Identifier la ligne pivot (test du rapport minimal)
        ratios = tableau[1:, -1] / tableau[1:, col_pivot]
        ratios[ratios <= 0] = np.inf  # Ignorer les rapports négatifs ou nuls
        row_pivot = np.argmin(ratios) + 1  # +1 car tableau[1:]

        if ratios[row_pivot - 1] == np.inf:
            # Problème non borné
            return None, None

        # Pivotage
        pivot_value = tableau[row_pivot, col_pivot]
        tableau[row_pivot, :] /= pivot_value
        for i in range(len(tableau)):
            if i != row_pivot:
                tableau[i, :] -= tableau[i, col_pivot] * tableau[row_pivot, :]

        # Mise à jour de la base
        basis[row_pivot - 1] = col_pivot

    # Extraire les solutions
    x = np.zeros(num_vars)
    for i, var_index in enumerate(basis):
        if var_index < num_vars:
            x[var_index] = tableau[i + 1, -1]

    # Valeur optimale
    z = tableau[0, -1]
    return x, z


# Définition des contraintes et de la fonction objectif
# Fonction objectif : z = 2x1 + 3x2
c = np.array([2, 3])  # Coefficients de la fonction objectif

# Contraintes d'inégalité : A @ x <= b
A = np.array([[1, 2], [2, 1]])  # Coefficients des contraintes d'inégalité
b = np.array([8, 6])  # Limites supérieures des inégalités

# Contraintes d'égalité : A_eq @ x = b_eq
A_eq = np.array([[1, 1]])  # Coefficients des contraintes d'égalité
b_eq = np.array([5])  # Limites des égalités

# Résolution
solution, optimal_value = solve_linear_program_mixed(c, A, b, A_eq, b_eq)

# Affichage des résultats
if solution is not None:
    print("Solution optimale trouvée :")
    print("x =", solution)
    print("Valeur optimale de z =", optimal_value)
else:
    print("Aucune solution optimale trouvée.")
