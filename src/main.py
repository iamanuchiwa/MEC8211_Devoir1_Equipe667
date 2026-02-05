import numpy as np
import matplotlib.pyplot as plt
import math

# --- Paramètres Physiques ---
Deff = 1e-10        # Coefficient de diffusion effectif [m^2/s]
S = 2e-8            # Terme source constant [mol/m^3/s]
Ce = 20.0           # Concentration à la surface [mol/m^3]
R_pilier = 0.5      # Rayon du pilier (Diamètre = 1m)

# --- Solution Analytique (Eq. 2) ---
def solution_analytique(r):
    #C(r) = (S / 4Deff) * R^2 * ((r^2/R^2) - 1) + Ce
    terme1 = (S * 0.25 / ( Deff)) * (R_pilier**2)
    terme2 = ((r**2) / (R_pilier**2)) - 1
    return terme1 * terme2 + Ce


def solve_diffusion_schema1(N):
    """
    Résout l'équation 1D stationnaire avec:
    - Dérivée seconde : Centrée
    - Dérivée première : Décentrée avant (Forward)
    """
    # 1. Maillage
    r = np.linspace(0, R_pilier, N)
    dr = r[1] - r[0]  # Pas spatial delta_r

    # 2. Initialisation des matrices (Ax = b)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # 3. Remplissage des nœuds internes (i = 1 à N-2)
    # Equation: C_{i-1} + (-2 - dr/r_i)*C_i + (1 + dr/r_i)*C_{i+1} = S*dr^2/Deff
    for i in range(1, N - 1):
        ri = r[i]
        
        # Remplissage de la matrice A
        A[i, i-1] = 1.0
        A[i, i]   = -2.0 - (dr / ri) 
        A[i, i+1] = 1.0 + (dr / ri)
        
        # Remplissage du membre de droite b
        b[i] = (S * dr**2) / Deff

    # 4. Conditions Frontières (CF)
    
    # CF au Centre (i=0) : Neumann (dC/dr = 0)
    # Schéma avant: (C_1 - C_0) / dr = 0  => -C_0 + C_1 = 0
    A[0, 0] = -1.0
    A[0, 1] = 1.0
    b[0] = 0.0

    #CF à la Surface (i=N-1) : Dirichlet (C = Ce)
    A[N-1, N-1] = 1.0
    b[N-1] = Ce

    # 5. Résolution du système linéaire
    C_num = np.linalg.solve(A, b)
    
    return r, C_num, dr