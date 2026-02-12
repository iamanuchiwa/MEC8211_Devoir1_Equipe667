"""
Ce programme contient les fonctions necessaires à la résolution du profil de concentration:
    - Résolution analytique
    - Résolution numérique
    - Analyse de convergence
"""

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
    """
    Retourne la concentration en un point r à partir de la solution analytique
        - r: rayon auquel calculer la concentration
    """
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
        A[i, i] = -2.0 - (dr / ri) 
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

def solve_diffusion_schema2(N):
    """
    Résout l'équation 1D stationnaire avec:
    - Dérivée seconde : Centrée (Ordre 2)
    - Dérivée première : Centrée (Ordre 2) -> Changement ici !
    """
    r = np.linspace(0, R_pilier, N)
    dr = r[1] - r[0]

    A = np.zeros((N, N))
    b = np.zeros(N)

    # --- Nœuds internes (i = 1 à N-2) ---
    # Equation avec schéma centré pour dC/dr:
    # Coeffs issus de : (C_i+1 - 2C_i + C_i-1)/dr^2 + (1/r_i)*(C_i+1 - C_i-1)/(2dr)
    for i in range(1, N - 1):
        ri = r[i]
        
        # Nouveaux coefficients (Schéma 2)
        coeff_im1 = 1.0 - (dr * 0.5 / ri)
        coeff_i   = -2.0
        coeff_ip1 = 1.0 + (dr * 0.5 / ri)
        
        A[i, i-1] = coeff_im1
        A[i, i]   = coeff_i
        A[i, i+1] = coeff_ip1
        b[i]      = (S * dr**2) / Deff

    # --- Conditions Frontières ---
    
    # CF au Centre (i=0) : Neumann (dC/dr = 0)
    # ATTENTION : Pour maintenir l'ordre 2, on utilise un schéma décentré à 3 points
    # Formule : -3*C0 + 4*C1 - 1*C2 = 0
    A[0, 0] = -3.0
    A[0, 1] = 4.0
    A[0, 2] = -1.0
    b[0]    = 0.0

    # CF à la Surface (i=N-1) : Dirichlet (C = Ce)
    A[N-1, N-1] = 1.0
    b[N-1] = Ce

    # Résolution
    C_num = np.linalg.solve(A, b)
    return r, C_num, dr

# =============================================================================
# ANALYSE DE CONVERGENCE ET GÉNÉRATION DES GRAPHIQUES
# =============================================================================

def analyser_convergence(fonction_solveur, N_values):
    """Calcule L1, L2 et Linf pour une liste de maillages."""
    results = {'dr': [], 'L1': [], 'L2': [], 'Linf': []}
    
    for N in N_values:
        r, C_num, dr = fonction_solveur(N)
        C_exact = solution_analytique(r)
        diff = np.abs(C_num - C_exact)  #Calcul erreur de discrétisation
        
        # Calcul des 3 normes demandées à la question D.b
        results['dr'].append(dr)
        results['L1'].append(np.sum(diff) / N)             # Norme L1
        results['L2'].append(np.sqrt(np.sum(diff**2) / N)) # Norme L2
        results['Linf'].append(np.max(diff))               # Norme Linf
        
    return results

def calculer_pente(results, metric='Linf'):
    """Calcule la pente entre les 2 maillages les plus fins (Diapo 19)."""
    err = results[metric]
    dr = results['dr']
    return np.log(err[-2] / err[-1]) / np.log(dr[-2] / dr[-1])
