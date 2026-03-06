"""
Fichier : fonctions.py
Contient les paramètres physiques, le solveur numérique 1D,
et le générateur symbolique pour la Méthode des Solutions Manufacturées (MMS).
"""
import numpy as np
import sympy as sp

# =========================================================
# PARAMÈTRES PHYSIQUES GLOBAUX
# =========================================================
Deff = 1e-10        # Coefficient de diffusion effectif [m^2/s]
k = 4e-9            # Constante d'évolution [s^-1]
Ce = 20.0           # Concentration à la surface [mol/m^3]
R_pilier = 0.5      # Rayon du pilier (Diamètre = 1m)

# =========================================================
# SOLVEUR NUMÉRIQUE
# =========================================================
def solve_diffusion_schema2(N, tf, dt, is_mms=False, f_source=None, f_dirichlet=None, params_mms=None):
    """
    Solveur 1D Transitoire Implicite.
    Peut résoudre le problème physique réel ou le problème MMS.
    """
    D_val = params_mms['Deff'] if is_mms else Deff
    k_val = params_mms['k'] if is_mms else k
    R_val = params_mms['R_pilier'] if is_mms else R_pilier
    Ce_val = params_mms['Ce'] if is_mms else Ce

    r = np.linspace(0, R_val, N)
    dr = r[1] - r[0]
    A = np.zeros((N, N))
    
    # 1. ASSEMBLAGE DE LA MATRICE A
    for i in range(1, N - 1):
        ri = r[i]
        alpha = D_val * dt
        A[i, i-1] = alpha * (1/(2*ri*dr) - 1/dr**2)
        A[i, i]   = 1.0 + 2*alpha/dr**2 + k_val*dt
        A[i, i+1] = alpha * (-1/(2*ri*dr) - 1/dr**2)

    # CF Centre (Neumann - Gear)
    A[0, 0] = -3.0
    A[0, 1] = 4.0
    A[0, 2] = -1.0

    # CF Surface (Dirichlet)
    A[N-1, N-1] = 1.0

    # 2. INITIALISATION
    C_num = np.zeros(N)
    
    if is_mms:
        C_num = Ce_val * np.exp(0) * (r/R_val)**3 

    temps = [0.0]
    C_evolution = [C_num.copy()]

    num_steps = int(round(tf / dt))
    t = 0.0

    # 3. BOUCLE TEMPORELLE
    for n in range(num_steps):
        t += dt  
        b = C_num.copy() 

        if is_mms and f_source is not None:
            S_eval = f_source(t, r[1:-1]) 
            b[1:-1] += S_eval * dt

        b[0] = 0.0
        if is_mms and f_dirichlet is not None:
            b[-1] = f_dirichlet(t)
        else:
            b[-1] = Ce_val         

        C_num = np.linalg.solve(A, b)
        
        temps.append(t)
        C_evolution.append(C_num.copy())

    return r, np.array(temps), np.array(C_evolution)

# =========================================================
# GÉNÉRATEUR MMS (SYMPY)
# =========================================================
def generer_fonctions_mms():
    """
    Génère et retourne les fonctions Python (lambdify) de la solution 
    manufacturée et de son terme source, ainsi que le dictionnaire de paramètres.
    """
    t, r = sp.symbols("t r")
    Ce_sym, R_sym, Deff_sym, k_sym = sp.symbols("Ce R_pilier Deff k")

    # Solution Manufacturée
    C_mms = sp.exp(-t) * Ce_sym * (r/R_sym)**3

    # Dérivées
    C_t = sp.diff(C_mms, t)
    C_r = sp.diff(C_mms, r)
    C_rr = sp.diff(C_mms, r, r)

    # Calcul du terme source analytique
    source = C_t - Deff_sym*(C_rr + C_r/r) + k_sym*C_mms

    # Paramètres unitaires pour la vérification numérique
    params_mms_sym = {Ce_sym: 1.0, R_sym: 1.0, Deff_sym: 1.0, k_sym: 1.0}
    params_mms_dict = {'Deff': 1.0, 'k': 1.0, 'R_pilier': 1.0, 'Ce': 1.0}

    # Conversion en fonctions utilisables par Numpy
    f_C_mms = sp.lambdify([t, r], C_mms.subs(params_mms_sym), "numpy")
    f_source = sp.lambdify([t, r], source.subs(params_mms_sym), "numpy")

    # Fonction pour la condition de Dirichlet
    C_dirichlet_expr = C_mms.subs(r, R_sym)
    f_C_dirichlet = sp.lambdify([t], C_dirichlet_expr.subs(params_mms_sym), "numpy")
    
    return f_C_mms, f_source, f_C_dirichlet, params_mms_dict