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

# --- Analyse de Convergence ---
# On teste plusieurs résolutions pour voir l'ordre de l'erreur
N_values = [5, 10, 20, 40, 80, 160, 320]
erreurs_L1 = []
erreurs_L2 = []
erreurs_Linf = []
deltas_r = []

print(f"{'N':<5} {'dr':<10} {'L_inf':<12}")
print("-" * 30)

for N in N_values:
    # Résolution numérique
    r, C_num, dr = solve_diffusion_schema1(N)
    
    # Solution exacte aux mêmes points
    C_exact = solution_analytique(r)
    
    # [cite_start]Calcul des erreurs
    diff = np.abs(C_num - C_exact)
    
    # Erreur L1 (Moyenne des valeurs absolues)
    err_L1 = np.sum(diff) / N
    
    # Erreur L2 (Racine carrée de la moyenne des carrés)
    err_L2 = np.sqrt(np.sum(diff**2) / N)
    
    # Erreur L_inf (Erreur maximale)
    err_Linf = np.max(diff)
    
    # Stockage pour le graphique
    erreurs_L1.append(err_L1)
    erreurs_L2.append(err_L2)
    erreurs_Linf.append(err_Linf)
    deltas_r.append(dr)
    
    print(f"{N:<5} {dr:<10.4f} {err_Linf:<12.2e}")

r_plot, C_plot, _ = solve_diffusion_schema1(20)
r_fine = np.linspace(0, R_pilier, 100) # Pour tracer la courbe analytique lisse

plt.figure(figsize=(10, 6))
plt.plot(r_fine, solution_analytique(r_fine), 'k-', label='Solution Analytique')
plt.plot(r_plot, C_plot, 'ro', label='Numérique (N=20)')
plt.xlabel('Rayon r [m]')
plt.ylabel('Concentration C [mol/m^3]')
plt.title('Comparaison Analytique vs Numérique')
plt.grid(True)
plt.legend()
plt.show()

# --- Graphique 2 : Analyse de Convergence (Log-Log) ---
plt.figure(figsize=(10, 6))
plt.loglog(deltas_r, erreurs_L1, 'b-o', label='Erreur L1')
plt.loglog(deltas_r, erreurs_L2, 'g-s', label='Erreur L2')
plt.loglog(deltas_r, erreurs_Linf, 'r-^', label='Erreur L_inf')

# Ajout d'une pente de référence (Ordre 1) pour comparer
# On prend un point de référence et on trace une ligne de pente 1
ref_x = np.array([deltas_r[0], deltas_r[-1]])
ref_y = erreurs_Linf[0] * (ref_x / ref_x[0])**1  # Pente 1
plt.loglog(ref_x, ref_y, 'k--', label='Pente Ordre 1 (Référence)')

plt.xlabel('Pas spatial $\Delta r$ [m]')
plt.ylabel('Erreur')
plt.title("Analyse de Convergence (Log-Log)")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()

# --- Comparaison des deux schémas (Question E.c) ---
N_test = 20  # Nombre de noeuds pour le graphique
r1, C1, _ = solve_diffusion_schema1(N_test)
r2, C2, _ = solve_diffusion_schema2(N_test)
r_fine = np.linspace(0, R_pilier, 200)

plt.figure(figsize=(10, 6))
plt.plot(r_fine, solution_analytique(r_fine), 'k-', label='Analytique Exacte')
plt.plot(r1, C1, 'bo--', label='Schéma 1 (Ordre 1)', markerfacecolor='none')
plt.plot(r2, C2, 'rs--', label='Schéma 2 (Ordre 2)', markerfacecolor='none')
plt.xlabel('Rayon r [m]')
plt.ylabel('Concentration C')
plt.title(f'Comparaison des schémas (N={N_test})')
plt.legend()
plt.grid(True)
plt.show()