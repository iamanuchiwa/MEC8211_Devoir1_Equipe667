import numpy as np
import matplotlib.pyplot as plt
import pytest
try:
    from fonctions import *
except:
    pass

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