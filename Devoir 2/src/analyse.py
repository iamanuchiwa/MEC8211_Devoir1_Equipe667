"""
Fichier : analyse.py
Script d'exécution principal qui génère les graphiques du devoir.
Importe les outils depuis le fichier fonctions.py.
"""
import numpy as np
import matplotlib.pyplot as plt

# Importation depuis votre fichier fonctions.py
from fonctions import solve_diffusion_schema2, generer_fonctions_mms

# --- Préparation des fonctions MMS ---
f_C_mms, f_source, f_C_dirichlet, params_mms_dict = generer_fonctions_mms()


# =========================================================
# QUESTION c) TRACÉ DES PROFILS MMS
# =========================================================
print("Génération des graphiques MMS (Question c)...")
tdom = np.linspace(0, 5, 50)
rdom = np.linspace(0, 1, 50)
ti, ri = np.meshgrid(tdom, rdom, indexing='ij')

z_MMS = f_C_mms(ti, ri)
z_source = f_source(ti, ri)

plt.figure(figsize=(8, 6))
contour1 = plt.contourf(ri, ti, z_MMS, levels=50, cmap='viridis')
plt.colorbar(contour1, label='Concentration C_MMS')
plt.title('Solution Manufacturée $C_{MMS}(r,t)$')
plt.xlabel('Rayon r [m]')
plt.ylabel('Temps t [s]')
plt.show()

plt.figure(figsize=(8, 6))
contour2 = plt.contourf(ri, ti, z_source, levels=50, cmap='plasma')
plt.colorbar(contour2, label='Terme Source S_MMS')
plt.title('Terme Source Analytique $S_{MMS}(r,t)$')
plt.xlabel('Rayon r [m]')
plt.ylabel('Temps t [s]')
plt.show()


# =========================================================
# QUESTION d) ANALYSE DE CONVERGENCE SPATIALE
# =========================================================
print("\nAnalyse de convergence spatiale (Question d)...")
tf_conv = 0.5
dt_minuscule = 1e-5
N_values = [10, 20, 40, 80, 160]

erreurs_Linf_espace = []
drs_espace = []

for N_test in N_values:
    r_num, t_num, C_evolution = solve_diffusion_schema2(
        N=N_test, tf=tf_conv, dt=dt_minuscule, 
        is_mms=True, f_source=f_source, f_dirichlet=f_C_dirichlet, params_mms=params_mms_dict
    )
    C_num_final = C_evolution[-1]
    dr = r_num[1] - r_num[0]
    C_exact_final = f_C_mms(tf_conv, r_num)
    
    erreur_max = np.max(np.abs(C_num_final - C_exact_final))
    erreurs_Linf_espace.append(erreur_max)
    drs_espace.append(dr)
    print(f"N = {N_test:<4} | dr = {dr:.4f} | Erreur Linf = {erreur_max:.4e}")

coefs_espace = np.polyfit(np.log(drs_espace[-3:]), np.log(erreurs_Linf_espace[-3:]), 1)
print(f"--> PENTE SPATIALE OBSERVÉE : {coefs_espace[0]:.3f} (Ordre 2 attendu)")

plt.figure(figsize=(9, 6))
plt.loglog(drs_espace, erreurs_Linf_espace, 'b-o', linewidth=2, label=f'Erreur spatiale (Pente = {coefs_espace[0]:.2f})')
x_guide = np.array([drs_espace[0], drs_espace[-1]])
plt.loglog(x_guide, erreurs_Linf_espace[0] * (x_guide / drs_espace[0])**2, 'k--', alpha=0.6, label='Guide Pente 2')
plt.xlabel(r'Pas spatial $\Delta r$ [m]')
plt.ylabel(r'Erreur maximale $L_\infty$')
plt.title('Question d) Convergence spatiale (MMS)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# =========================================================
# QUESTION e) ANALYSE DE CONVERGENCE TEMPORELLE
# =========================================================
print("\nAnalyse de convergence temporelle (Question e)...")
tf_conv = 1.0
N_tres_fin = 300  
dt_values = [0.2, 0.1, 0.05, 0.025, 0.0125]

erreurs_Linf_temps = []

for dt_test in dt_values:
    r_num, t_num, C_evolution = solve_diffusion_schema2(
        N=N_tres_fin, tf=tf_conv, dt=dt_test, 
        is_mms=True, f_source=f_source, f_dirichlet=f_C_dirichlet, params_mms=params_mms_dict
    )
    C_num_final = C_evolution[-1]
    C_exact_final = f_C_mms(tf_conv, r_num)
    
    erreur_max = np.max(np.abs(C_num_final - C_exact_final))
    erreurs_Linf_temps.append(erreur_max)
    print(f"dt = {dt_test:<7} | Erreur Linf = {erreur_max:.4e}")

coefs_temps = np.polyfit(np.log(dt_values), np.log(erreurs_Linf_temps), 1)
print(f"--> PENTE TEMPORELLE OBSERVÉE : {coefs_temps[0]:.3f} (Ordre 1 attendu)")

plt.figure(figsize=(9, 6))
plt.loglog(dt_values, erreurs_Linf_temps, 'r-s', linewidth=2, label=f'Erreur temporelle (Pente = {coefs_temps[0]:.2f})')
x_guide_t = np.array([dt_values[0], dt_values[-1]])
plt.loglog(x_guide_t, erreurs_Linf_temps[0] * (x_guide_t / dt_values[0])**1, 'k--', alpha=0.6, label='Guide Pente 1')
plt.xlabel(r'Pas de temps $\Delta t$ [s]')
plt.ylabel(r'Erreur maximale $L_\infty$')
plt.title('Question e) Convergence temporelle (MMS)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# =========================================================
# QUESTION f) SIMULATION DU PROBLÈME PHYSIQUE RÉEL
# =========================================================
print("\nSimulation sur 126 ans (Question f)...")
tf_physique = 4e9          
dt_physique = 1e7          
N_physique = 50            

r_vrai, temps_vrai, C_evolution_vrai = solve_diffusion_schema2(
    N=N_physique, tf=tf_physique, dt=dt_physique, is_mms=False
)

plt.figure(figsize=(10, 6))
annees_cibles = [0.5, 2, 5, 15, 126]
indices_a_tracer = []

for annee in annees_cibles:
    temps_sec = annee * 365.25 * 24 * 3600
    idx = np.argmin(np.abs(temps_vrai - temps_sec))
    indices_a_tracer.append(idx)

couleurs = ['#a8ddb5', '#4eb3d3', '#2b8cbe', '#0868ac', '#023858']

for i, idx in enumerate(indices_a_tracer):
    t_en_annees = temps_vrai[idx] / (3600 * 24 * 365.25)
    plt.plot(r_vrai, C_evolution_vrai[idx], color=couleurs[i], linewidth=2, label=f't = {t_en_annees:.1f} ans')

plt.plot(r_vrai, C_evolution_vrai[0], 'k--', linewidth=1.5, label='t = 0 ans')
plt.xlabel('Rayon du pilier r [m]')
plt.ylabel('Concentration de sel C [mol/m$^3$]')
plt.title('Pénétration du sel vers l\'état d\'équilibre (126 ans)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(0, 0.5)
plt.ylim(0, 22)
plt.show()