"""
Ce code effectue des tracés et analyses pour la vérification de code et de solution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
try:
    from levesque_fonctions import *
except:
    pass

# J'ai mis ca dans levesque_fonctions pour éviter les imports circulaires:) 
# class parametres():
#     C0 = 1     #[mol]
#     u_max = 1  #[m/s]
#     L = 10      #[m]
#     H = 10      #[m]
#     P = 1      #[m]
#     Pe = 0.5
#     Da = 50
# prm = parametres()

analyses = "5"

"""
1 - Profil de température
2 - MMS et terme source
3 - Analyse de convergence avec MMS
4 - Calcul du GCI
5 - Monte-Carlo et analyse de sensibilité globale

Exemple: analyses = "14" --> donnera le tracé du profil de température et calculera le GCI
"""

#Génération du profil de température
if "1" in analyses:
    trace_profil(100, 100, prm, 1)

#Tracé de la mms et du terme source
if "2" in analyses:
    trace_profil(100, 100, prm, 2)
    
if "3" in analyses:
    nx_list = [60, 120, 240, 480]
    erreurs_inf = []
    erreurs_l2 = []
    h_list = []
    
    for nx in nx_list:
        ny = nx
        h = prm.L / (nx - 1)
        h_list.append(h)
        
        # Résolution numérique 
        C_num = concentration(nx, ny, prm, 2, True)
        
        # Solution exacte
        fonct_mms = genere_mms(prm)
        x = np.linspace(0, prm.L, nx)
        y = np.linspace(0, prm.H, ny)
        X, Y = np.meshgrid(x, y)
        C_exact = fonct_mms[0](X, Y).flatten()

        diff = np.abs(C_num - C_exact)
        erreurs_inf.append(np.max(diff))
        err_l2 = np.sqrt(np.mean(diff**2))
        erreurs_l2.append(err_l2)
        
    # Calcul des pentes
    pente_inf = np.polyfit(np.log(h_list[-3:]), np.log(erreurs_inf[-3:]), 1)[0]
    pente_l2 = np.polyfit(np.log(h_list[-3:]), np.log(erreurs_l2[-3:]), 1)[0]
    
    print(f"--> PENTE L_inf : {pente_inf:.3f}")
    print(f"--> PENTE L_2   : {pente_l2:.3f}")

    # --- Graphique ---
    plt.figure(figsize=(9, 6))
    plt.loglog(h_list, erreurs_inf, 'b-o', linewidth=2, label=f'Erreur $L_infty$ (Pente = {pente_inf:.2f})')
    plt.loglog(h_list, erreurs_l2, 'r-s', linewidth=2, label=f'Erreur $L_2$ (Pente = {pente_l2:.2f})')
    
    # Guide visuel (Pente 2)
    x_guide = np.array([h_list[0], h_list[-1]])
    y_guide = erreurs_inf[0] * (x_guide / h_list[0])**2
    plt.loglog(x_guide, y_guide, 'k--', alpha=0.6, label='Référence Pente 2')
    
    plt.xlabel('Pas spatial $Delta x$ [m]')
    plt.ylabel('Erreur')
    plt.title('Analyse de convergence spatiale (Normes $L_infty$ et $L_2$)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# GCI
if "4" in analyses:
    nx = [657, 219, 73]
    r = 3
    p_f = 2
    Q = []

    for n in nx:
        Q_i = Q_c_simpson(n, n, prm, 2)
        Q.append(Q_i)

    ordre_obs = np.log(abs((Q[2] - Q[1])/(Q[1] - Q[0])))/np.log(r)
    print(Q)
    print("Ordre observé: ", ordre_obs)

    if abs((ordre_obs - p_f)/p_f) > 0.1:
        p = min(max(0.5, ordre_obs), p_f)
        f_s = 3
    else:
        p = p_f
        f_s = 1.25

    print("Fs: ", f_s)
    gci = f_s*abs(Q[1] - Q[0])/(r**p - 1)
    print("GCI: ", gci)


# Resultats Monte-Carlo et analyse de sensibilité globale
if "5" in analyses:
    results = monte_carlo_Qc(prm, N=500, nx=129, ny=129)
    df = pd.DataFrame(results)
    corr = df.corr()["Q"].sort_values()
    print("\n----- GLOBAL SENSITIVITY (Pearson R) -----") #Need to verify this w. method etc.
    print(corr)

#Propagation des incertitudes (Tilde)
"""Write your code down there"""


#Validation

# Validation selon le standard ASME V&V 20
if "5" in analyses:
    print("\n" + "="*50)
    print(" VALIDATION DE MODÈLE (ASME V&V 20)")
    print("="*50)
    
    # 1. Configuration des paramètres (Levesque exige Pe >= 100 pour la relation empirique)
    prm_val = parametres()
    prm_val.Pe = 100
    prm_val.Da = 10000
    prm_val.L = 100
    prm_val.u_max = 0.1
    prm_val.H = 10
    
    print("\n--- A. Calcul de S (Simulation) et u_num (GCI) ---")
    # Maillages impairs requis pour Simpson. Ratio r=3.
    # Nb points: 163 (fin), 55 (moyen), 19 (grossier)
    nx_v = [163, 55, 19]
    Q_gci = [Q_c_simpson(n, n, prm_val, 2) for n in nx_v]
    
    r = 3
    p_f = 2
    ordre_obs = np.log(abs((Q_gci[2] - Q_gci[1])/(Q_gci[1] - Q_gci[0])))/np.log(r)
    
    if abs((ordre_obs - p_f)/p_f) > 0.1:
        p = min(max(0.5, ordre_obs), p_f)
        f_s = 3
    else:
        p = p_f
        f_s = 1.25

    gci = f_s * abs(Q_gci[1] - Q_gci[0]) / (r**p - 1)
    
    # Résultat de la simulation (maillage le plus fin)
    S = Q_gci[0]
    # Incertitude numérique selon ASME V&V 20
    u_num = gci / 2  
    
    print(f"Valeur simulée S   = {S:.5e} mol/m^2")
    print(f"Incertitude u_num  = {u_num:.5e}")

    print("\n--- B. Calcul de D (Expérimental/Empirique) et E ---")
    D = Q_c_empirique(prm_val)
    E = S - D 
    print(f"Valeur empirique D = {D:.5e} mol/m^2")
    print(f"Erreur comparaison E = S - D = {E:.5e}")

    print("\n--- C. Propagation des incertitudes d'entrée (u_input) ---")
    # Hypothèse: incertitude de 5% sur la vitesse max et 5% sur la hauteur H
    u_umax = 0.05 * prm_val.u_max
    u_H = 0.05 * prm_val.H
    
    # Perturbations pour les différences finies (1%)
    du = 0.01 * prm_val.u_max
    dH = 0.01 * prm_val.H
    
    n_fine = 163
    
    # Sensibilité par rapport à u_max
    prm_u_plus = parametres(); prm_u_plus.Pe=100; prm_u_plus.Da=10000; prm_u_plus.L=100; prm_u_plus.H=10; prm_u_plus.u_max = prm_val.u_max + du
    prm_u_moins = parametres(); prm_u_moins.Pe=100; prm_u_moins.Da=10000; prm_u_moins.L=100; prm_u_moins.H=10; prm_u_moins.u_max = prm_val.u_max - du
    dS_du = (Q_c_simpson(n_fine, n_fine, prm_u_plus, 2) - Q_c_simpson(n_fine, n_fine, prm_u_moins, 2)) / (2 * du)
    
    # Sensibilité par rapport à H
    prm_H_plus = parametres(); prm_H_plus.Pe=100; prm_H_plus.Da=10000; prm_H_plus.L=100; prm_H_plus.u_max=0.1; prm_H_plus.H = prm_val.H + dH
    prm_H_moins = parametres(); prm_H_moins.Pe=100; prm_H_moins.Da=10000; prm_H_moins.L=100; prm_H_moins.u_max=0.1; prm_H_moins.H = prm_val.H - dH
    dS_dH = (Q_c_simpson(n_fine, n_fine, prm_H_plus, 2) - Q_c_simpson(n_fine, n_fine, prm_H_moins, 2)) / (2 * dH)
    
    # Incertitude d'entrée totale
    u_input = np.sqrt((dS_du * u_umax)**2 + (dS_dH * u_H)**2)
    print(f"Incertitude u_input = {u_input:.5e}")

    print("\n--- D. Incertitude expérimentale (u_D) ---")
    # Hypothèse: la relation empirique a une incertitude inhérente de 2%
    u_D = 0.02 * D
    print(f"Incertitude u_D     = {u_D:.5e}")

    print("\n--- E. Bilan de validation (u_val et erreur du modèle) ---")
    u_val = np.sqrt(u_num**2 + u_input**2 + u_D**2)
    print(f"Incertitude globale u_val = {u_val:.5e}")
    
    # Intervalle de confiance à 95.4% (k=2)
    k = 2
    borne_inf = E - k * u_val
    borne_sup = E + k * u_val
    
    print(f"\nCONCLUSION :")
    print(f"L'erreur du modèle (delta_model) se situe dans l'intervalle de confiance à 95.4% :")
    print(f"[ {borne_inf:.5e}  ,  {borne_sup:.5e} ]")
    print(f"Ratio |E| / u_val = {abs(E)/u_val:.2f}")
    print("="*50)


# ==========================================
# GRAPHIQUE 1 : Intervalle ASME V&V 20
# ==========================================
# On réutilise les valeurs exactes obtenues lors de ton exécution
E = 19.4088
u_val = 8.21595
k = 2 # Facteur d'élargissement pour 95.4% de confiance
U_val_95 = k * u_val

plt.figure(figsize=(6, 8))

# Ligne zéro (référence où la simulation correspondrait exactement à l'expérience)
plt.axhline(0, color='black', linewidth=1.5, linestyle='-')

# Point d'erreur E avec la barre d'incertitude globale U_val
plt.errorbar(1, E, yerr=U_val_95, fmt='ko', capsize=10, capthick=2, markersize=8, label=r'Erreur de comparaison $E$')

# Annotations pour bien lier au cours MEC8211
plt.text(1.05, E, r'$E \approx \delta_{model}$', fontsize=12, verticalalignment='center')
plt.text(1.05, E + U_val_95, r'$(\hat{\delta}_{model})_{max}$', fontsize=12, verticalalignment='bottom')
plt.text(1.05, E - U_val_95, r'$(\hat{\delta}_{model})_{min}$', fontsize=12, verticalalignment='top')

# Remplissage pour mettre en évidence la zone d'erreur du modèle
plt.axhspan(E - U_val_95, E + U_val_95, alpha=0.15, color='red', label=r"Intervalle de confiance de $\delta_{model}$")

# Mise en forme du graphique
plt.xlim(0.5, 2.0)
plt.xticks([]) # On cache l'axe X car c'est un point de validation unique
plt.ylabel('Erreur de comparaison $E = S - D$ $[mol/m^2]$', fontsize=12)
plt.title('Validation ASME V&V 20 : Estimation de l\'erreur du modèle\n(Correspond au "Case 3a")', fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()


# ==========================================
# GRAPHIQUE 2 : Profil de flux local le long de la paroi
# ==========================================
print("\n--- Génération du Graphique 2 : Profil de flux local ---")

# 1. Configuration des paramètres (validité Pe >= 100)
prm_plot = parametres()
prm_plot.Pe = 100
prm_plot.Da = 10000
prm_plot.L = 100
prm_plot.H = 10
prm_plot.u_max = 0.1
prm_plot.C0 = 1.0

# Maillage suffisamment fin pour une belle courbe
nx_plot = 241
ny_plot = 241

# Calcul des constantes physiques
Ds = prm_plot.u_max * prm_plot.H / prm_plot.Pe
k = prm_plot.Da * Ds / prm_plot.L

# 2. Obtention de la solution numérique
print("Résolution du champ de concentration en cours...")
C_num = concentration(nx_plot, ny_plot, prm_plot, ordre=2, mms=False)

# Extraction de la concentration sur la paroi inférieure (les nx premiers noeuds)
C_bas = C_num[0:nx_plot]
x_vect = np.linspace(0, prm_plot.L, nx_plot)

# Flux numérique via la condition de Robin imposée
q_num = (prm_plot.H / prm_plot.C0) * (k / Ds) * C_bas

# 3. Calcul de la solution empirique analytique
q_emp = np.zeros(nx_plot)
q_emp[0] = np.nan # On assigne NaN à x=0 pour éviter l'erreur de division par zéro

for i in range(1, nx_plot):
    q_emp[i] = 0.854 * ( (prm_plot.u_max * (prm_plot.H**2)) / (x_vect[i] * Ds) )**(1/3)

# 4. Tracé du graphique
plt.figure(figsize=(8, 5))

# Tracé des courbes à partir du 2e noeud (index 1) pour esquiver la singularité
plt.plot(x_vect[1:], q_emp[1:], 'k-', linewidth=2, label='Solution empirique')
plt.plot(x_vect[1:], q_num[1:], 'r--', linewidth=2, label='Solution numérique')

plt.xlabel('Position le long de la plaque x [m]', fontsize=12)
plt.ylabel('Flux adimensionnel', fontsize=12)
plt.title(f'Validation du profil de flux surfacique (Pe = {prm_plot.Pe})', fontsize=13)

# Ajustement de l'axe Y pour ne pas aplatir le graphique à cause de la divergence proche de x=0
plt.ylim(0, max(q_num[1:]) * 1.5)
plt.xlim(0, prm_plot.L)

plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()

# ==========================================
# GRAPHIQUE 3 : Qc en fonction de Pe
# ==========================================
print("\n--- Génération du Graphique 3 : Qc en fonction de Pe ---")

prm_sweep = parametres()
prm_sweep.Da = 10000
prm_sweep.L = 100
prm_sweep.H = 10
prm_sweep.u_max = 0.1
prm_sweep.C0 = 1.0

pe_list = [100, 325, 550, 775, 1000]
Qc_num_list = []
u_num_list = []

nx_v = [163, 55, 19]
r = 3
p_f = 2

for pe in pe_list:
    print(f"Calcul des simulations pour Pe = {pe}...")
    prm_sweep.Pe = pe
    
    Q_gci = [Q_c_simpson(n, n, prm_sweep, 2) for n in nx_v]
    
    ordre_obs = np.log(abs((Q_gci[2] - Q_gci[1])/(Q_gci[1] - Q_gci[0])))/np.log(r)
    if abs((ordre_obs - p_f)/p_f) > 0.1:
        p = min(max(0.5, ordre_obs), p_f)
        f_s = 3
    else:
        p = p_f
        f_s = 1.25
        
    gci = f_s * abs(Q_gci[1] - Q_gci[0]) / (r**p - 1)
    
    Qc_num_list.append(Q_gci[0])
    u_num_list.append(gci / 2) 

pe_dense = np.linspace(100, 1000, 100)
Qc_emp_dense = []

for pe in pe_dense:
    prm_sweep.Pe = pe
    Qc_emp_dense.append(Q_c_empirique(prm_sweep))

# --- CORRECTION ICI : Création préalable de la liste pour le fill_between ---
Qc_emp_list = []
for pe in pe_list:
    prm_sweep.Pe = pe
    Qc_emp_list.append(Q_c_empirique(prm_sweep))
# --------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(pe_dense, Qc_emp_dense, 'k-', linewidth=2, label='Modèle empirique (Vérité terrain)')

plt.errorbar(pe_list, Qc_num_list, yerr=u_num_list, fmt='rs', capsize=5, capthick=1.5, markersize=7, label='Simulation numérique avec $u_{num}$')

plt.xlabel('Nombre de Péclet (Pe)', fontsize=12)
plt.ylabel('Quantité totale adsorbée $Q_c$ $[mol/m^2]$', fontsize=12)
plt.title('Validation du modèle : Réponse du système selon le régime d\'écoulement', fontsize=13)

# On utilise la nouvelle liste directement
plt.fill_between(pe_list, Qc_num_list, Qc_emp_list, color='red', alpha=0.1, label='Erreur de comparaison $E$')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()