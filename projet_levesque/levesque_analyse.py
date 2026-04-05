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
    plt.loglog(h_list, erreurs_inf, 'b-o', linewidth=2, label=f'Erreur $L_\infty$ (Pente = {pente_inf:.2f})')
    plt.loglog(h_list, erreurs_l2, 'r-s', linewidth=2, label=f'Erreur $L_2$ (Pente = {pente_l2:.2f})')
    
    # Guide visuel (Pente 2)
    x_guide = np.array([h_list[0], h_list[-1]])
    y_guide = erreurs_inf[0] * (x_guide / h_list[0])**2
    plt.loglog(x_guide, y_guide, 'k--', alpha=0.6, label='Référence Pente 2')
    
    plt.xlabel('Pas spatial $\Delta x$ [m]')
    plt.ylabel('Erreur')
    plt.title('Analyse de convergence spatiale (Normes $L_\infty$ et $L_2$)')
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
