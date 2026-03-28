# Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from levesque_fonctions import *
except:
    pass

class parametres():
    C0 = 1     #[mol]
    u_max = 1  #[m/s]
    L = 10      #[m]
    H = 10      #[m]
    P = 1      #[m]
    Pe = 200
    Da = 100
prm = parametres()

analyses = "3"

#Génération du profil de température
if "1" in analyses:
    trace_profil(100, 100, prm, 1)

#Tracé de la mms et du terme source
if "2" in analyses:
    trace_profil(100, 100, prm, 2)
    
if "3" in analyses:
    nx_list = [60, 120, 240, 480, 960]
    erreurs_inf = []
    erreurs_l2 = []
    h_list = []
    
    for nx in nx_list:
        ny = nx
        h = prm.L / (nx - 1)
        h_list.append(h)
        
        # Résolution numérique 
        C_num = concentration(nx, ny, prm, 1, 2, True)
        
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