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
    u_max=0.1  #[m/s]
    L=1      #[m]
    H=5      #[m]
    P = 1      #[m]
    Pe = 1
    Da = 1

prm = parametres()

analyses = "3"

#Génération du profil de température
if "1" in analyses:
    trace_profil(100, 100, prm, 1)

#Tracé de la mms et du terme source
if "2" in analyses:
    trace_profil(200, 200, prm, 2)
    
if "3" in analyses:
    nx_list = [40, 80, 160, 320, 640]
    erreurs = []
    h_list = []
    
    for nx in nx_list:
        ny = nx
        C_num = concentration(nx, ny, prm, 1, 2, True)
        f_c, f_s, f_diri, f_neum, f_rob = genere_mms()
        x = np.linspace(0, prm.L, nx)
        y = np.linspace(0, prm.H, ny)
        err_max = 0.0

        for j in range(ny):
            for i in range(nx):
                n = j * nx + i
                C_exact = f_c(x[i], y[j])
                err = abs(C_num[n] - C_exact)

                if err > err_max:
                    err_max = err

        erreurs.append(err_max)
        h_list.append(prm.L / (nx - 1))
        coefs = np.polyfit(np.log(h_list[-3:]), np.log(erreurs[-3:]), 1)
    
    print(f"--> PENTE SPATIALE OBSERVÉE : {coefs[0]:.3f} (Ordre 2 attendu)")
    plt.figure(figsize=(9, 6))
    plt.loglog(h_list, erreurs, 'b-o', linewidth=2, label=f'Erreur spatiale (Pente = {coefs[0]:.2f})')
    x_guide = np.array([h_list[0], h_list[-1]])
    plt.loglog(x_guide, erreurs[0] * (x_guide / h_list[0])**2, 'k--', alpha=0.6, label='Guide Pente 2')
    plt.xlabel('Pas spatial dx = dy [m]')
    plt.ylabel(r'Erreur maximale $L_\infty$')
    plt.title('Convergence spatiale (MMS)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()