"""
Fichier : analyse.py
Description : Script principal d'automatisation (Analyse de convergence, Monte-Carlo, Validation ASME V&V 20).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Importation des fonctions depuis notre module externe
from fonctions import Generate_sample, LBM

dossier_devoir3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#On force Python à travailler depuis ce dossier
os.chdir(dossier_devoir3)
# -------------

if __name__ == "__main__":
    
    # --------------------------------------------------------------------------
    # PARTIE A : ANALYSE DE CONVERGENCE (u_num)
    # Objectif : Déterminer l'ordre de convergence du solveur et estimer l'incertitude numérique (GCI).
    # --------------------------------------------------------------------------
    deltaP = 0.1
    poro_base = 0.9
    mean_d_base = 12.5
    std_d_base = 2.85
    NX_base = 100
    dx_base = 2e-6
    L_phys = NX_base * dx_base  
    
    Nx_list = [50, 75, 100, 150, 200, 400]
    seeds = [101, 102, 103, 104, 105] 
    
    permeabilites_moyennes = []
    
    print(f"--- PARTIE A : ANALYSE DE CONVERGENCE ---")
    
    # Boucle sur les différentes résolutions de maillage
    for Nx in Nx_list:
        dx = L_phys / Nx
        print(f">> Maillage Nx = {Nx}...")
        perm_pour_ce_Nx = []
        
        # Test sur plusieurs configurations aléatoires (seeds) pour lisser les effets géométriques
        for seed in seeds:
            filename = os.path.join("results", f'fiber_mat_seed{seed}_Nx{Nx}.tiff')
            #filename = f'fiber_mat_seed{seed}_Nx{Nx}.tiff'
            save_image = (seed == 101 and Nx in [50, 100, 200])
            
            d_equivalent = Generate_sample(seed, filename, mean_d_base, std_d_base, poro_base, Nx, dx, save_img=save_image)
            k_sim = LBM(filename, Nx, deltaP, dx, d_equivalent)
            perm_pour_ce_Nx.append(k_sim)
            
            # Nettoyage des fichiers temporaires
            if os.path.exists(filename):
                os.remove(filename)
                
        k_moyen = np.mean(perm_pour_ce_Nx)
        permeabilites_moyennes.append(k_moyen)

    # Calculs pour l'erreur de discrétisation
    permeabilites_moyennes = np.array(permeabilites_moyennes)
    dx_array = L_phys / np.array(Nx_list)
    k_finest = permeabilites_moyennes[-1]
    erreur_relative = np.abs(k_finest - permeabilites_moyennes[:-1]) / k_finest
    dx_plot = dx_array[:-1]
    
    # Régression linéaire pour trouver l'ordre de convergence p
    coeffs = np.polyfit(np.log(dx_plot), np.log(erreur_relative), 1)
    ordre_convergence = coeffs[0]
    
    # Calcul de u_num selon la méthode GCI (Grid Convergence Index, ASME V&V 20)
    GCI = (1.25 / (2**ordre_convergence - 1)) * np.abs(permeabilites_moyennes[-2] - permeabilites_moyennes[-1])
    u_num = GCI / 2

    # Génération du graphique de convergence
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_plot, erreur_relative, 'bo-', label="Erreur relative moyenne")
    plt.loglog(dx_plot, np.exp(coeffs[1]) * dx_plot**coeffs[0], 'r--', label=f"Régression (p = {ordre_convergence:.2f})")
    plt.xlabel("Taille de maille dx (m)")
    plt.ylabel("Erreur relative")
    plt.title("Convergence de l'erreur de discrétisation")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    #plt.savefig("convergence.png")
    plt.savefig(os.path.join("results", "convergence.png"))
    plt.close()

    print(f"Ordre de convergence p = {ordre_convergence:.2f}")
    print(f"Incertitude numérique u_num = {u_num:.4f} µm²\n")

    # --------------------------------------------------------------------------
    # PARTIE B : PROPAGATION DES INCERTITUDES (MONTE-CARLO)
    # Objectif : Propager l'incertitude des données d'entrée vers les résultats.
    # --------------------------------------------------------------------------
    print("--- PARTIE B : MONTE-CARLO (u_input) ---")
    N_samples = 50
    Nx_MC = 100 
    dx_MC = L_phys / Nx_MC
    
    permeabilites_MC = []
    
    for i in range(N_samples):
        # Tirage aléatoire des paramètres d'entrée
        poro_tirage = np.random.normal(0.900, 0.0075)
        diam_tirage = np.random.normal(12.5, 2.85)
        
        # Sécurités physiques pour éviter des valeurs aberrantes
        poro_tirage = np.clip(poro_tirage, 0.5, 0.99)
        diam_tirage = np.clip(diam_tirage, 1.0, 50.0)
        
       #filename = f'fiber_mat_MC.tiff'
        filename = os.path.join("results", 'fiber_mat_MC.tiff')
        seed_auto = 0 # Génération entièrement stochastique
        
        # Génération et simulation
        d_eq = Generate_sample(seed_auto, filename, diam_tirage, 2.85, poro_tirage, Nx_MC, dx_MC)
        k_sim = LBM(filename, Nx_MC, deltaP, dx_MC, d_eq)
        permeabilites_MC.append(k_sim)
        print(f"Échantillon {i+1}/{N_samples} : poro={poro_tirage:.4f}, D={diam_tirage:.2f} -> k={k_sim:.2f}")
        
        if os.path.exists(filename):
            os.remove(filename)

    permeabilites_MC = np.array(permeabilites_MC)

    # Analyse statistique (distribution log-normale usuelle pour la perméabilité)
    log_k = np.log(permeabilites_MC)
    mu_log = np.mean(log_k)
    sigma_log = np.std(log_k, ddof=1)

    S_mediane = np.exp(mu_log)
    u_input_moins = S_mediane - np.exp(mu_log - sigma_log)
    u_input_plus = np.exp(mu_log + sigma_log) - S_mediane

    print(f"\nMédiane numérique S : {S_mediane:.4f} µm²")
    print(f"u_input (-) = {u_input_moins:.4f} µm²")
    print(f"u_input (+) = {u_input_plus:.4f} µm²\n")

    # Tracé des distributions PDF (densité) et CDF (cumulative)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(permeabilites_MC, bins=10, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title("PDF de la perméabilité")
    ax1.set_xlabel("Perméabilité (µm²)")
    ax1.set_ylabel("Densité")

    ax2.hist(permeabilites_MC, bins=10, density=True, cumulative=True, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title("CDF de la perméabilité")
    ax2.set_xlabel("Perméabilité (µm²)")
    ax2.set_ylabel("Probabilité cumulative")

    plt.tight_layout()
    #plt.savefig("PDF_CDF.png")
    plt.savefig(os.path.join("results", "PDF_CDF.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # PARTIES C, D, E : INCERTITUDE DE VALIDATION ET ERREUR DU MODÈLE
    # Objectif : Combiner les incertitudes et valider formellement le modèle.
    # --------------------------------------------------------------------------
    print("--- PARTIES C, D, E : VALIDATION ASME V&V 20 ---")
    
    # Partie C : Incertitude expérimentale u_D
    std_reproducibilite = 14.7
    std_permeametre = 10.0
    u_D = np.sqrt(std_reproducibilite**2 + std_permeametre**2)
    print(f"Incertitude expérimentale u_D = {u_D:.4f} µm²")

    # Partie D : Erreur de simulation E
    D_mediane = 80.6 # Valeur de référence expérimentale
    E = S_mediane - D_mediane
    print(f"Erreur de simulation E (S - D) = {E:.4f} µm²")

    # Partie E : Calcul de l'incertitude de validation (u_val)
    # Les incertitudes asymétriques issues du Monte-Carlo sont propagées séparément
    u_val_moins = np.sqrt(u_num**2 + u_input_moins**2 + u_D**2)
    u_val_plus = np.sqrt(u_num**2 + u_input_plus**2 + u_D**2)
    
    # Calcul de l'intervalle de confiance élargi (k=2 pour ~95.4%)
    borne_inf_intervalle = E - 2 * u_val_moins
    borne_sup_intervalle = E + 2 * u_val_plus
    
    print(f"\nIncertitude de validation asymétrique :")
    print(f"u_val (-) = {u_val_moins:.4f} µm²")
    print(f"u_val (+) = {u_val_plus:.4f} µm²")
    print(f"\nIntervalle de confiance de l'erreur du modèle (95.4%) :")
    print(f"[{borne_inf_intervalle:.4f} ; {borne_sup_intervalle:.4f}] µm²")
    
    # Conclusion sur l'adéquation du modèle selon la norme
    adequat = (borne_inf_intervalle <= 0 <= borne_sup_intervalle)
    print(f"\nLe modèle est-il adéquat (zéro inclus dans l'intervalle) ? {'OUI' if adequat else 'NON'}")