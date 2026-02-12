import datetime 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
#import pytest
#import os

try:
    from fonctions import *
except:
    pass

class parametres:
    Deff = 1e-10        # Coefficient de diffusion effectif [m^2/s]
    S = 2e-8            # Terme source constant [mol/m^3/s]
    Ce = 20.0           # Concentration à la surface [mol/m^3]
    R_pilier = 0.5      # Rayon du pilier (Diamètre = 1m)

# Création du dossier dans results avec horodatage
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
repo_root = Path(__file__).parent.parent
results_dir = repo_root / "results" / f"{timestamp}"
results_dir.mkdir(exist_ok=True)  # Crée le dossier s'il n'existe pas

# --- 1. Exécution des simulations ---
N_values = [10, 20, 40, 80, 160, 320]

print("Analyse du Schéma 1 (Ordre 1)...")
res1 = analyser_convergence(solve_diffusion_schema1, N_values)
pente1 = calculer_pente(res1, 'Linf')

print("Analyse du Schéma 2 (Ordre 2)...")
res2 = analyser_convergence(solve_diffusion_schema2, N_values)
pente2 = calculer_pente(res2, 'Linf')
# Note: Pente inutile pour Schéma 2 ici car l'erreur est ~3e-12 (précision machine)

# --- 2. Affichage des résultats consoles ---
print("-" * 60)
print(f"Schéma 1 - Pente Linf observée : {pente1:.3f} (Attendu : 1.0)")
print(f"Schéma 1 - Erreur Linf finale  : {res1['Linf'][-1]:.2e}")

print(f"Schéma 2 - Pente Linf observée : {pente2:.3f}")
print(f"Schéma 2 - Erreur Linf finale  : {res2['Linf'][-1]:.2e} (Précision Machine)")
print("-" * 60)

# =============================================================================
# GRAPHIQUE 1 : QUESTION D.a) : Profil Schéma 1 et Paramètres
# =============================================================================

# 1. Choix d'un maillage représentatif pour la figure
N_demo = 20  # On choisit N=20 pour avoir des points clairs
r_num, C_num, dr_demo = solve_diffusion_schema1(N_demo)

# 2. Affichage des paramètres dans la console (requis par la question)
print("="*40)
print("RÉPONSE D.a) - PARAMÈTRES DE SIMULATION")
print("="*40)
print(f"Coefficient diffusion (Deff) : {Deff:.1e} m^2/s")
print(f"Terme source (S)             : {S:.1e} mol/m^3/s")
print(f"Concentration surface (Ce)   : {Ce:.1f} mol/m^3")
print(f"Rayon du pilier (R)          : {R_pilier:.1f} m")
print(f"Nombre de nœuds (N)          : {N_demo}")
print(f"Pas spatial (dr)             : {dr_demo:.4f} m")
print("="*40)

# 3. Tracé du graphique
plt.figure(1,figsize=(10, 6))

# Courbe Analytique (Ligne continue)
r_fine = np.linspace(0, R_pilier, 200)
plt.plot(r_fine, solution_analytique(r_fine), 'k-', linewidth=2, label='Solution Analytique (Eq. 2)')

# Points Numériques Schéma 1 (Points)
plt.plot(r_num, C_num, 'bo', markersize=8, label=f'Solution Numérique (Schéma 1, N={N_demo})')

# Mise en forme
plt.xlabel('Rayon r [m]', fontsize=12)
plt.ylabel(r'Concentration C [mol/m$^3$]', fontsize=12)
plt.title(f'Question D.a) : Profil de concentration stationnaire\n(Comparaison Analytique vs Numérique Ordre 1)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Ajout d'une boîte de texte avec les paramètres directement sur le graphe (Bonus présentation !)
textstr = '\n'.join((
    r'$D_{eff}=%.1e$' % (Deff, ),
    r'$S=%.1e$' % (S, ),
    r'$C_e=%.1f$' % (Ce, ),
    r'$N=%d$' % (N_demo, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)


nom_f_da = f"QUESTION_Da_{timestamp}.png"
sauv_res(plt.figure(1), results_dir, nom_f_da)

#plt.show()

# =============================================================================
# GRAPHIQUE 2 : Question D.b (Les 3 erreurs du Schéma 1 avec Régression)
# =============================================================================
plt.figure(2, figsize=(10, 6))

# 1. Tracé des points expérimentaux
plt.loglog(res1['dr'], res1['L1'], 'b-^', label='Erreur L1')
plt.loglog(res1['dr'], res1['L2'], 'g-s', label='Erreur L2')
plt.loglog(res1['dr'], res1['Linf'], 'r-o', label='Erreur Linf')

# 2. Calcul de la régression linéaire sur la norme Linf
# On ne prend que les 4 derniers points (maillages fins) pour être dans la "partie linéaire" (asymptotique)
x_reg = np.array(res1['dr'][-4:])  
y_reg = np.array(res1['Linf'][-4:])
coefs = np.polyfit(np.log(x_reg), np.log(y_reg), 1) # Fit polynomial degré 1 sur les logs
pente_reg = coefs[0]
intercept = coefs[1]

# 3. Génération de la droite de régression pour l'affichage
# On l'étend sur tout le domaine pour voir l'alignement
x_fit = np.array([res1['dr'][0], res1['dr'][-1]])
y_fit = np.exp(intercept) * x_fit**pente_reg

# 4. Tracé de la régression
plt.loglog(x_fit, y_fit, 'k--', linewidth=1.5, label=f'Régression (Pente = {pente_reg:.3f})')

plt.xlabel(r'Pas spatial $\Delta r$ [m]')
plt.ylabel('Erreur')
plt.title('Question D.b : Convergence des erreurs avec Régression Linéaire')
plt.grid(True, which="both", ls="-")
plt.legend()

# Sauvegarde
nom_f_db = f"QUESTION_Db_{timestamp}.png"
sauv_res(plt.figure(2), results_dir, nom_f_db)

#plt.show()

# =============================================================================
# GRAPHIQUE 3 : QUESTION E.b) (Vérification du Schéma 2 - Les 3 normes)
# =============================================================================
plt.figure(3,figsize=(10, 6))

# 1. Tracé des 3 normes pour le Schéma 2
# Note : Comme l'erreur est de l'ordre de la précision machine (~1e-14),
# les courbes seront probablement "bruiteuses" ou plates, c'est normal.
plt.loglog(res2['dr'], res2['L1'], 'b-^', label='Erreur L1 (Schéma 2)')
plt.loglog(res2['dr'], res2['L2'], 'g-s', label='Erreur L2 (Schéma 2)')
plt.loglog(res2['dr'], res2['Linf'], 'r-o', label='Erreur Linf (Schéma 2)')

# 2. Ajout d'une ligne de référence "Précision Machine"
# La précision standard (float64) est environ 2e-16, mais les opérations accumulent l'erreur vers 1e-14/1e-13.
plt.axhline(y=1e-14, color='k', linestyle='--', alpha=0.5, label='Seuil Précision Machine')

plt.xlabel(r'Pas spatial $\Delta r$ [m]')
plt.ylabel('Erreur')
plt.title('Question E.b : Vérification du Schéma 2 (Erreurs vs Delta r)')
plt.grid(True, which="both", ls="-")
plt.legend()

# Sauvegarde
nom_f_eb = f"QUESTION_Eb_{timestamp}.png"
sauv_res(plt.figure(3), results_dir, nom_f_eb)

plt.show()

print("Graphique E.b généré : Le schéma est 'exact' pour ce problème, d'où l'erreur machine.")

# =============================================================================
# GRAPHIQUE 4 : Question E.c (Comparaison des Profils)
# =============================================================================
plt.figure(4, figsize=(10, 6))
# Courbe analytique fine
r_fine = np.linspace(0, R_pilier, 200)
plt.plot(r_fine, solution_analytique(r_fine), 'k-', label='Analytique')

# Points numériques (N=20)
r1, C1, _ = solve_diffusion_schema1(20)
plt.plot(r1, C1, 'bo', label='Schéma 1 (N=20)', fillstyle='none')

r2, C2, _ = solve_diffusion_schema2(20)
plt.plot(r2, C2, 'rx', label='Schéma 2 (N=20)')

plt.xlabel('Rayon r [m]')
plt.ylabel(r'Concentration $C$ [mol/m$^3$]')
plt.title('Question E.c : Comparaison des profils de concentration')
plt.legend()
plt.grid(True)

nom_f_ec = f"QUESTION_Ec_{timestamp}.png"
sauv_res(plt.figure(4), results_dir, nom_f_ec)

#plt.show()

# =============================================================================
# GRAPHIQUE 5 : Comparaison de Performance (Schéma 1 vs Schéma 2)
# =============================================================================
plt.figure(5, figsize=(10, 6))
plt.loglog(res1['dr'], res1['Linf'], 'b-o', label=f'Schéma 1 (Pente={pente1:.2f})')
plt.loglog(res2['dr'], res2['Linf'], 'r-s', label='Schéma 2 (Erreur Machine)')

plt.xlabel(r'Pas spatial $\Delta r$ [m]')
plt.ylabel(r'Erreur $L_{\infty}$')
plt.title('Comparaison de convergence : Schéma 1 vs Schéma 2')
plt.grid(True, which="both", ls="-")
plt.legend()

nom_f_ed = f"QUESTION_Ed_{timestamp}.png"
sauv_res(plt.figure(5), results_dir, nom_f_ed)

plt.show()