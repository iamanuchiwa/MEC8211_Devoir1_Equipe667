# Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from Levesque_Fonctions_A25 import *
except:
    pass

class parametres():
    C0 = 1 #[mol]
    u_max=0.1 #[m/s]
    L=100 #[m]
    H=50 #[m]
    P = 1 #[m]
    Pe = 100
    Da = 1000

prm = parametres()

"""
Ce fichier d'analyse comprend 2 parties:
    - Partie 1: vérification/validation: script pour génération du profil et des graphiques de comparaisons et d'erreurs
    - Partie 2: Questions d'analyse: graphiques de variation de Pe et Da, comparaison de matière adsorbée avec 2 parois ou une surface doublée
Les parties sont mises en commentaire, il suffit d'afficher la partie voulue pour générer le graphique ou afficher une solution
"""

#PARTIE 1: VÉRIFICATION ET VALIDATION

#Génération du profil de température
"""
nx = 100
ny = 100
x=np.linspace(0,prm.L,nx)
y=np.linspace(0,prm.H,ny)
C = concentration(nx,ny,prm,2)
C2D = C.reshape((ny, nx))
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(7,5))
plt.pcolormesh(X, Y, C2D, shading='auto')
plt.colorbar(label='Concentration')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Profil de concentration avec 2 parois adsorbantes (nx = ny = 100)')
plt.show()
"""

#Vérification et comparaison ordre 1/ordre 2
"""
nb_points = np.arange(35,130,1)

H = prm.H
L = prm.L
C0 = prm.C0
u_max = prm.u_max
Pe = prm.Pe
Da = prm.Da
Ds = u_max*H/Pe
k = Da*Ds/L

#Générationn du graphqiue de l'erreur en fonction du nombre de points nx / ny
erreurs1 = []
erreurs2 = []
for n in nb_points:
    indices_inf = np.arange(1, n)
    dx = L/(n-1)
    x = indices_inf * dx #créer le vecteur des positions sur la paroi inférieure
    ny = n
    q_analytique = 0.854*(u_max*H**2/(x*Ds))**(1/3) #calcul du flux adimensionnel analytique
    C1 = concentration(n, ny, prm,1,1)
    C1_bas_num = C1[indices_inf]
    q1_num = k*C1_bas_num*H/(Ds*C0) #calcul du flux adimensionnel numérique
    C2 = concentration(n, ny, prm,1,2)
    C2_bas_num = C2[indices_inf]
    q2_num = k*C2_bas_num*H/(Ds*C0)
    indice_10 = np.where(x > 10)[0][0] #trouver l'indice où x > 10m pour calculer l'erreur seulement sur ces valeurs
    erreur1 = np.sqrt(np.mean((q_analytique[indice_10:] - q1_num[indice_10:])**2))
    erreur2 = np.sqrt(np.mean((q_analytique[indice_10:] - q2_num[indice_10:])**2))
    erreurs1.append(erreur1)
    erreurs2.append(erreur2)

plt.plot(nb_points, erreurs1, label="Ordre 1 (pas de points fantômes)")
plt.plot(nb_points, erreurs2, label="Ordre 2 (points fantômes)")
plt.title("Erreur absolue sur le flux adimensionnel")
plt.xlabel("Nombre de points nx = ny")
plt.ylabel("Erreur absolue")
plt.legend()
plt.show()
"""

#Graphique de comparaison de la solution analytique et numérique (ordre 1 et ordre 2)
"""
n=149
indices_inf = np.arange(1, n)
dx = L/(n-1)
x = indices_inf * dx #créer le vecteur des positions sur la paroi inférieure
ny = n
q_analytique = 0.854*(u_max*H**2/(x*Ds))**(1/3) #calcul du flux adimensionnel analytique
C1 = concentration(n, ny, prm,1,1)
C1_bas_num = C1[indices_inf]
q1_num = k*C1_bas_num*H/(Ds*C0) #calcul du flux adimensionnel numérique
C2 = concentration(n, ny, prm,1,2)
C2_bas_num = C2[indices_inf]
q2_num = k*C2_bas_num*H/(Ds*C0)
indice_10 = np.where(x > 10)[0][0]
plt.plot(x[indice_10:],q_analytique[indice_10:], label="Analytique")
plt.plot(x[indice_10:],q1_num[indice_10:], label="Numérique, ordre 1 aux points limites")
plt.plot(x[indice_10:],q2_num[indice_10:], label="Numérique, ordre 2 avec points fantômes", linestyle="--")
plt.title("Flux adimesionnel sur la paroi inférieure pour nx = ny = 99")
plt.xlabel("x[m]")
plt.ylabel("Flux")
plt.legend()
plt.show()
"""
#PARTIE ANALYSE

#Analyse 1.a) variation du nombre de Péclet et Damkohler
"""
Pe = np.linspace(100,1000,50)
Q = []

for val_Pe in Pe:
    prm_i = prm
    prm_i.Pe = val_Pe
    Q.append(Q_c_simpson(nx, ny, prm_i,1))

plt.plot(Pe, Q)
plt.title("Quantité totale de matière adsorbée en faisant varier le nombre de Péclet et Da = 1000")
plt.xlabel("Nombre de Péclet (adimensionnel)")
plt.ylabel("Quantité totale de matière absorbée [mol/m^2]")
plt.show()

#Analyse 1.b) variation du nombre de Damkohler
Da = np.linspace(1e3,1e6,500)
Q = []

for val_Da in Da:
    prm_i = prm
    prm_i.Da = val_Da
    Q.append(Q_c_simpson(nx, ny, prm_i,1))

plt.semilogx(Da, Q)
plt.title("Quantité totale de matière adsorbée en faisant varier le nombre de Damköhler et Pe = 100")
plt.xlabel("Nombre de Damköhler (adimensionnel)")
plt.ylabel("Quantité totale de matière absorbée [mol/m^2]")
plt.show()
"""

#Analyse 2: quantité totale de matière adsorbée avec 2 parois ou une paroi à surface doublée
"""
#Analyse 2.a)
class parametres2():
    C0 = 1 #[mol]
    u_max=0.1 #[m/s]
    L=100 #[m]
    H=50 #[m]
    P = 1 #[m]
    Pe = 100 
    Da = 1e4

prm2 = parametres2()

A = prm2.L*prm2.P
Q_2_plaques = Q_c_simpson(nx,ny,prm2,2)*A
print("Qc = ", Q_2_plaques, "(2 parois asdorbantes)")

#Analyse 2.b) 
prmb = prm2
prmb.H = prm2.H/2
prmb.P = 2*prm2.P
A2 = prmb.P*prmb.L
Q_c2 = Q_c_simpson(nx,ny,prmb,1)*A2
print("Qc", Q_c2, "(Surface de la paroi inférieure doublée)")

"""
