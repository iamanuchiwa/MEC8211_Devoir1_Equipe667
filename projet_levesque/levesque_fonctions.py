"""
Ce code contient les fonctions nécessaire pour:
- Résoudre le problème de Levesque 2D 
- Calculer la qauntité totale de matière adsorbée
- Créer des graphiques de convergence à partir d'une MMS

"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

def concentration(nx, ny, prm, ordre = 2, mms = False):
    '''
    Entrées:
        + nx : Nombre de points en x
        + ny : Nombre de points en y
        + prm:
            - C0: concentration initiale (condition de Dirichlet) [mol]
            - u_max: vitesse maximale du fluide [m/s]
            - L: longueur du domaine [m]
            - H: hauteur du domaine [m]
            - P: profondeur du domaine [m]
            - Pe: nombre de Péclet (adimensionnel)
            - Da: nombre de Damkohler (adimensionnel)
        + ordre
            - 1: ordre 1 utilisé pour l'évaluation des points du centre 
            - 2: ordre 2 utilisé partout
        + mms: True pour utiliser le terme source
    Sortie:
        - Vecteur dimension N contenant les concentrations à chaque noeud
    '''

    assert nx >= 3, "Le nombre de points nx doit être supérieur ou égal à 3"
    assert ny >= 3, "Le nombre de points ny doit être supérieur ou égal à 3"

    #Extraction des paramètres, calcul de k et Ds à partir des nombres adimensionnels (Pe et Da)
    H = prm.H
    C0 = prm.C0
    Ds = prm.u_max*H/prm.Pe
    k = prm.Da*Ds/prm.L
    
    if mms:
        f_mms = genere_mms(prm)
    
    #Calcul du nombre de noeuds N, des pas dx et dy et des vecteurs de position et vitesses
    N = nx * ny
    dx = prm.L/(nx - 1)
    dy = H/(ny - 1)
    y_vect = np.linspace(0, H, ny)
    u_vect = -4*prm.u_max*y_vect*(y_vect - H)/(H**2)

    #Création des matrices du système: matrice creuse A et matrice b
    A = sparse.lil_matrix((N, N))
    b = np.zeros(N)

    #Bord gauche: condition de Dirichlet (coins exclus)
    for i in range(nx, N - nx - ny + 1, nx):
        A[i, i] = 1
        if mms:
            b[i] = f_mms[2](0, (i//nx)*dy)
        else:
            b[i] = C0

    #Bord droit: condition de Neumann (coins exclus)
    for i in range(2*nx - 1, N - nx, nx):
        A[i, i] = 3/(2*dx)
        A[i, i - 1] = -4/(2*dx)
        A[i, i - 2] = 1/(2*dx)

        if mms:
            b[i] = f_mms[3]((nx - 1)*dx, (i//nx)*dy)
    #Paroi du haut
    
        #Condition de Neumann 
    for i in range(N-nx, N):
        A[i, i] = 3/(2*dy)
        A[i, i - nx] = -4/(2*dy)
        A[i, i - 2*nx] = 1/(2*dy)
            
        if mms:
            b[i] = f_mms[4]((i % nx)*dx, (ny - 1)*dy)

    #Paroi du bas: conditon de Robin
    for i in range(0, nx):
        A[i, i] = -k - 3*Ds/(2*dy)
        A[i, i + nx] = 2*Ds/dy
        A[i, i + 2*nx] = -Ds/(2*dy)

        if mms:
            b[i] = f_mms[5](i*dx, 0)

    #Points intérieurs (advection-diffusion)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            n = j * nx + i
            u = u_vect[j]

            # Diffusion
            diag_diff = 2*Ds/(dx**2) + 2*Ds/(dy**2)
            off_x = -Ds/(dx**2)
            off_y = -Ds/(dy**2)

            if ordre == 1:
                # Advection Upwind Ordre 1
                A[n, n] = u/dx + diag_diff
                A[n, n - 1] = -u/dx + off_x
                A[n, n + 1] = off_x
                A[n, n + nx] = off_y
                A[n, n - nx] = off_y
        
            else:
                # Advection Ordre 2
                if i == 1:
                    # Cas spécial pour le premier point (Schéma centré)
                    A[n, n] = diag_diff
                    A[n, n + 1] = u/(2*dx) + off_x
                    A[n, n - 1] = -u/(2*dx) + off_x
                else:
                    # Schéma décentré arrière ordre 2 (3 points)
                    A[n, n] = 3*u/(2*dx) + diag_diff
                    A[n, n - 1] = -2*u/dx + off_x
                    A[n, n - 2] = u/(2*dx)
                    A[n, n + 1] = off_x 
            
                A[n, n + nx] = off_y
                A[n, n - nx] = off_y

            if mms:
                b[n] += f_mms[1](i*dx, j*dy)

    #Résolution du système matriciel
    A_csr = A.tocsr()
    C = spsolve(A_csr, b)

    return C

def Q_c_simpson(nx, ny, prm, ordre = 2, mms = False):
    '''
    Entrées:
        + nx : Nombre de points en x
        + ny : Nombre de points en y
        + prm:
            - C0: concentration initiale (condition de Dirichlet) [mol]
            - u_max: vitesse maximale du fluide [m/s]
            - L: longueur du domaine [m]
            - H: hauteur du domaine [m]
            - P: profondeur du domaine [m]
            - Pe: nombre de Péclet (adimensionnel)
            - Da: nombre de Damkohler (adimensionnel)
        + ordre
            - 1: ordre 1 utilisé pour l'évaluation des points à gauche du centre
            - 2: ordre 2 utilisé partout 
    Sortie:
        - Qc: quantité totale de matière adsorbée par unité de surface (mol/m^2)
    '''
    #Impose un nombre de points impair pour l'intégration Simpson
    assert nx%2 != 0, "Le nombre de points nx doit être impair"

    #Calcul de k et Ds à partir des nombres adimensionnels (Pe et Da)
    Ds = prm.u_max*prm.H/prm.Pe
    k = prm.Da*Ds/prm.L

    #Calcul de l'espacement des point h et le nombre de sous-intervalles N
    h = prm.L/(nx - 1)
    N = int((nx - 1)/2)

    #Génération des concentrations et extractions des concentrations à la paroi inférieure
    C = concentration(nx, ny, prm, ordre, mms)
    C_bas = C[0 : nx]
    Qc = 0

    #Calcul de l'intégrale
    for i in range(0, N):
        Qc += C_bas[2*i] + 4*C_bas[2*i + 1] + C_bas[2*i + 2]

    
    return Qc*h*prm.H*k/(3*Ds)


def genere_mms(prm):
    """
    Génère la MMS en utilisant les paramètres réels (u_max, Ds, k) 
    calculés à partir de Pe et Da.
    """
    x, y = sp.symbols("x y")
    C0_sym, L_sym, H_sym, u_max_sym, Ds_sym, k_sym = sp.symbols("C0 L H u_max Ds k")

    C = C0_sym * (1 + sp.sin(sp.pi * x / L_sym) * sp.cos(sp.pi * y / H_sym))
    u = -4 * u_max_sym * y * (y - H_sym) / (H_sym**2)

    Cx = sp.diff(C, x)
    Cxx = sp.diff(C, x, x)
    Cy = sp.diff(C, y)
    Cyy = sp.diff(C, y, y)

    #Terme source 
    s = u * Cx - Ds_sym * (Cxx + Cyy)

    #Préparation des valeurs numériques
    ds_val = (prm.u_max * prm.H) / prm.Pe
    k_val = (prm.Da * ds_val) / prm.L
    
    params = {
        C0_sym: prm.C0, 
        L_sym: prm.L, 
        H_sym: prm.H, 
        u_max_sym: prm.u_max, 
        Ds_sym: ds_val,
        k_sym: k_val
    }

    #fonctions lambdify
    f_C = sp.lambdify((x, y), C.subs(params), "numpy")
    f_source = sp.lambdify((x, y), s.subs(params), "numpy")

    #Conditions frontières
    f_dirichlet = sp.lambdify((x, y), C.subs(x, 0).subs(params), "numpy")
    f_neum_x = sp.lambdify((x, y), Cx.subs(x, prm.L).subs(params), "numpy")
    f_neum_y = sp.lambdify((x, y), Cy.subs(y, prm.H).subs(params), "numpy")
    robin_expr = -k_sym * C + Ds_sym * Cy
    f_robin = sp.lambdify((x, y), robin_expr.subs(y, 0).subs(params), "numpy")

    return [f_C, f_source, f_dirichlet, f_neum_x, f_neum_y, f_robin]


def trace_profil(nx, ny, prm, mode = 1):
    """
    Trace le profil de concentration en fonction de nx et ny
    - Mode = 1 --> solution numérique
    - Mode = 2 --> solution manufacturée et terme source
    """
    assert mode == 1 or mode == 2 or mode == 3

    x=np.linspace(0,prm.L,nx)
    y=np.linspace(0,prm.H,ny)

    if mode == 1:
        C = concentration(nx,ny,prm)
        C2D = C.reshape((ny, nx))
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(7,5))
        plt.pcolormesh(X, Y, C2D, shading='auto')
        plt.colorbar(label='Concentration')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f'Profil de concentration de la solution numérique\nnx = {nx} et ny = {ny}')
        plt.show()

    elif mode == 2:
        fonc_mms = genere_mms(prm)
        xi, yi = np.meshgrid(x, y)
        C = fonc_mms[0](xi, yi)
        S = fonc_mms[1](xi, yi)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(7,5))
        plt.pcolormesh(X, Y, C, shading='auto')
        plt.colorbar(label='Concentration')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f'Profil de concentration de la solution manufacturée\nnx = {nx} et ny = {ny}')
        plt.show()
        
        plt.figure(figsize=(7,5))
        plt.pcolormesh(X, Y, S, shading='auto')
        plt.colorbar(label='Concentration')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f'Terme source analytique\nnx = {nx} et ny = {ny}')
        plt.show()
