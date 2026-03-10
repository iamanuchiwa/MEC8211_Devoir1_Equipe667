import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def concentration(nx, ny, prm, nb_plaque = 1, Ordre = 2):
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
        + nb_plaque: nombre de parois adsorbantes
            - 1: paroi inférieure seulement (valeur par défaut)
            - 2: parois inférieure et supérieure 
        + Ordre
            - 1: ordre 1 utilisé pour l'évaluation des points à gauche du centre --> recommandé pour nx et ny < 40
            - 2: ordre 2 utilisé partout (méthode des points fantômes) (valeur par défaut) --> recommandé pour nx et ny > 40
    Sortie:
        - Vecteur dimension N contenant les concentrations à chaque noeud
    '''

    assert nx >= 3, "Le nombre de points nx doit être supérieur ou égal à 3"
    assert ny >= 3, "Le nombre de points ny doit être supérieur ou égal à 3"

    #Extraction des paramètres, calcul de k et Ds à partir des nombres adimensionnels (Pe et Da)
    H = prm.H
    L = prm.L
    C0 = prm.C0
    u_max = prm.u_max
    Pe = prm.Pe
    Da = prm.Da
    Ds = u_max*H/Pe
    k = Da*Ds/L

    #Calcul du nombre de noeuds N, des pas dx et dy et des vecteurs de position et vitesses
    N = nx * ny
    dx = L/(nx-1)
    dy = H/(ny-1)
    y_vect = np.linspace(0, H, ny)
    u_vect = -4*u_max*y_vect*(y_vect - H)/(H**2)

    #Création des matrices du système: matrice creuse A et matrice b
    A = sparse.lil_matrix((N, N))
    b = np.zeros(N)
    
    #Bord gauche: condition de Dirichlet (coins exclus)
    for i in range(nx, N-nx-ny+1, nx):
        A[i, i] = 1
        b[i] = C0

    #Bord droit: condition de Neumann (coins exclus)
    for i in range(2*nx-1, N-nx, nx):
        A[i, i] = 3/(2*dx)
        A[i, i-1] = -4/(2*dx)
        A[i, i-2] = 1/(2*dx)
    
    #Paroi du haut
    if nb_plaque == 1:
        #Condition de Neumann si la paroi est non adsorbante
        for i in range(N-nx, N):
            A[i, i] = 3/(2*dy)
            A[i, i-nx] = -4/(2*dy)
            A[i, i-2*nx] = 1/(2*dy)
    else:
        #Condition de Robin si la paroi est adsorbante
        for i in range(N-nx, N):
            A[i, i] = k + 3*Ds/(2*dy)
            A[i, i-nx] = -2*Ds/dy
            A[i, i-2*nx] = Ds/(2*dy)
    
    #Paroi du bas: conditon de Robin
    for i in range(0, nx):
        A[i, i] = -k - 3*Ds/(2*dy)
        A[i, i+nx] = 2*Ds/dy
        A[i, i+2*nx] = -Ds/(2*dy)
    
    #Points intérieurs gauches (advection-diffusion)
    for i in range(nx+1, N-nx-ny+2, nx):
        j = i//nx
        u = u_vect[j]
        
        A[i,i+1] = - Ds/(dx**2)
        A[i,i+nx] = -Ds/(dy**2)
        A[i,i-nx] = -Ds/(dy**2)

        if Ordre == 2:  
            #Utilisation de points fantômes C[i-2]=C[i-1]=C0 (une colonne de points à C0 est ajoutée à gauche)
            A[i,i] = 3*u/(2*dx) + 2*Ds/(dx**2) + 2*Ds/(dy**2)
            A[i,i-1] = -3*u/(2*dx) - Ds/(dx**2)
        else:
            #Dérivée arrière d'ordre 1 (pas de points fantômes)
            A[i,i] = u/(dx) + 2*Ds/(dx**2) + 2*Ds/(dy**2)
            A[i,i-1] = -u/(dx) - Ds/(dx**2)

    #Points intérieurs (advection-diffusion)
    for i in range(2, nx-1):
        for j in range(1, ny-1):
            n = j * nx + i
            u = u_vect[j]
        
            A[n,n] = 3*u/(2*dx) + 2*Ds/(dx**2) + 2*Ds/(dy**2)
            A[n,n+1] = -Ds/(dx**2)
            A[n,n-1] = -2*u/dx - Ds/(dx**2)
            A[n,n-2] = u/(2*dx)
            A[n,n+nx] = -Ds/(dy**2)
            A[n,n-nx] = -Ds/(dy**2)
    
    #Résolution du système matriciel
    A_csr = A.tocsr()
    C = spsolve(A_csr, b)

    return C

def Q_c_simpson(nx, ny, prm, nb_plaque = 1, Ordre = 2):
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
        + nb_plaque: nombre de parois adsorbantes
            - 1: paroi inférieure seulement (valeur par défaut)
            - 2: parois inférieure et supérieure 
        + Ordre
            - 1: ordre 1 utilisé pour l'évaluation des points à gauche du centre --> recommandé pour nx et ny < 40
            - 2: ordre 2 utilisé partout (méthode des points fantômes) (valeur par défaut) --> recommandé pour nx et ny > 40
    Sortie:
        - Qc: quantité totale de matière adsorbée par unité de surface (mol/m^2)
    '''
    #Impose un nombre de points impair pour l'intégration Simpson
    assert nx%2 != 0, "Le nombre de points nx doit être impair"
    
    #Extraction des paramètres, calcul de k et Ds à partir des nombres adimensionnels (Pe et Da)
    H = prm.H
    L = prm.L
    C0 = prm.C0
    u_max = prm.u_max
    Pe = prm.Pe
    Da = prm.Da
    Ds = u_max*H/Pe
    k = Da*Ds/L

    #Calcul de l'espacement des point h et le nombre de sous-intervalles N
    h = L/(nx - 1)
    N = int((nx - 1)/2)

    #Génération des concentrations et extractions des concentrations à la paroi inférieure
    C = concentration(nx,ny,prm,nb_plaque, Ordre)
    C_bas = C[0 : nx]
    Qc=0
    
    #Calcul de l'intégrale
    for i in range(0, N):
        Qc+=h/3*H*k/(C0*Ds)*(C_bas[2*i]+4*C_bas[2*i+1]+C_bas[2*i+2])

    if nb_plaque == 2:
        #Extraction des concentrations de la paroi supérieure et calcul de l'intégrale
        C_haut = C[ny*nx-nx : ny*nx]
        Qc_haut = 0

        for i in range(0, N):
            Qc_haut+=h/3*H*k/(C0*Ds)*(C_haut[2*i]+4*C_haut[2*i+1]+C_haut[2*i+2])
        
        Qc+=Qc_haut

    return Qc  
    
    