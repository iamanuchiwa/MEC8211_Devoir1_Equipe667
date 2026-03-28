"""
Fichier : fonctions.py
Description : Contient les fonctions de génération de géométrie poreuse et le solveur LBM.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit, prange

# ==============================================================================
# FONCTION : Generate_sample
# ==============================================================================
def Generate_sample(seed, filename, mean_d, std_d, poro, nx, dx, save_img=False):
    """
    Génère un milieu poreux aléatoire (fibres cylindriques) avec une porosité cible.
    """
    # Initialisation du générateur de nombres aléatoires
    rng = np.random.default_rng(None if seed == 0 else seed)
    dx_um = dx * 1e6
    domain = nx * dx_um

    # Tirage d'une large distribution de diamètres de fibres
    dist_full = rng.normal(mean_d, std_d, 10000)

    # Calcul initial de la porosité avec une seule fibre
    nb_fiber = 1
    poro_eff = 1.0 - np.sum(dist_full[:nb_fiber] ** 2 / 4 * np.pi) / domain ** 2
    poro_eff_old = poro_eff

    # Ajout itératif de fibres jusqu'à atteindre la porosité cible
    while poro_eff >= poro:
        poro_eff_old = poro_eff
        nb_fiber += 1
        poro_eff = 1.0 - np.sum(dist_full[:nb_fiber] ** 2 / 4 * np.pi) / domain ** 2

    # Ajustement final pour être au plus proche de la porosité voulue
    if abs(poro_eff - poro) > abs(poro_eff_old - poro):
        nb_fiber -= 1
        poro_eff = poro_eff_old

    # Tri des diamètres (du plus grand au plus petit)
    dist_d = np.sort(dist_full[:nb_fiber])[::-1]
    d_equivalent = np.sum(dist_d ** 2) / np.sum(dist_d)

    # Placement des fibres en évitant les chevauchements
    circles = np.zeros((nb_fiber, 3))
    circles[0] = [rng.random() * domain, rng.random() * domain, dist_d[0]]
    fiber_count = 1
    offsets = np.array([0.0, domain, -domain]) # Pour gérer la périodicité du domaine

    while fiber_count < nb_fiber:
        di = dist_d[fiber_count]
        xi = rng.random() * domain
        yi = rng.random() * domain

        xc = circles[:fiber_count, 0]
        yc = circles[:fiber_count, 1]
        dc = circles[:fiber_count, 2]
        r2 = (di + dc) ** 2                            

        ox, oy = np.meshgrid(offsets, offsets, indexing='ij')
        ox = ox.ravel()                                  
        oy = oy.ravel()

        # Vérification des distances entre la nouvelle fibre et les existantes (avec périodicité)
        dx2 = (xi - xc[:, np.newaxis] + ox[np.newaxis, :]) ** 2
        dy2 = (yi - yc[:, np.newaxis] + oy[np.newaxis, :]) ** 2

        if np.any(dx2 + dy2 < r2[:, np.newaxis]):
            continue   

        # Acceptation de la position
        circles[fiber_count] = [xi, yi, di]
        fiber_count += 1

    # Création de la matrice binaire (Solide/Fluide)
    coords = (0.5 + np.arange(nx)) * dx_um
    px, py = np.meshgrid(coords, coords, indexing='ij')   
    poremat = np.zeros((nx, nx), dtype=bool)

    xc = circles[:, 0]
    yc = circles[:, 1]
    r2 = (circles[:, 2] / 2) ** 2

    # Remplissage de la matrice poreuse
    for k in range(nb_fiber):
        for ox in offsets:
            for oy in offsets:
                poremat |= (px - (xc[k] + ox)) ** 2 + (py - (yc[k] + oy)) ** 2 < r2[k]

    poremat_img = poremat.T   
    # Sauvegarde de la géométrie au format image
    Image.fromarray(poremat_img.astype(np.uint8) * 255).save(filename)

    # Sauvegarde facultative d'une image de visualisation
    if save_img:
        plt.figure(figsize=(6,6))
        plt.imshow(np.rot90(poremat_img, k=1), cmap='gray')
        plt.title(f"Géométrie : Seed={seed}, Nx={nx}")
        plt.axis('off')
        plt.tight_layout()
        #plt.savefig(f"geom_seed{seed}_Nx{nx}.png")
        plt.savefig(os.path.join("results", f"geom_seed{seed}_Nx{nx}.png"))
        plt.close()

    return d_equivalent

# ==============================================================================
# NOYAU LBM COMPILÉ PAR NUMBA  
# ==============================================================================
@njit(parallel=True, cache=True)
def _lbm_step(N, SOLID, W, cx, cy, NX, NY, deltaP, dx, rho0, dt, OMEGA, bb_idx):
    """
    Exécute une itération de la méthode de Lattice Boltzmann (Streaming + Collision).
    Optimisé avec Numba pour des calculs parallèles rapides.
    """
    NQ = 9 # Modèle D2Q9 (9 directions)
    NCELL = NX * NY
    N_stream = np.empty_like(N)
    N_stream[:, 0] = N[:, 0]   

    # 1. Étape de Streaming (propagation des populations de particules)
    for q in prange(1, NQ):
        shift_x = int(cx[q])
        shift_y = int(cy[q])
        for idx in range(NCELL):
            i = idx // NY
            j = idx % NY
            src_i = (i - shift_x) % NX
            src_j = (j - shift_y) % NY
            N_stream[idx, q] = N[src_i * NY + src_j, q]

    # Sauvegarde de l'état avant condition limite de rebond (Bounce-back)
    N_solid_save = np.empty((NCELL, NQ), dtype=N.dtype)
    for idx in prange(NCELL):
        if SOLID[idx]:
            for q in range(NQ):
                N_solid_save[idx, q] = N_stream[idx, bb_idx[q]]

    ux_out = np.empty(NCELL)

    # 2. Étape de Collision (relaxation vers l'équilibre)
    for idx in prange(NCELL):
        rho_i = 0.0
        ux_i = 0.0
        uy_i = 0.0
        
        # Calcul des moments macroscopiques (densité et vitesse)
        for q in range(NQ):
            f = N_stream[idx, q]
            rho_i += f
            ux_i += f * cx[q]
            uy_i += f * cy[q]

        # Ajout du forçage (gradient de pression) à la vitesse
        ux_i = ux_i / rho_i + deltaP / (2.0 * NX * dx * rho0) * dt
        uy_i = uy_i / rho_i
        ux_out[idx] = ux_i

        u2 = ux_i ** 2 + uy_i ** 2
        
        # Opérateur de collision BGK
        for q in range(NQ):
            cu = ux_i * cx[q] + uy_i * cy[q]
            feq = rho_i * W[q] * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2)
            N_stream[idx, q] += OMEGA * (feq - N_stream[idx, q])

    # 3. Application des conditions limites de type Bounce-back sur les obstacles solides
    for idx in prange(NCELL):
        if SOLID[idx]:
            for q in range(NQ):
                N_stream[idx, q] = N_solid_save[idx, q]

    # Calcul du débit global pour vérifier la convergence
    flow = 0.0
    for j in range(NY):
        flow += ux_out[j]
    flow /= (NX * dx)

    return N_stream, ux_out, flow

# ==============================================================================
# FONCTION : LBM
# ==============================================================================
def LBM(filename, NX, deltaP, dx, d_equivalent):
    """
    Initialise et exécute la simulation LBM jusqu'à convergence stationnaire,
    puis calcule et retourne la perméabilité.
    """
    NY = NX
    OMEGA = 1.0
    rho0 = 1.0
    mu = 1.8e-5 # Viscosité dynamique
    epsilon = 1e-8 # Critère de convergence
    dt = (1.0 / OMEGA - 0.5) * rho0 * dx ** 2 / 3.0 / mu

    # Chargement de la géométrie
    A = np.array(Image.open(filename)).astype(bool)
    SOLID = A.flatten()

    # Paramètres du modèle D2Q9 (poids, vecteurs de vitesse, indices de rebond)
    W = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    cx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1], dtype=np.float64)
    cy = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1], dtype=np.float64)
    bb_idx = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4], dtype=np.int64)

    # Initialisation des fonctions de distribution
    N = np.outer(np.ones(NX * NY), rho0 * W)
    FlowRate_old = 1.0
    FlowRate = 0.0
    t_ = 1

    # Boucle temporelle jusqu'à atteindre le régime stationnaire
    while FlowRate == 0.0 or abs(FlowRate_old - FlowRate) / abs(FlowRate) >= epsilon:
        N, ux, FlowRate_new = _lbm_step(
            N, SOLID, W, cx, cy, NX, NY, deltaP, dx, rho0, dt, OMEGA, bb_idx
        )
        FlowRate_old = FlowRate
        FlowRate = FlowRate_new
        t_ += 1

    # Calcul de la perméabilité via la loi de Darcy
    u_mean = ux[:NY].mean()
    k = u_mean * mu / deltaP * (NX * dx) * 1e12 # Perméabilité en µm²

    return k
