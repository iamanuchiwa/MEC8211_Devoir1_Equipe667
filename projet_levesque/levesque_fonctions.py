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
from scipy.stats import norm, uniform
from torch import seed

#from projet_levesque.levesque_analyse import parametres
class parametres():
    C0 = 1     #[mol]
    u_max = 1  #[m/s]
    L = 10      #[m]
    H = 10      #[m]
    P = 1      #[m]
    Pe = 50
    Da = 0.5
prm = parametres()

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

#Propagation des incertitudes
def monte_carlo_Qc(prm_base, N=300, nx=129, ny=129, seed=42, plot_pdfs=True, plot_cdf=True):
    """
    Monte-Carlo propagation pour évaluer l'incertitude sur la quantité totale de matière adsorbée Qc
    Paramètres d'entrée:
    prm_base : parametres
        Paramètre de base pour la simulation, contenant les valeurs moyennes des paramètres d'entrée.
        Utilise prm dans analyse
    N : int
        Nr de simulations Monte-Carlo à effectuer.
    nx, ny : int
        resolution spatiale pour chaque pde
    seed : int
        Random seed
    plot_pdfs : bool
        Si True, affiche les PDF des paramètres d'entrée utilisés dans la simulation Monte-Carlo
    plot_cdf : bool
        Si True, affiche la CDF des paramètres d'entrée utilisés dans la simulation Monte-Carlo

    retourns:
    results : dict
        archive des résultats de la simulation Monte-Carlo, contenant les valeurs de Qc et les échantillons des paramètres
        d'entrée pour l'analyse de sensibilité.
    """

    np.random.seed(seed)

    # Stockage des résultats
    Q_vals = []
    C0_vals = []
    umax_vals = []
    L_vals = []
    H_vals = []
    Pe_vals = []
    Da_vals = []

    for _ in range(N):

        # parametres pour cette simulation Monte-Carlo
        prm_m = parametres()

        #  incertitudes aléatoires (gaussian)
        prm_m.C0     = np.random.normal(prm_base.C0,   0.03 * prm_base.C0)      # ±3%
        prm_m.u_max  = np.random.normal(prm_base.u_max,0.05 * prm_base.u_max)   # ±5%
        prm_m.L      = np.random.normal(prm_base.L,    0.01 * prm_base.L)       # ±1%
        prm_m.H      = np.random.normal(prm_base.H,    0.01 * prm_base.H)       # ±1%

        #  incertitudes épistémiques (uniforme)
        prm_m.Pe = np.random.uniform(0.9*prm_base.Pe, 1.1*prm_base.Pe)  # ±10% interval
        prm_m.Da = np.random.uniform(0.9*prm_base.Da, 1.1*prm_base.Da)  # ±10% interval

        #  Calcul de Qc pour cette simulation Monte-Carlo
        Q_val = Q_c_simpson(nx, ny, prm_m, ordre=2, mms=False)
        Q_vals.append(Q_val)

        # Sauvegarde des échantillons de paramètres pour l'analyse de sensibilité
        C0_vals.append(prm_m.C0)
        umax_vals.append(prm_m.u_max)
        L_vals.append(prm_m.L)
        H_vals.append(prm_m.H)
        Pe_vals.append(prm_m.Pe)
        Da_vals.append(prm_m.Da)

    Q_vals = np.array(Q_vals)
    # Plot PDFs d'entrée
    if plot_pdfs:
        plot_input_pdfs(
            C0_vals, prm_base.C0, 0.03*prm_base.C0,
            umax_vals, prm_base.u_max, 0.05*prm_base.u_max,
            L_vals, prm_base.L, 0.01*prm_base.L,
            H_vals, prm_base.H, 0.01*prm_base.H,
            Pe_vals, 0.9*prm_base.Pe, 1.1*prm_base.Pe,
            Da_vals, 0.9*prm_base.Da, 1.1*prm_base.Da
        )
    # Plot CDFs d'entrée
    if plot_cdf:
        plot_input_cdfs(C0_vals, umax_vals, L_vals, H_vals, Pe_vals, Da_vals)

    # présentation des résultats de la simulation Monte-Carlo
    print("\n----- MONTE-CARLO RESULTS -----")
    print(f"Mean Qc      : {Q_vals.mean():.6e}")
    print(f"Std dev Qc   : {Q_vals.std():.6e}")
    print(f"Rel. uncert. : {100*Q_vals.std()/Q_vals.mean():.2f} %")
    print(f"95% CI       : [{np.percentile(Q_vals,2.5):.6e} ; {np.percentile(Q_vals,97.5):.6e}]")

    # Plot CDF de Qc
    sorted_Q = np.sort(Q_vals)
    cdf = np.linspace(0, 1, N)

    plt.figure(figsize=(8,5))
    plt.plot(sorted_Q, cdf, lw=2)
    plt.xlabel("Qc (mol/m²)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.title("CDF of Qc from Monte-Carlo Propagation")
    plt.tight_layout()
    plt.show()

    return {
        "Q": Q_vals,
        "C0": np.array(C0_vals),
        "u_max": np.array(umax_vals),
        "L": np.array(L_vals),
        "H": np.array(H_vals),
        "Pe": np.array(Pe_vals),
        "Da": np.array(Da_vals)
    }

# Fonction pour générer un échantillonnage Latin Hypercube pour les paramètres aléatoires (C0, u_max, L, H) avec des distributions gaussiennes
def lhs_norm_uni(N,prm_base, seed=42):
    """
    Génère un échantillonnage Latin Hypercube pour:
    - Gaussien: C0, u_max, L, H
    """
    np.random.seed(seed)
    d = 4  # nr. variables aléatoires (C0, u_max, L, H)
    lhs = np.zeros((N, d))
    
    for j in range(d):
        perm = np.random.permutation(N)
        lhs[:, j] = (perm + np.random.rand(N)) / N
        
    # Transformation inverse pour les distributions gaussiennes

    C0_lhs = norm.ppf(lhs[:, 0], prm_base.C0, 0.03 * prm_base.C0)
    u_max_lhs = norm.ppf(lhs[:, 1], prm_base.u_max, 0.05 * prm_base.u_max)
    L_lhs = norm.ppf(lhs[:, 2], prm_base.L, 0.01 * prm_base.L)
    H_lhs = norm.ppf(lhs[:, 3], prm_base.H, 0.01 * prm_base.H)
    
    # Securisation pour éviter les valeurs négatives (en cas de queue de distribution)
    C0_lhs = np.maximum(C0_lhs, 1e-12)
    u_max_lhs = np.maximum(u_max_lhs, 1e-12)
    L_lhs = np.maximum(L_lhs, 1e-12)
    H_lhs = np.maximum(H_lhs, 1e-12)
    
    return C0_lhs, u_max_lhs, L_lhs, H_lhs

# def plot_input_pdfs(C0_samples, C0_mu, C0_sigma, umax_samples, umax_mu, umax_sigma, L_samples, L_mu, L_sigma, H_samples, H_mu, H_sigma, Pe_samples, Pe_min, Pe_max, Da_samples, Da_min, Da_max):
    #Plot pdfs des entrées (gaussian pour les paramètres aléatoires et uniforme pour les paramètres épistémiques)
    #Utilisé dans Monte-Carlo pour visualiser les distributions d'entrée

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.flatten()

    def plot_gaussian(ax, samples, mu, sigma, label):
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        ax.hist(samples, bins=20, density=True, alpha=0.5)
        ax.plot(x, norm.pdf(x, mu, sigma), 'r', lw=2)
        ax.set_title(label)
        ax.grid(alpha=0.3)

    def plot_uniform(ax, samples, a, b, label):
        x = np.linspace(a, b, 200)
        ax.hist(samples, bins=20, density=True, alpha=0.5)
        ax.plot(x, uniform.pdf(x, a, b - a), 'r', lw=2)
        ax.set_title(label)
        ax.grid(alpha=0.3)

    plot_gaussian(axs[0], C0_samples,    C0_mu,  C0_sigma,  "C0 (Gaussian)")
    plot_gaussian(axs[1], umax_samples,  umax_mu,umax_sigma,"u_max (Gaussian)")
    plot_gaussian(axs[2], L_samples,     L_mu,  L_sigma,   "L (Gaussian)")
    plot_gaussian(axs[3], H_samples,     H_mu,  H_sigma,   "H (Gaussian)")
    plot_uniform(axs[4], Pe_samples, Pe_min, Pe_max, "Pe (Uniform)")
    plot_uniform(axs[5], Da_samples, Da_min, Da_max, "Da (Uniform)")

    plt.tight_layout()
    plt.show()

# def plot_input_cdfs(C0_vals, umax_vals, L_vals, H_vals, Pe_vals, Da_vals):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.flatten()

    def _plot(ax, samples, label):
        samples = np.array(samples)
        sorted_s = np.sort(samples)
        cdf = np.linspace(0, 1, len(samples))
        ax.plot(sorted_s, cdf, lw=2)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.3)

    _plot(axs[0], C0_vals, "C0")
    _plot(axs[1], umax_vals, "u_max")
    _plot(axs[2], L_vals, "L")
    _plot(axs[3], H_vals, "H")
    _plot(axs[4], Pe_vals, "Pe")
    _plot(axs[5], Da_vals, "Da")

    plt.tight_layout()
    plt.show()

def int_aleatory_MC_ep_fix(prm_fixed, N, nx, ny, seed):
    """
    Monte-Carlo aléatoire pour des paramètres épistémiques fixés (Pe, Da).
    Retourne les valeurs de Q triées et la grille de CDF correspondante.
    Paramètres de LHS pour les paramètres aléatoires (C0, u_max, L, H) avec des distributions gaussiennes.
    """
    C0_lhs, u_max_lhs, L_lhs, H_lhs = lhs_norm_uni(N, prm_fixed, seed)

    np.random.seed(seed)
    Q_vals = []
    C0_vals = []
    umax_vals = []
    L_vals = []
    H_vals = []

    for i in range(N):

        prm_m = parametres()

        # epistemic - paramètres fixés
        prm_m.Pe = prm_fixed.Pe
        prm_m.Da = prm_fixed.Da

        # # aléatoire: Gaussian
        # prm_m.C0     = np.random.normal(prm_fixed.C0,   0.03 * prm_fixed.C0)
        # prm_m.u_max  = np.random.normal(prm_fixed.u_max,0.05 * prm_fixed.u_max)
        # prm_m.L      = np.random.normal(prm_fixed.L,    0.01 * prm_fixed.L)
        # prm_m.H      = np.random.normal(prm_fixed.H,    0.01 * prm_fixed.H)

        # Aleatory (LHS)
        prm_m.C0 = C0_lhs[i]
        prm_m.u_max = u_max_lhs[i]
        prm_m.L = L_lhs[i]
        prm_m.H = H_lhs[i]

        # évaluer Qc et stocker les résultats
        C0_vals.append(prm_m.C0)
        umax_vals.append(prm_m.u_max)
        L_vals.append(prm_m.L)
        H_vals.append(prm_m.H)
        Q_vals.append(Q_c_simpson(nx, ny, prm_m, ordre=2, mms=False))

    Q_vals = np.sort(np.array(Q_vals))

    CDF = np.linspace(0, 1, len(Q_vals))

    return Q_vals, CDF, {
    "C0": np.array(C0_vals),
    "u_max": np.array(umax_vals),
    "L": np.array(L_vals),
    "H": np.array(H_vals)
    }

def pbox(prm_base, N=200, nx=129, ny=129, seed=42):
    """
    Calculations de l'enveloppe des CDFs de Qc pour les différentes combinaisons extrêmes de Pe et Da, en utilisant Monte-Carlo pour les paramètres aléatoires.
    Epistemic: Pe, Da (intervals)
    Aleatoire: C0, u_max, L, H (Gaussian)
    """

    # epistemic bounds - pm 10%
    Pe_min, Pe_max = 0.9*prm_base.Pe, 1.1*prm_base.Pe
    Da_min, Da_max = 0.9*prm_base.Da, 1.1*prm_base.Da

    epistemic_cases = [
        ("Pe_min, Da_min", Pe_min, Da_min),
        ("Pe_min, Da_max", Pe_min, Da_max),
        ("Pe_max, Da_min", Pe_max, Da_min),
        ("Pe_max, Da_max", Pe_max, Da_max),
    ]

    all_Q = []
    all_F = []

    for label, Pe_val, Da_val in epistemic_cases:
        print(f"\n=== Epistemic case: {label} ===")

        prm_fixed = parametres()
        prm_fixed.C0    = prm_base.C0
        prm_fixed.u_max = prm_base.u_max
        prm_fixed.L     = prm_base.L
        prm_fixed.H     = prm_base.H
        prm_fixed.Pe    = Pe_val
        prm_fixed.Da    = Da_val

        Q_vals, F_vals, samples = int_aleatory_MC_ep_fix(
            prm_fixed, N, nx, ny, seed
        )

        all_Q.append(Q_vals)
        all_F.append(F_vals)

    # Q_grid commun pour l'interpolation des CDFs
    Q_min = min(Q[0] for Q in all_Q)
    Q_max = max(Q[-1] for Q in all_Q)
    Q_grid = np.linspace(Q_min, Q_max, 400)

    F_min = np.ones_like(Q_grid)
    F_max = np.zeros_like(Q_grid)

    # Construction de l'enveloppe des CDFs (P-box)
    for Q_vals, F_vals in zip(all_Q, all_F):
        F_interp = np.interp(Q_grid, Q_vals, F_vals)
        F_min = np.minimum(F_min, F_interp)
        F_max = np.maximum(F_max, F_interp)

    if label == "Pe_mid, Da_mid":  # or just first case
        sens_data = samples
        sens_Q = Q_vals

    # --- Plot
    plt.figure(figsize=(8,5))
    plt.fill_between(Q_grid, F_min, F_max, color='lightgray', label="P-box")
    plt.plot(Q_grid, F_min, 'k', lw=2, label="F_min", color='green')
    plt.plot(Q_grid, F_max, 'k', lw=2, label="F_max", color='red')
    plt.xlabel("Qc (mol/m²)")
    plt.ylabel("CDF")
    plt.grid(alpha=0.3)
    plt.title("P-box of Qc (epistemic + aleatory)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"Q_grid": Q_grid, "F_min": F_min, "F_max": F_max, "sens_Q": sens_Q, "sens_data": sens_data}

#Validation
def Q_c_empirique(prm):
    '''
    Calcule la quantité totale de matière adsorbée Qc (mol/m^2)
    à partir de la relation empirique fournie dans l'énoncé.
    Valide uniquement pour Pe > 100.
    '''
    # Calcul de Ds à partir de Pe
    Ds = prm.u_max * prm.H / prm.Pe

    # Intégrale de la solution empirique:
    # Qc_emp = C0 * 0.854 * (u_max * H^2 / Ds)**(1/3) * (3/2) * L**(2/3)
    terme1 = 0.854 * (prm.u_max * (prm.H**2) / Ds)**(1/3)
    terme2 = 1.5 * (prm.L**(2/3))

    Qc_emp = prm.C0 * terme1 * terme2
    return Qc_emp
