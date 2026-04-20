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
import lhsmdu

#from projet_levesque.levesque_analyse import parametres
class parametres():
    C0 = 1     #[mol]
    u_max = 1  #[m/s]
    L = 10      #[m]
    H = 10      #[m]
    P = 1      #[m]
    Pe = 50
    Da = 5
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


#ordre observé avec formule itérative
def ordre_iteratif(f1, f2, f3, r, p0=2, tol=1e-6, max_iter=50):
    p = p0

    for k in range(max_iter):
        num = (r**p - 1)*((f3 - f2)/(f2 - f1)) + r**p
        p_new = np.log(abs(num))/np.log(r*r)

        if abs(p_new - p) < tol:
            return p_new
        p = p_new

    return p  # si pas convergé

#Propagation des incertitudes
# Fonction pour générer un échantillonnage Latin Hypercube pour les paramètres aléatoires (C0, u_max, L, H) avec des distributions gaussiennes
def lhs_norm_uni(N,prm_base, seed=42):
    """
    Génère un échantillonnage Latin Hypercube pour:
    - Gaussien: C0, u_max, L, H
    """
    d = 4  # nr. variables aléatoires (C0, u_max, L, H)
    lhs = np.array(lhsmdu.sample(d, N, randomSeed=seed)).T
    # if lhs.shape[0] == 4:
    #     lhs = lhs.T
    eps = 1e-12
    lhs = np.clip(lhs, eps, 1 - eps)
        
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

def mc_sampling(N, prm_base, seed=42):
    '''
    Génère un échantillonnage Monte-Carlo pour les paramètres aléatoires 
    (C0, u_max, L, H) avec des distributions gaussiennes.'
        C0 +/- 3%,
        u_max +/- 5%,
        L +/- 1%,
        H +/- 1%
    '''
    rng = np.random.default_rng(seed)

    C0 = rng.normal(prm_base.C0,    0.03 * prm_base.C0,    size=N)
    u  = rng.normal(prm_base.u_max, 0.05 * prm_base.u_max, size=N)
    L  = rng.normal(prm_base.L,     0.01 * prm_base.L,     size=N)
    H  = rng.normal(prm_base.H,     0.01 * prm_base.H,     size=N)

    return (
        np.maximum(C0, 1e-12),
        np.maximum(u,  1e-12),
        np.maximum(L,  1e-12),
        np.maximum(H,  1e-12),
    )

def int_aleatory_ep_fix(prm_fixed, N, nx, ny, seed = 42, method = "MC"):
    """
    Monte-Carlo aléatoire pour des paramètres épistémiques fixés (Pe, Da).
    Retourne les valeurs de Q triées et la grille de CDF correspondante.
    Paramètres de LHS ou MC selon le paramètre 'method' pour les paramètres aléatoires (C0, u_max, L, H) avec des distributions gaussiennes.
    """
    # --- choose aleatory sampling ---
    if method.upper() == "MC":
        C0_vals, umax_vals, L_vals, H_vals = mc_sampling(N, prm_fixed, seed)

    elif method.upper() == "LHS":
        C0_vals, umax_vals, L_vals, H_vals = lhs_norm_uni(N, prm_fixed, seed)
        # --- evaluate Qc for each sample ---
    Q = np.zeros(N)
    for i in range(N):
        prm_i = parametres()
        prm_i.Pe     = prm_fixed.Pe
        prm_i.Da     = prm_fixed.Da
        prm_i.C0     = C0_vals[i]
        prm_i.u_max  = umax_vals[i]
        prm_i.L      = L_vals[i]
        prm_i.H      = H_vals[i]

        Q[i] = Q_c_simpson(nx, ny, prm_i, ordre=2, mms=False)

    # CDF components
    Q_sorted = np.sort(Q)
    CDF = np.linspace(1/N, 1, N)

    return Q_sorted, CDF, {
        "Q": Q,
        "C0": C0_vals,
        "u_max": umax_vals,
        "L": L_vals,
        "H": H_vals,
    }

    # np.random.seed(seed)
    # Q_vals = []
    # C0_vals = []
    # umax_vals = []
    # L_vals = []
    # H_vals = []

    # for i in range(N):

    #     prm_m = parametres()

    #     # epistemic - paramètres fixés
    #     prm_m.Pe = prm_fixed.Pe
    #     prm_m.Da = prm_fixed.Da

    #     # Aleatory (LHS)
    #     prm_m.C0 = C0_lhs[i]
    #     prm_m.u_max = u_max_lhs[i]
    #     prm_m.L = L_lhs[i]
    #     prm_m.H = H_lhs[i]

    #     q = Q_c_simpson(nx, ny, prm_m, ordre=2, mms=False) #NEW
    #     Q_vals.append(q) #NEW
    #     # évaluer Qc et stocker les résultats
    #     C0_vals.append(prm_m.C0)
    #     umax_vals.append(prm_m.u_max)
    #     L_vals.append(prm_m.L)
    #     H_vals.append(prm_m.H)
    #     #Q_vals.append(Q_c_simpson(nx, ny, prm_m, ordre=2, mms=False)) IF NEW PART DOESN*T WORK, UNCOMMENT

    # #Q_vals = np.sort(np.array(Q_vals)) IF NEW PART DOESN*T WORK, UNCOMMENT
 
    # # CDF = np.linspace(0, 1, len(Q_vals)) IF THE PART BELLLOW DOESN*T WORK; THIS IS FROM THE ORIGINAL!!
    # Q_arr    = np.array(Q_vals)          # ← UNSORTED: keeps alignment with samples
    # Q_sorted = np.sort(Q_arr)            #   sorted copy only for the CDF
    # CDF      = np.linspace(0, 1, N)

    # samples = {
    #     "Q":     Q_arr,                  # aligned, unsorted
    #     "C0":    np.array(C0_vals),
    #     "u_max": np.array(umax_vals),
    #     "L":     np.array(L_vals),
    #     "H":     np.array(H_vals),
    # }
    # return Q_sorted, CDF, samples

def pbox(prm_base, N=200, nx=129, ny=129, seed=42, method="MC"):
    """
    Calculations de l'enveloppe des CDFs de Qc pour les différentes combinaisons extrêmes de Pe et Da, en utilisant Monte-Carlo pour les paramètres aléatoires.
    Epistemic: Pe, Da (intervals)
    Aleatoire: C0, u_max, L, H (Gaussian)
    Method supporté: "MC" ou "LHS" pour les paramètres aléatoires. Par défaut "MC".
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

    all_Q_sorted = []
    all_F = []
    pooled = {k: [] for k in ("Q", "C0", "u_max", "L", "H")}  # NEW pour stocker tous les échantillons aléatoires de tous les cas épistémiques

    for i, (label, Pe_val, Da_val) in enumerate(epistemic_cases):
        print(f"\n=== Epistemic case: {label} ===")

        prm_fixed        = parametres()
        prm_fixed.C0     = prm_base.C0
        prm_fixed.u_max  = prm_base.u_max
        prm_fixed.L      = prm_base.L
        prm_fixed.H      = prm_base.H
        prm_fixed.Pe     = Pe_val
        prm_fixed.Da     = Da_val

        Qs, Fs, samples = int_aleatory_ep_fix(prm_fixed, N, nx, ny, seed + i, method=method)
        all_Q_sorted.append(Qs)
        all_F.append(Fs)

        for k in pooled:               # pool aligned samples for global SA
            pooled[k].append(samples[k])

    # convert pooled lists to arrays
    sens_data = {k: np.concatenate(pooled[k]) for k in pooled}

    # Q_grid commun pour l'interpolation des CDFs
    Q_min = min(Q[0] for Q in all_Q_sorted)  ##prev. all_Q ,   min des premiers éléments (min de chaque CDF)
    Q_max = max(Q[-1] for Q in all_Q_sorted) ##prev. all_Q ,   max des derniers éléments (max de chaque CDF)
    Q_grid = np.linspace(Q_min, Q_max, 400)

    F_min = np.ones_like(Q_grid)
    F_max = np.zeros_like(Q_grid)

    # Construction de l'enveloppe des CDFs (P-box)
    for Q_vals, F_vals in zip(all_Q_sorted, all_F):
        F_interp = np.interp(Q_grid, Q_vals, F_vals)
        F_min = np.minimum(F_min, F_interp)
        F_max = np.maximum(F_max, F_interp)

    # P-box plot
    plt.figure(figsize=(8, 5))
    plt.fill_between(Q_grid, F_min, F_max, color="lightgray", label="P-box")
    plt.plot(Q_grid, F_min, lw=2, color="green", label="F_min")
    plt.plot(Q_grid, F_max, lw=2, color="red",   label="F_max")
    plt.xlabel("Qc (mol/m²)")
    plt.ylabel("CDF")
    plt.grid(alpha=0.3)
    plt.title("P-box of Qc  (epistemic × aleatory uncertainty)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "Q_grid":    Q_grid,
        "F_min":     F_min,
        "F_max":     F_max,
        "sens_data": sens_data,   # pooled, aligned samples from all epistemic cases
    }

def plot_pdf_cdf(vals, var_name="Variable", method="", bins=30):
    """
    Plot PDF and CDF for any input variable.
    
    Parameters
    ----------
    vals : array-like
        Values to plot
    var_name : str
        Name of the variable (C0, u_max, L, H, Qc, etc.)
    method : str
        Sampling method label ("MC", "LHS", etc.)
    """
    data = np.array(vals)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # PDF
    ax[0].hist(data, bins=bins, density=True, color="steelblue", alpha=0.7)
    ax[0].set_title(f"PDF of {var_name} ({method})")
    ax[0].set_xlabel(var_name)
    ax[0].set_ylabel("PDF")
    ax[0].grid(alpha=0.3)

    # CDF
    data_sorted = np.sort(data)
    F = np.linspace(0, 1, len(data_sorted))

    ax[1].plot(data_sorted, F, lw=2, color="darkorange")
    ax[1].set_title(f"CDF of {var_name} ({method})")
    ax[1].set_xlabel(var_name)
    ax[1].set_ylabel("CDF")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)

#COMPLETLEY NEW AND UNVERIFIED
# Global Sensitivity Analysis:
def global_sensitivity_analysis(pbox_results):
    """
    MEC8211-compatible Global Sensitivity Analysis
    ---------------------------------------------------
    Methods used:
        - Spearman rank correlation  (Oberkampf & Roy)
        - SRC : Standardized Regression Coefficients
        - SRRC: Standardized Rank Regression Coefficients (rank-based SRC)

    Inputs
        pbox_results["sens_data"] : pooled aleatory samples across epistemic cases
    """

    from scipy.stats import spearmanr, rankdata
    from numpy.linalg import lstsq

    sens_data = pbox_results["sens_data"]
    Q = sens_data["Q"]
    params = ["C0", "u_max", "L", "H"]
    labels = {"C0": r"$C_0$", "u_max": r"$u_{\max}$", "L": r"$L$", "H": r"$H$"}

    # Spearman rank correlations
    spearman_r = {}
    for p in params:
        rho, _ = spearmanr(sens_data[p], Q)
        spearman_r[p] = rho

    # SRC
    X = np.column_stack([sens_data[p] for p in params])
    X_std = (X - X.mean(0)) / X.std(0)
    Q_std = (Q - Q.mean()) / Q.std()

    beta_SRC, _, _, _ = lstsq(X_std, Q_std, rcond=None)
    SRC = dict(zip(params, beta_SRC))

    # SRRC (rank-based SRC)
    Xr = np.column_stack([rankdata(sens_data[p]) for p in params])
    Xr_std = (Xr - Xr.mean(0)) / Xr.std(0)
    Qr_std = (rankdata(Q) - np.mean(rankdata(Q))) / np.std(rankdata(Q))

    beta_SRRC, _, _, _ = lstsq(Xr_std, Qr_std, rcond=None)
    SRRC = dict(zip(params, beta_SRRC))

    # Plots
    fig, axes = plt.subplots(1, len(params), figsize=(14, 4), sharey=True)
    fig.suptitle("Global SA — Spearman + SRC + SRRC", fontsize=12, fontweight="bold")

    for ax, p in zip(axes, params):
        ax.scatter(sens_data[p], Q, s=10, alpha=0.35, color="steelblue")
        ax.set_xlabel(labels[p], fontsize=11)
        ax.set_title(f"$\\rho$={spearman_r[p]:+.3f}\nSRC={SRC[p]:+.3f}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$Q_c$  (mol/m²)")

    plt.tight_layout()
    plt.show()

    # ───────────── Summary bar chart ─────────────
    width = 0.25
    x_pos = np.arange(len(params))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos - width, [spearman_r[p] for p in params],
           width, label="Spearman ρ", color="darkorange", alpha=0.85)
    ax.bar(x_pos, [SRC[p] for p in params],
           width, label="SRC", color="steelblue", alpha=0.85)
    ax.bar(x_pos + width, [SRRC[p] for p in params],
           width, label="SRRC", color="seagreen", alpha=0.85)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[p] for p in params], fontsize=11)
    ax.set_ylabel("Sensitivity coefficient")
    ax.set_title("Global Sensitivity Analysis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ───────────── Console output ─────────────
    print("\n─── Global Sensitivity Analysis (MEC8211) ───")
    print(f"{'Param':<8} {'Spearman':>10}  {'SRC':>10}  {'SRRC':>10}")
    print("-" * 50)
    for p in params:
        print(f"{p:<8} {spearman_r[p]:>+10.4f}  {SRC[p]:>+10.4f}  {SRRC[p]:>+10.4f}")

    return {"spearman": spearman_r, "SRC": SRC, "SRRC": SRRC} 

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
