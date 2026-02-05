import numpy as np
import matplotlib.pyplot as plt
import math
y=7
x = np.linspace(-10, 10, 400)

# --- Paramètres Physiques ---
Deff = 1e-10        # Coefficient de diffusion effectif [m^2/s]
S = 2e-8            # Terme source constant [mol/m^3/s]
Ce = 20.0           # Concentration à la surface [mol/m^3]
R_pilier = 0.5      # Rayon du pilier (Diamètre = 1m)

# --- Solution Analytique (Eq. 2) ---
def solution_analytique(r):
    #C(r) = (S / 4Deff) * R^2 * ((r^2/R^2) - 1) + Ce
    terme1 = (S / (4 * Deff)) * (R_pilier**2)
    terme2 = ((r**2) / (R_pilier**2)) - 1
    return terme1 * terme2 + Ce