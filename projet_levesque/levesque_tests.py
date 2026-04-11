import pytest
import numpy as np
from levesque_fonctions import *

# --- FIXTURES ---
@pytest.fixture
def prm_base():
    """Paramètres de base utilisés pour la majorité des tests."""
    prm = parametres()
    prm.C0 = 1.0
    prm.L = 10.0
    prm.H = 10.0
    prm.Pe = 50.0
    prm.Da = 0.5
    prm.u_max = 1.0
    return prm


# TESTS PARAMÉTRÉS (Ordre 1 et Ordre 2) 

@pytest.mark.parametrize("ordre", [1, 2])
def test_concentration_positive(prm_base, ordre):
    """Vérifie que la concentration ne devient jamais négative."""
    nx, ny = 21, 21
    C = concentration(nx, ny, prm_base, ordre=ordre, mms=False)
    
    # On tolère une petite valeur négative due aux erreurs d'arrondi machine
    assert np.all(C >= -1e-10), f"Concentrations négatives trouvées avec l'ordre {ordre}."

@pytest.mark.parametrize("ordre", [1, 2])
def test_condition_dirichlet_gauche(prm_base, ordre):
    """Vérifie que le bord gauche respecte C = C0."""
    nx, ny = 15, 15
    C = concentration(nx, ny, prm_base, ordre=ordre, mms=False)
    C_2D = C.reshape((ny, nx))
    
    # Exclure les coins (y=0 et y=H)
    bord_gauche = C_2D[1:-1, 0]
    
    np.testing.assert_allclose(bord_gauche, prm_base.C0, atol=1e-10, 
                               err_msg=f"Dirichlet non respecté à gauche pour l'ordre {ordre}.")

@pytest.mark.parametrize("ordre", [1, 2])
def test_condition_neumann_droite(prm_base, ordre):
    """Vérifie que dC/dx = 0 au bord droit (x=L)."""
    
    nx, ny = 15, 15
    dx = prm_base.L / (nx - 1)
    C = concentration(nx, ny, prm_base, ordre=ordre, mms=False)
    C_2D = C.reshape((ny, nx))
    
    C_n = C_2D[1:-1, -1]
    C_n1 = C_2D[1:-1, -2]
    C_n2 = C_2D[1:-1, -3]
    
    derivee_x = (3*C_n - 4*C_n1 + C_n2) / (2*dx)
    np.testing.assert_allclose(derivee_x, 0, atol=1e-10, 
                               err_msg=f"Neumann (dC/dx=0) échoue à droite pour l'ordre {ordre}.")

@pytest.mark.parametrize("ordre", [1, 2])
def test_condition_neumann_haut(prm_base, ordre):
    """Vérifie que dC/dy = 0 au bord supérieur (y=H)."""
    nx, ny = 15, 15
    dy = prm_base.H / (ny - 1)
    C = concentration(nx, ny, prm_base, ordre=ordre, mms=False)
    C_2D = C.reshape((ny, nx))
    
    C_n = C_2D[-1, :]
    C_n1 = C_2D[-2, :]
    C_n2 = C_2D[-3, :]
    
    derivee_y = (3*C_n - 4*C_n1 + C_n2) / (2*dy)
    np.testing.assert_allclose(derivee_y, 0, atol=1e-10, 
                               err_msg=f"Neumann (dC/dy=0) échoue en haut pour l'ordre {ordre}.")

@pytest.mark.parametrize("ordre", [1, 2])
def test_integration_simpson_logique(prm_base, ordre):
    """Vérifie que Qc calculé n'est pas nul et est bien positif."""
    nx, ny = 21, 21
    Qc = Q_c_simpson(nx, ny, prm_base, ordre=ordre, mms=False)
    assert Qc > 0, f"La quantité adsorbée Qc devrait être positive (Ordre {ordre})."

@pytest.mark.parametrize("ordre", [1, 2])
def test_validation_mms(prm_base, ordre):
    """
    Compare la solution numérique MMS à la solution analytique exacte.
    
    """
    nx, ny = 41, 41 
    
    C_num = concentration(nx, ny, prm_base, ordre=ordre, mms=True)
    
    x_vect = np.linspace(0, prm_base.L, nx)
    y_vect = np.linspace(0, prm_base.H, ny)
    X, Y = np.meshgrid(x_vect, y_vect)
    
    fonc_mms = genere_mms(prm_base)
    C_exacte_2D = fonc_mms[0](X, Y)
    C_exacte = C_exacte_2D.flatten()
    
    erreur_max = np.max(np.abs(C_num - C_exacte))
    
    # Tolérances différentes selon l'ordre du schéma
    tolerance = 0.5 if ordre == 1 else 0.05
    assert erreur_max < tolerance, f"Erreur MMS trop élevée pour l'ordre {ordre} (Erreur: {erreur_max})"

# --- TESTS SPÉCIFIQUES ---

def test_assertions_dimensions(prm_base):
    """Vérifie que le code rejette les maillages trop petits."""
    with pytest.raises(AssertionError, match="doit être supérieur ou égal à 3"):
        concentration(2, 5, prm_base)
        
    with pytest.raises(AssertionError, match="doit être supérieur ou égal à 3"):
        concentration(5, 2, prm_base)

def test_assertion_simpson(prm_base):
    """Vérifie que l'intégration de Simpson rejette les maillages pairs en x."""
    with pytest.raises(AssertionError, match="doit être impair"):
        Q_c_simpson(10, 11, prm_base) 

# --- POINT D'ENTRÉE POUR LANCER LES TESTS FACILEMENT ---
if __name__ == "__main__":
    import sys
    # Lance pytest sur ce fichier de manière verbeuse (-v)
    sys.exit(pytest.main(["-v", __file__]))