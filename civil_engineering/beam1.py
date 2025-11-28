
import numpy as np
def beam1(L, q, E, I, n=50):
    """Uniformly loaded simply supported beam.
    Parameters:
        L (float): beam length [m]
        q (float): distributed load [kN/m]
        E (float): Young's modulus [Pa]
        I (float): second moment of area [m^4]
        n (int): number of points
    Returns:
        x (ndarray): positions along beam [m]
        y (ndarray): deflections [m]
"""
    # Convert kN/m to N/m
    q = q * 1000  
    # Convert m to cm
    L= L * 100
# Discretize beam
    x = np.linspace(0, L, n)
    # Deflection formula
    y = (q * x * (L**3 - 2*L*x**2 + x**3)) / (24 * E * I)
    return x, y

