# %%
import sys
import numpy as np
# --- Add rootDir (C:\python) to sys.path ---
rootDir = r"C:\python"
if rootDir not in sys.path:
    sys.path.insert(0, rootDir)
# --- Now import the function ---
from shared_functions.draw_beam_bending import draw_beam_bending
# --- Beam definition ---
L = 6.0
h = 0.3
b = 0.25
density = 2500        # concrete
E = 30e9              # concrete ~30 GPa
q_ext_kN_per_m = 7.0  # extra dead load  kN/m
point_loads = [(15.0, 3.0)]  # single 15 kN at midspan
n_points = 600
scale = 1000.0
figsize = (10, 3)
# --- Run beam bending ---
x, deflection = draw_beam_bending(
    L=L, h=h, b=b, density=density, g=9.81, E=E,
    q_ext_kN_per_m=q_ext_kN_per_m, point_loads=point_loads,
    n_points=n_points, scale=scale, draw_deformed_outline=True,
    figsize=figsize
)
# --- Results ---
print("Max deflection (m):", np.max(deflection))
print("Max deflection (mm):", np.max(deflection) * 1000.0)


# %%
