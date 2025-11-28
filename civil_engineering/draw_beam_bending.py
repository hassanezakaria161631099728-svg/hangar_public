"""
High-quality 2D beam drawing + bending under self-weight + extra loads.
Units: SI
- Lengths: meters (m)
- Density: kg/m^3
- E (Young's modulus): Pa (N/m^2)
- I (second moment): m^4 (if not provided, computed for rectangular cross-section as b*h^3/12)
- Uniform external load q_ext given in kN/m (converted to N/m)
- Point loads list: [(P_kN, a_m), ...] where a_m is distance from left support
Returns: (x, w) arrays of positions and deflections in meters
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def draw_beam_bending(L,h,b,density,E,I=None,q_ext_kN_per_m=None,point_loads=None,g=9.81,n_points=None,scale=None,draw_deformed_outline=None,figsize=None):
#    L=10.0,                # beam length (m)
#    h=0.3,                 # beam height (m) (rectangular cross-section thickness in vertical)
#    b=0.3,                 # beam width (m) (depth into page; used for I if I not provided)
#    density=2500,          # kg/m^3 (concrete ≈ 2400-2500)
#    E=30e9,                # Pa (concrete ~30e9; steel ~210e9)
#    I=None,                # m^4 (if None, use rectangle I = b*h^3/12)
#    q_ext_kN_per_m=0.0,    # extra uniform load (kN/m)
#    point_loads=None,      # list of tuples [(P_kN, a_m), ...]
#    g=9.81,                # gravity m/s^2
#    n_points=400,          # discretization points
#    scale=1000.0,          # scale factor for plotting deflection (visualization: multiply deflection by scale)
#    draw_deformed_outline=True,  # draw deformed top/bottom edges (approx)
#    figsize=(10,3)         # figure size
    if point_loads is None:
        point_loads = []
    # --- Units conversions and geometry ---
    # area (m^2)
    area = b * h
    # second moment (m^4) if not provided
    if I is None:
        I = b * h**3 / 12
    # Self-weight as uniform load (N/m)
    q_self = density * area * g      # N/m
    # External uniform load convert kN/m -> N/m
    q_ext = q_ext_kN_per_m * 1000  # N/m
    # Total uniform load
    q_total = q_self + q_ext
    # Convert point loads to N and collect positions
    point_loads_N = [(P_kN * 1000.0, a_m) for (P_kN, a_m) in point_loads]
    # --- Spatial discretization ---
    x = np.linspace(0.0, L, n_points)
    # --- Reactions (simply supported) ---
    # Reaction from distributed load:
    R_dist_left = q_total * L / 2.0
    # Reactions due to point loads: RL_i = P*(L - a)/L, RR_i = P*a/L
    RL_points = sum(P * (L - a) / L for P, a in point_loads_N)
    RR_points = sum(P * a / L for P, a in point_loads_N)
    RL = R_dist_left + RL_points
    RR = q_total * L - RL  # total load must balance, but we mainly use RL for M(x)
    # --- Compute bending moment M(x) for each x (sign convention: positive sagging) ---
    # M(x) = RL * x - q_total * x^2 / 2 - sum_{points} P * max(0, x - a)
    M = np.zeros_like(x)
    for i, xi in enumerate(x):
        Mxi = RL * xi - q_total * xi**2 / 2.0
        # subtract contributions from point loads (they produce negative moment after their position)
        for P, a in point_loads_N:
            if xi > a:
                Mxi -= P * (xi - a)
        M[i] = Mxi
    # --- Integrate twice to get deflection (numerical) ---
    # slope(x) = (1/(E*I)) * integral_0^x M(t) dt
    # deflection(x) = (1/(E*I)) * integral_0^x [ integral_0^s M(t) dt ] ds
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    slope_raw = cumtrapz(M, x, initial=0.0) / (E * I)
    w_raw = cumtrapz(slope_raw, x, initial=0.0)
    # Enforce simply-supported BC: w(0)=0 and w(L)=0 by subtracting linear correction
    w0 = w_raw[0]
    wL = w_raw[-1]
    alpha = (wL - w0) / L
    w = w_raw - (w0 + alpha * x)
    # For plotting, downward deflection is typically negative, so we'll plot -w
    deflection = -w   # positive downward
    # --- Prepare figure ---
    fig, ax = plt.subplots(figsize=figsize)
    # draw original beam rectangle (centered vertically at y=0)
    beam = Rectangle((0.0, -h/2.0), L, h, facecolor="#cfe8ff", edgecolor="k", linewidth=1.2)
    ax.add_patch(beam)
    # deformed midline (scaled for visualization)
    disp_scaled = deflection * scale
    # draw deformed centerline
    ax.plot(x, disp_scaled, color="red", linewidth=2.0, label=f"Deformed midline (scale={scale}x)")
    # optionally draw deformed rectangle outline by shifting top/bottom by deflection
    if draw_deformed_outline:
        top_y = +h/2.0 + disp_scaled
        bot_y = -h/2.0 + disp_scaled
        ax.plot(x, top_y, color="r", linestyle="-", linewidth=1.0)
        ax.plot(x, bot_y, color="r", linestyle="-", linewidth=1.0)
# --- Plotting loads ---
# Uniform distributed load: represent as a series of arrows along the beam
        n_arrows = 15
        arrow_x = np.linspace(0.05*L, 0.95*L, n_arrows)
    for ax_pos in arrow_x:
        ax.annotate(
        "", 
        xy=(ax_pos, +h/2.0 + 0.4*h),  # arrow tip (downward/upward depending on sign)
        xytext=(ax_pos, +h/2.0 + 0.9*h),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5)
    )
    ax.text(0.5*L, +h/2.0 + 1.2*h, f"q = {q_total/1000:.2f} kN/m", ha="center", color="blue")
    # Point loads: arrows downward (positive P) or upward (negative P)
    for (P, a) in point_loads_N:
     if P >= 0:
        ax.annotate("", xy=(a, +h/2.0 + 0.4*h), xytext=(a, +h/2.0 + 1.0*h),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(a, +h/2.0 + 1.3*h, f"{P/1000:.1f} kN", ha="center")
    else:  # uplift (negative load)
        ax.annotate("", xy=(a, +h/2.0 + 1.0*h), xytext=(a, +h/2.0 + 0.4*h),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax.text(a, +h/2.0 + 1.3*h, f"{-P/1000:.1f} kN (up)", ha="center", color="red")
    # aesthetics
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Vertical (scaled units)")
    ax.set_title("Beam: original (filled) and deformed midline (red)")
    ax.set_xlim(-0.05*L, 1.05*L)
    # vertical limits: include some margin above top and below bottom + max deflection
    ymax = max(h, np.max(np.abs(disp_scaled))) + 0.8*h
    ymin = -h - 0.8*h + np.min(np.abs(disp_scaled)) - 0.1*h
    ax.set_ylim(-ymax, ymax)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_yaxis() #make positive downward for deflection 
    plt.show()
    # Return arrays in SI units (deflection in meters)
    return x, deflection
