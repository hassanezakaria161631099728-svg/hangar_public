import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.patches import Arc

def FEM2D(A_h, A_v, A_d, nodes, elements, loads, constraints, E=210e6):
    """
    FEM2D - Simplified 2D Truss Solver (Pythonic convention) [kN, m units]
    Parameters
    ----------
    A_h, A_v, A_d : float
        Cross-sectional areas (m²) for horizontal, vertical, and diagonal members.
    nodes : ndarray (n_nodes x 2)
        Node coordinates [x, y] in meters.
    elements : ndarray (n_elems x 2)
        Element connectivity (0-based node indices).
    loads : ndarray (n_loads x 2)
        External loads: [DOF_index, force_value] in kN.
    constraints : list or ndarray
        Constrained DOFs (0-based global DOF indices).
    E : float
        Young’s modulus (kN/m²). Default = 210000 kN/m² ≈ 210 GPa.
    Returns
    -------
    u : ndarray (n_dofs,)
        Global displacement vector (m).
    reactions : ndarray
        Reaction forces at constrained DOFs (kN).
    axial_forces : ndarray (n_elems,)
        Axial force in each truss element (kN).
    elem_types : list of str
        List of element types: 'H', 'V', or 'D'.
    """
    # ---- Convert raw lists to numpy arrays for safe math ----
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    loads = np.array(loads, dtype=float) if loads is not None else np.zeros((0,2))
    constraints = np.array(constraints, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 2 * n_nodes
    # --- Global matrices ---
    K_global = np.zeros((n_dofs, n_dofs))
    F_global = np.zeros(n_dofs)
    elem_types = []  # Track element type: 'H', 'V', 'D'
    # --- Assemble stiffness matrix ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        # Assign correct area and type
        if abs(y1 - y2) < 1e-9:
            A = A_h
            elem_types.append('H')
        elif abs(x1 - x2) < 1e-9:
            A = A_v
            elem_types.append('V')
        else:
            A = A_d
            elem_types.append('D')
        # Local stiffness matrix
        k_local = (E * A / L) * np.array([
            [ C*C,  C*S, -C*C, -C*S],
            [ C*S,  S*S, -C*S, -S*S],
            [-C*C, -C*S,  C*C,  C*S],
            [-C*S, -S*S,  C*S,  S*S]
        ])
        # DOF mapping
        dof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        K_global[np.ix_(dof, dof)] += k_local
    # --- Apply loads (in kN) ---
    for dof, value in loads:
        F_global[int(dof)] += value
    # --- Solve system ---
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, constraints)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_ff = F_global[free_dofs]
    u = np.zeros(n_dofs)
    u[free_dofs] = np.linalg.solve(K_ff, F_ff)
    # --- Reactions ---
    R = K_global @ u - F_global
    reactions = R[constraints]
    # --- Axial forces ---
    axial_forces = np.zeros(n_elems)
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        # Reuse same A logic
        if elem_types[e] == 'H':
            A = A_h
        elif elem_types[e] == 'V':
            A = A_v
        else:
            A = A_d
        dof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        u_elem = u[dof]
        T = np.array([-C, -S, C, S])
        axial_forces[e] = (E * A / L) * (T @ u_elem)
    return u, reactions, axial_forces, elem_types

def plot_truss(nodes, elements=None, loads=None, constraints=None,load_scale=None):
    """
    Plot a 2D truss structure with color-coded members, boundary conditions, and nodal loads.
    Colors:
        - Green: horizontal members
        - Yellow: vertical members
        - Blue: diagonal members
        - Purple: double fixed node (X and Y)
        - Green: fixed X
        - Brown: fixed Y
        - Black: free node
        - Red arrows: applied loads
    Parameters
    ----------
    nodes : ndarray (n_nodes x 2)
        Nodal coordinates [x, y].
    elements : ndarray (n_elems x 2), optional
        Element connectivity (node1, node2). 0- or 1-based indexing.
    constraints : list or ndarray, optional
        List of fixed DOFs (0-based). Example [0, 1, 7] means:
            - Node 0 fixed in X and Y
            - Node 3 fixed in Y (since 7 = 3*2 + 1)
    loads : ndarray or list, optional
        [[DOF_index, force_value], ...] — global DOF loads (FEM2D2 style)
    load_scale : float
        Arrow scaling for load visualization.
    """
    # Convert all inputs safely
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int) if elements is not None else None
    loads = np.array(loads, dtype=float) if loads is not None else None
    constraints = np.array(constraints, dtype=int) if constraints is not None else None
    plt.figure(figsize=(8, 5))
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Truss Geometry")
    n_nodes = len(nodes)
    n_dofs = n_nodes * 2
    # --- Build constraint map per node ---
    node_constraints = np.zeros((n_nodes, 2), dtype=bool)  # [Fx_fixed, Fy_fixed]
    if constraints is not None:
        for dof in constraints:
            node = dof // 2
            dirn = dof % 2  # 0 = X, 1 = Y
            if node < n_nodes:
                node_constraints[node, dirn] = True
    # --- Node color logic ---
    for i, (x, y) in enumerate(nodes):
        fixed_x, fixed_y = node_constraints[i]
        if fixed_x and fixed_y:
            color = 'purple'  # double fixed
        elif fixed_x:
            color = 'green'   # fixed X
        elif fixed_y:
            color = 'brown'   # fixed Y
        else:
            color = 'black'   # free
        plt.plot(x, y, 'o', color=color, markersize=8)
        plt.text(x + 0.15, y, f"N{i+1}", fontsize=10, color='k')
    # --- Adjust element indexing if 1-based ---
    if elements is not None and len(elements) > 0:
        if elements.min() == 1:
            elements = elements - 1
        # --- Plot members and label verticals/diagonals ---
        vert_count = 0
        diag_count = 0
        for i in range(elements.shape[0]):
            n1, n2 = elements[i, :]
            x = [nodes[n1, 0], nodes[n2, 0]]
            y = [nodes[n1, 1], nodes[n2, 1]]
            xm, ym = np.mean(x), np.mean(y)
            # Determine orientation
            if abs(y[0] - y[1]) < 1e-6:
                color = "g"  # horizontal
                label = None
            elif abs(x[0] - x[1]) < 1e-6:
                color = "y"  # vertical
                vert_count += 1
                label = f"P{vert_count}"
            else:
                color = "b"  # diagonal
                diag_count += 1
                label = f"D{diag_count}"
            plt.plot(x, y, color=color, linewidth=2)
            if label:
                plt.text(xm, ym, label, fontsize=10, color="k",
                         fontweight="bold", ha="center", va="center")
    # --- Plot loads (DOF-based only) ---
    if loads is not None and len(loads) > 0:
        node_loads = np.zeros((n_nodes, 2))
        for dof, val in loads:
            dof = int(dof)
            node = dof // 2
            dirn = dof % 2
            if node < n_nodes:
                node_loads[node, dirn] += val
        for i, (Fx, Fy) in enumerate(node_loads):
            if abs(Fx) > 1e-9 or abs(Fy) > 1e-9:
                x, y = nodes[i]
                plt.arrow(
                    x, y,
                    load_scale * Fx, load_scale * Fy,
                    head_width=0.15, head_length=0.25,
                    fc='red', ec='red', linewidth=1.5, zorder=5
                )
    # --- Legend ---
    legend_items = [
        plt.Line2D([], [], color='g', lw=2, label='Horizontal member'),
        plt.Line2D([], [], color='y', lw=2, label='Vertical member (Pn)'),
        plt.Line2D([], [], color='b', lw=2, label='Diagonal member (Dn)'),
        plt.Line2D([], [], color='purple', marker='o', linestyle='None', label='Double fixed'),
        plt.Line2D([], [], color='green', marker='o', linestyle='None', label='Fixed X'),
        plt.Line2D([], [], color='brown', marker='o', linestyle='None', label='Fixed Y'),
        plt.Line2D([], [], color='black', marker='o', linestyle='None', label='Free'),
        plt.Line2D([], [], color='red', marker=r'$\rightarrow$', linestyle='None', label='Load')
    ]
    plt.legend(handles=legend_items, loc='upper right', fontsize=8)
    plt.show()

def create_roof_bracing(num_panels=4, panel_length=4.0, height=4.0):
    """
    Create geometry (nodes and elements) for a 2D X-braced truss structure.
    Each bay forms an X pattern between top and bottom chords.
    """
    # --- Nodes ---
    x_bottom = np.arange(0, (num_panels + 1) * panel_length, panel_length)
    x_top = x_bottom.copy()
    y_bottom = np.zeros_like(x_bottom)
    y_top = np.full_like(x_top, height)
    bottom_nodes = np.column_stack((x_bottom, y_bottom))
    top_nodes = np.column_stack((x_top, y_top))
    nodes = np.vstack((bottom_nodes, top_nodes))  # (N1...N(bottom), N(top)...)
    # --- Elements ---
    elements = []
    # Bottom chord (green)
    for i in range(num_panels):
        elements.append([i + 1, i + 2])  # bottom nodes
    # Top chord (green)
    for i in range(num_panels):
        elements.append([num_panels + 1 + i, num_panels + 2 + i])  # top nodes
    # Verticals (yellow)
    for i in range(num_panels + 1):
        elements.append([i + 1, num_panels + 1 + i])  # connect bottom to top
    # Diagonals (blue) — X bracing in each bay
    for i in range(num_panels):
        # D1: bottom-left → top-right
        elements.append([i + 1, num_panels + 2 + i])
        # D2: top-left → bottom-right
        elements.append([num_panels + 1 + i, i + 2])
    return np.array(nodes), np.array(elements)

def plot_truss_bracing(nodes, elements):
    """
    Plot a 2D X-braced truss (roof or wall bracing) with color-coded elements.
    Color legend:
    - Green  → horizontal members (top & bottom chords)
    - Yellow → verticals (P-members)
    - Blue   → diagonals (D-members)
    Parameters
    ----------
    nodes : ndarray (n_nodes x 2)
        Nodal coordinates [x, y].
    elements : ndarray (n_elems x 2)
        Element connectivity (1-based, as from create_roof_bracing).
    """
    plt.figure(figsize=(9, 5))
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Roof/Wall Bracing Geometry")
    # Counters for labeling
    Pcount = 0
    Dcount = 0
    # --- Plot nodes ---
    for i in range(nodes.shape[0]):
        plt.plot(nodes[i, 0], nodes[i, 1],
                 'ro', markersize=8, markerfacecolor='r')
        plt.text(nodes[i, 0] + 0.1, nodes[i, 1],
                 f"N{i+1}", fontsize=9)
    # --- Plot elements ---
    for i in range(elements.shape[0]):
        n1, n2 = elements[i, :] - 1  # convert to 0-based
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]
        xm, ym = np.mean(x), np.mean(y)
        dx, dy = x[1] - x[0], y[1] - y[0]
        # Determine orientation
        if abs(dy) < 1e-8:       # horizontal member
            color = "g"
            plt.plot(x, y, color=color, linewidth=2)
        elif abs(dx) < 1e-8:     # vertical post
            color = "y"
            plt.plot(x, y, color=color, linewidth=2)
            Pcount += 1
            plt.text(xm, ym, f"P{Pcount}", fontsize=9, color="k",
                     fontweight="bold", ha="center", va="center")
        else:                    # diagonal brace
            color = "b"
            plt.plot(x, y, color=color, linewidth=2)
            Dcount += 1
            plt.text(xm, ym, f"D{Dcount}", fontsize=9, color="k",
                     fontweight="bold", ha="center", va="center")
    plt.show()

def create_truss_nodes(n_panel: int, L: float, t: float):
    """
    Create nodal coordinates for a horizontal truss
    with top and bottom chords of constant height t.
    Parameters
    ----------
    n_panel : int
        Number of horizontal panels (bays)
    L : float
        Horizontal distance between verticals
    t : float
        Truss height (vertical distance between chords)
    Returns
    -------
    nodes : np.ndarray
        Array of nodal coordinates [[x0, y0], [x1, y1], ...]
    """
    # horizontal distances
    d = np.arange(0, (n_panel + 1) * L, L)
    # bottom chord (y = 0)
    x_bottom = d
    y_bottom = np.zeros_like(d)
    # top chord (y = t)
    x_top = d
    y_top = np.full_like(d, t)
    # concatenate bottom + top
    x_all = np.concatenate([x_bottom, x_top])
    y_all = np.concatenate([y_bottom, y_top])
    nodes = np.column_stack((x_all, y_all))
    return nodes

def create_X_horizontal_truss(n_panel=None, L=None, t=None):
    """
    Create node coordinates and connectivity matrix for a 2D X-braced truss.
    Parameters
    ----------
    n_panel : int
        Number of horizontal panels (bays)
    L : float
        Horizontal distance between verticals
    t : float
        Truss height (vertical distance between chords)
    Returns
    -------
    nodes : np.ndarray
        Array of nodal coordinates [[x, y], ...]
    elements : np.ndarray
        Element connectivity [[n1, n2], ...] (0-based indices)
    """
    # --- Node coordinates ---
    d = np.arange(0, (n_panel + 1) * L, L)
    x_bottom, y_bottom = d, np.zeros_like(d)
    x_top, y_top = d, np.full_like(d, t)
    nodes = np.column_stack((np.concatenate([x_bottom, x_top]),
                             np.concatenate([y_bottom, y_top])))
    n_bottom = n_panel + 1
    elements = []
    # --- Bottom chord ---
    for i in range(n_panel):
        elements.append([i, i + 1])
    # --- Top chord ---
    for i in range(n_panel):
        elements.append([i + n_bottom, i + n_bottom + 1])
    # --- Verticals (P-members) ---
    for i in range(n_bottom):
        elements.append([i, i + n_bottom])
    # --- Diagonals (X bracing) ---
    for i in range(n_panel):
        # Diagonal 1: bottom-left → top-right
        elements.append([i, i + 1 + n_bottom])
        # Diagonal 2: top-left → bottom-right
        elements.append([i + n_bottom, i + 1])
    return np.array(nodes), np.array(elements)

def create_vertical_bracing_nodes(nf: int, h: float, L: float):
    """
    Create nodal coordinates for a vertical wall bracing truss.
    Parameters
    ----------
    nf : int
        Number of floors (vertical panels)
    h : float
        Height of each floor (m)
    L : float
        Horizontal distance between left and right chords (m)
    Returns
    -------
    nodes : np.ndarray
        Array of nodal coordinates [[x, y], ...] (0-based indexing)
    """
    # Vertical distances (heights)
    y_levels = np.arange(0, (nf + 1) * h, h)
    # Left chord nodes (x=0)
    x_left = np.zeros_like(y_levels)
    y_left = y_levels
    # Right chord nodes (x=L)
    x_right = np.full_like(y_levels, L)
    y_right = y_levels
    # Combine into full node set
    nodes = np.column_stack((
        np.concatenate([x_left, x_right]),
        np.concatenate([y_left, y_right])
    ))
    return nodes

def create_vertical_bracing_elements(nf: int):
    """
    Create element connectivity for a vertical X-braced wall truss.
    Parameters
    ----------
    nf : int
        Number of floors (vertical panels)
    Returns
    -------
    elements : np.ndarray
        Array of element connectivity [[n1, n2], ...] (0-based indexing)
    """
    elements = []
    n_left = nf + 1  # number of nodes on one side
    # --- Left vertical chord ---
    for i in range(nf):
        elements.append([i, i + 1])
    # --- Right vertical chord ---
    for i in range(nf):
        elements.append([i + n_left, i + n_left + 1])
    # --- Horizontal ties (no ground level) ---
    for i in range(1, n_left):  # start from level 1, skip ground (i=0)
        elements.append([i, i + n_left])
    # --- Diagonals for each floor bay ---
    for i in range(nf):
        # D1: left lower → right upper
        elements.append([i, i + n_left + 1])
        # D2: right lower → left upper
        elements.append([i + n_left, i + 1])
    return np.array(elements)

def create_vertical_bracing_truss(nf: int, h: float, L: float):
    """
    Full geometry generation for a vertical wall X-braced truss.
    Returns both node coordinates and element connectivity.
    """
    nodes = create_vertical_bracing_nodes(nf, h, L)
    elements = create_vertical_bracing_elements(nf)
    return nodes, elements

def animate_truss_deformation(nodes, elements, u, scale=100, frames=30, interval=80, save_path=None):
    """
    Animate the deformation of a 2D truss structure.
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    u = u.reshape((n_nodes, 2))
    nodes_def = nodes + scale * u
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Truss Deformation Animation")
    # Original (gray) truss
    for (n1, n2) in elements:
        ax.plot([nodes[n1,0], nodes[n2,0]], [nodes[n1,1], nodes[n2,1]], '0.7', lw=2)
    # Animated (red) truss
    lines = [ax.plot([], [], 'r', lw=2)[0] for _ in elements]
    # Set bounds
    all_x = np.hstack([nodes[:,0], nodes_def[:,0]])
    all_y = np.hstack([nodes[:,1], nodes_def[:,1]])
    pad = 0.1 * max(np.ptp(all_x), np.ptp(all_y))
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    # --- Frame update function ---
    def update(frame):
        alpha = frame / frames
        current = nodes + alpha * (nodes_def - nodes)
        for line, (n1, n2) in zip(lines, elements):
            line.set_data([current[n1,0], current[n2,0]], [current[n1,1], current[n2,1]])
        return lines
    # --- Create the animation ---
    ani = animation.FuncAnimation(
        fig, update, frames=frames + 1,
        blit=True, interval=interval, repeat=True
    )
    # --- Save or show depending on environment ---
    if save_path:
        if save_path.endswith(".mp4"):
            ani.save(save_path, writer="ffmpeg", fps=1000//interval)
        else:
            ani.save(save_path, writer="pillow")  # default: GIF
        print(f"✅ Animation saved to {save_path}")
    else:
        plt.show()
    return ani

#from CM pdf I extracted this
def plot_portal_frame(L=12, hp=6, rise=2, save_svg=False, filename="portal_frame.svg"):
    """
    Draw a 2D portal frame with fixed bases and roof loads.
    Optionally saves the figure as a scalable SVG file.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_aspect('equal')
    ax.axis('off')
    # --- Geometry ---
    half_L = L / 2
    roof_y = hp + rise
    nodes = np.array([
        [-half_L, 0],   # left base
        [ half_L, 0],   # right base
        [-half_L, hp],  # left eave
        [     0, roof_y], # ridge
        [ half_L, hp]   # right eave
    ])
    # --- Frame lines ---
    frame_segments = [[0, 2], [2, 3], [3, 4], [4, 1]]
    for n1, n2 in frame_segments:
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]
        ax.plot(x, y, color='k', lw=3)
    # --- Fixed supports ---
    for (x, y) in [nodes[0], nodes[1]]:
        ax.add_patch(plt.Rectangle((x-0.3, y-0.2), 0.6, 0.2, color='k'))
        ax.text(x, y - 0.6, "(Encastrement)", ha='center', fontsize=10, style='italic')
    # --- Vertical dimension (h_p) ---
    ax.annotate('', xy=(-half_L-1, 0), xytext=(-half_L-1, hp),
                arrowprops=dict(arrowstyle='<->', color='k'))
    ax.text(-half_L-1.3, hp/2, r"$h_p$", fontsize=14, va='center', ha='right')
    # --- Horizontal dimension (L) ---
    ax.annotate('', xy=(-half_L, -1), xytext=(half_L, -1),
                arrowprops=dict(arrowstyle='<->', color='k'))
    ax.text(0, -1.4, r"$L$", fontsize=14, va='center', ha='center')
    # --- Loads ---
    # point loads Q_G
    ax.arrow(nodes[2,0], nodes[2,1]+0.5, 0, -0.5, color='blue',
             head_width=0.3, head_length=0.4, lw=2, length_includes_head=True)
    ax.arrow(nodes[4,0], nodes[4,1]+0.5, 0, -0.5, color='blue',
             head_width=0.3, head_length=0.4, lw=2, length_includes_head=True)
    ax.text(nodes[2,0]-0.2, nodes[2,1]+1.0, r"$Q_G$", color='blue', fontsize=14)
    ax.text(nodes[4,0]+0.2, nodes[4,1]+1.0, r"$Q_G$", color='blue', fontsize=14)
    # distributed load q_G
    n_arrows = 8
    xs = np.linspace(nodes[2,0], nodes[4,0], n_arrows)
    ys = np.interp(xs, [nodes[2,0], nodes[3,0], nodes[4,0]], [nodes[2,1], nodes[3,1], nodes[4,1]])
    for x, y in zip(xs, ys):
        ax.arrow(x, y+0.5, 0, -0.5, color='red',
                 head_width=0.15, head_length=0.25, lw=1.5, length_includes_head=True)
    ax.text(0, roof_y+0.8, r"$q_G$", color='red', fontsize=14, ha='center')
    plt.tight_layout()
    # --- Export SVG if requested ---
    if save_svg:
        save_path = os.path.join(os.getcwd(), filename)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"✅ SVG saved successfully at:\n{save_path}")
    plt.show()

def plot_frame(nodes, elements, elem_props, loads, constraints, scale_load=0.3):

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------------------
    # 1. Draw Elements (with color coding)
    # --------------------------------------------------------------
    for e, (n1, n2) in enumerate(elements):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        dx = x2 - x1
        dy = y2 - y1

        # Determine element type by orientation
        if abs(dx) < 1e-6:          # vertical
            color = "yellow"
        elif abs(dy) < 1e-6:        # horizontal
            color = "green"
        else:                       # diagonal
            color = "blue"

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)

    # --------------------------------------------------------------
    # 2. Draw Nodes
    # --------------------------------------------------------------
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'ko')
        ax.text(x + 0.05, y + 0.05, f"{i}", fontsize=9)

    # --------------------------------------------------------------
    # 3. Supports (constraints)
    # --------------------------------------------------------------
    pinned = {}

    for c in constraints:
        node = c // 3
        dof = c % 3

        if node not in pinned:
            pinned[node] = [False, False, False]

        pinned[node][dof] = True

    for node, locked in pinned.items():
        ux, uy, rz = locked
        x, y = nodes[node]

        if ux and uy and rz:
            # fully fixed
            ax.plot(x, y, 's', color="purple", markersize=10)

        elif ux and uy and not rz:
            # **ball joint** (your request)
            ax.plot(x, y, 'o', color="red", markersize=12, fillstyle="none")

        elif uy and not ux:
            # vertical roller
            ax.plot(x, y, 'o', color="green", markersize=10)

        else:
            # just in case of rare combos
            ax.plot(x, y, 'o', color="black")

    # --------------------------------------------------------------
    # 4. Nodal Loads (now BLUE, and moved outward)
    # --------------------------------------------------------------
    for dof, P in loads:
        node = dof // 3
        comp = dof % 3
        x, y = nodes[node]

        offset = 0.25  # push arrow outward so it is visible

        if comp == 0:      # Fx
            ax.arrow(x + offset*np.sign(P), y,
                     scale_load*np.sign(P), 0,
                     head_width=0.1, fc='blue', ec='blue')

        elif comp == 1:    # Fy
            ax.arrow(x, y + offset*np.sign(P),
                     0, scale_load*np.sign(P),
                     head_width=0.1, fc='blue', ec='blue')

        elif comp == 2:    # Mz
            r = 0.3
            theta2 = -180 if P < 0 else 180
            arc = Arc((x, y), r, r, theta1=0, theta2=theta2,
                      color='blue', linewidth=2)
            ax.add_patch(arc)

    # --------------------------------------------------------------
    # 5. Distributed Loads (skip if q = 0)
    # --------------------------------------------------------------
    for e, prop in enumerate(elem_props):

        if 'q' not in prop and 'w' not in prop:
            continue

        q_raw = float(prop.get('q', prop.get('w', 0.0)))

        # nothing to draw if zero
        if abs(q_raw) < 1e-12:
            continue

        load_type = prop.get('load_type', 'local')

        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        C, S = dx/L, dy/L

        # local y-axis direction in global coords
        e_y_local = np.array([-S, C])

        # convert load to global coordinates
        if load_type == "local":
            w_vec = q_raw * e_y_local
        else:
            w_vec = np.array([q_raw, 0]) if np.isscalar(q_raw) else q_raw

        # draw 5 arrows along the element
        for i in range(1, 5):
            t = i / 5
            xp = x1 + t * dx
            yp = y1 + t * dy
            ax.arrow(xp, yp,
                     scale_load*w_vec[0],
                     scale_load*w_vec[1],
                     head_width=0.1,
                     fc='red', ec='red')

    # --------------------------------------------------------------
    # Final formatting
    # --------------------------------------------------------------
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Frame Model with Supports & Loads")

    plt.show()


def plot_frame_results(nodes, elements, elem_props, loads, constraints,
                       u=None, shear_forces=None, end_moments=None,
                       scale=1.0, load_scale=0.2, q_scale=0.2,
                       show_moments=True, show_shear=True, show_deformed=True):
    """
    Visualize a 2D frame model and optionally its FEM2D_frame results:
    - undeformed geometry
    - deformed shape
    - shear and bending moment diagrams
    - applied loads and supports
    Parameters
    ----------
    nodes : (n_nodes x 2)
        Nodal coordinates [x, y].
    elements : (n_elems x 2)
        Element connectivity.
    elem_props : list of dict
        Each dict can include 'q', 'type', 'A', 'I', etc.
    loads : list [[dof_index, value], ...]
        Global DOF loads.
    constraints : list of constrained DOFs (0-based).
    u : ndarray (3*n_nodes,)
        Global displacement vector from FEM2D_frame.
    shear_forces : ndarray (n_elems, 2)
        Shear forces [V1, V2] per element (optional).
    end_moments : ndarray (n_elems, 2)
        End moments [M1, M2] per element (optional).
    scale : float
        Deformation magnification factor.
    load_scale, q_scale : float
        Scaling for arrows.
    show_moments, show_shear, show_deformed : bool
        Toggles for displaying results.
    """
    plt.figure(figsize=(9, 6))
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Frame Analysis Visualization")
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]

    if elements.min() == 1:
        elements = elements - 1
    # --- Supports ---
    n_dofs = 3 * n_nodes
    node_constraints = np.zeros((n_nodes, 3), dtype=bool)
    if constraints is not None:
        for dof in constraints:
            node = dof // 3
            dirn = dof % 3
            if node < n_nodes:
                node_constraints[node, dirn] = True
    for i, (x, y) in enumerate(nodes):
        fixed = node_constraints[i]
        if fixed.all():
            color = "purple"
        elif fixed[0] and fixed[1]:
            color = "purple"
        elif fixed[0]:
            color = "green"
        elif fixed[1]:
            color = "brown"
        else:
            color = "black"
        plt.plot(x, y, 'o', color=color, markersize=7)
        plt.text(x + 0.1, y + 0.1, f"N{i+1}", fontsize=9)
    # --- Undeformed structure ---
    for e, (n1, n2) in enumerate(elements):
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]
        plt.plot(x, y, 'k-', lw=2)
    # --- Deformed shape ---
    if u is not None and show_deformed:
        u = np.asarray(u).reshape(-1)
        def_coords = np.zeros_like(nodes)
        for i in range(n_nodes):
            def_coords[i, 0] = nodes[i, 0] + scale * u[3*i]
            def_coords[i, 1] = nodes[i, 1] + scale * u[3*i + 1]
        for e, (n1, n2) in enumerate(elements):
            xd = [def_coords[n1, 0], def_coords[n2, 0]]
            yd = [def_coords[n1, 1], def_coords[n2, 1]]
            plt.plot(xd, yd, 'b--', lw=1.5, label="Deformed" if e == 0 else "")
    # --- Distributed loads ---
    for e, (n1, n2) in enumerate(elements):
        if elem_props is not None and e < len(elem_props):
            q = elem_props[e].get("q", 0)
            if abs(q) > 1e-9:
                x1, y1 = nodes[n1]
                x2, y2 = nodes[n2]
                L = np.hypot(x2 - x1, y2 - y1)
                nx = (x2 - x1) / L
                ny = (y2 - y1) / L
                perp_x, perp_y = -ny, nx
                n_arrows = max(3, int(L / 0.5))
                for j in range(1, n_arrows + 1):
                    s = j / (n_arrows + 1)
                    px = x1 + s * (x2 - x1)
                    py = y1 + s * (y2 - y1)
                    plt.arrow(px, py,
                              q_scale * q * perp_x,
                              q_scale * q * perp_y,
                              head_width=0.1, head_length=0.15,
                              fc='red', ec='red', alpha=0.7, zorder=5)
    # --- Nodal loads ---
    if loads is not None and len(loads) > 0:
        node_loads = np.zeros((n_nodes, 2))
        for dof, val in loads:
            node = dof // 3
            dirn = dof % 3
            if node < n_nodes and dirn < 2:
                node_loads[node, dirn] += val
        for i, (Fx, Fy) in enumerate(node_loads):
            if abs(Fx) > 1e-9 or abs(Fy) > 1e-9:
                x, y = nodes[i]
                plt.arrow(x, y, load_scale * Fx, load_scale * Fy,
                          head_width=0.1, head_length=0.2,
                          fc='red', ec='red', zorder=6)
    # --- Bending moment & shear diagrams ---
    for e, (n1, n2) in enumerate(elements):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.hypot(x2 - x1, y2 - y1)
        nx = (x2 - x1) / L
        ny = (y2 - y1) / L
        perp_x, perp_y = -ny, nx
        if end_moments is not None and show_moments:
            M1, M2 = end_moments[e]
            Mx = np.linspace(0, L, 10)
            My = np.linspace(M1, M2, 10)
            for xi, mi in zip(Mx, My):
                px = x1 + xi * nx
                py = y1 + xi * ny
                plt.plot([px, px + 0.001*mi*perp_x],
                         [py, py + 0.001*mi*perp_y],
                         color='magenta', lw=2, alpha=0.7)
        if shear_forces is not None and show_shear:
            V1, V2 = shear_forces[e]
            Vx = np.linspace(0, L, 10)
            Vy = np.linspace(V1, V2, 10)
            for xi, vi in zip(Vx, Vy):
                px = x1 + xi * nx
                py = y1 + xi * ny
                plt.plot([px, px + 0.003*vi*perp_x],
                         [py, py + 0.003*vi*perp_y],
                         color='orange', lw=2, alpha=0.7)
    plt.legend(loc='upper right')
    plt.show()

def FEM2D_frame(nodes, elements, elem_props, loads, constraints, default_E=210e6):
    """
    Modified 2D frame FEM solver that returns axial force at both ends of each element.
    Parameters same as your original function, plus:
      - elem_props[e] may include 'w' : global vertical uniform load (kN/m, positive downwards).
        If 'w' is provided, it's resolved into local transverse q_local and axial p_axial.
      - If 'q' is given in elem_props it is interpreted as local -Y uniform load (kN/m).
    Returns
    -------
    u, reactions, axial_avg, axial_ends, shear_forces, end_moments
      - axial_avg: (n_elems,) average axial (kN) same sign conv as before (+ tension)
      - axial_ends: (n_elems,2) [axial_at_node1, axial_at_node2] in local coordinates
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 3 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    F = np.zeros(n_dofs)
    def T_matrix(C, S):
        R = np.array([[C, S, 0],
                      [-S, C, 0],
                      [0, 0, 1]])
        T = np.zeros((6, 6))
        T[:3, :3] = R
        T[3:, 3:] = R
        return T
    axial_avg = np.zeros(n_elems)
    axial_ends = np.zeros((n_elems, 2))
    end_moments = np.zeros((n_elems, 2))
    shear_forces = np.zeros((n_elems, 2))
    # --- Assembly loop ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        C, S = dx / L, dy / L
        prop = elem_props[e]
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        # q defined as local -Y uniform load (kN/m)
        q_local = prop.get('q', 0.0)
        # w defined as global vertical uniform load (kN/m, positive downwards)
        w_global = prop.get('w', 0.0)
        # resolve global vertical load into local components:
        # local = R @ global; for global (0, -w_global) (negative because downwards),
        # local axial component p_axial = C*0 + S*(-w_global) = - S * w_global
        # (we take positive axial in +x local direction; sign depends on your convention)
        p_axial_from_w =  -S * w_global  # kN/m along local +x
        # note: the local transverse q_effective = q_local + (C * -w_global)
        # but if you intentionally provide q_local, treat it as local already.
        # Here we treat both: total local transverse load = q_local + (-C * w_global)
        q_from_w =  -C * w_global
        q_total = q_local + q_from_w
        etype = prop.get('type', 'beam')
        # stiffness
        k_ax = E * A / L
        if I > 0 and etype == 'beam':
            k_b11 = 12 * E * I / L**3
            k_b12 = 6 * E * I / L**2
            k_b22 = 4 * E * I / L
            k_b22_off = 2 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0
        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ])
        # transform
        T = T_matrix(C, S)
        k_global = T.T @ k_local @ T
        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        K[np.ix_(dof, dof)] += k_global
        # equivalent nodal forces (for total q_total (local -Y) AND axial p_axial_from_w)
        if abs(q_total) > 1e-12 or abs(p_axial_from_w) > 1e-12:
            # transverse & moment parts (local)
            f_local = np.array([
                0.0,
                q_total * L / 2.0,
                q_total * L**2 / 12.0,
                0.0,
                q_total * L / 2.0,
                - q_total * L**2 / 12.0
            ])
            # add axial distributed-load equivalent nodal axial forces:
            # for a uniform axial line load p (kN/m) along +x, equivalent nodal axial forces are:
            # [p*L/2, 0, 0, p*L/2, 0, 0]  (signs depend on sign of p)
            f_local[0] += p_axial_from_w * L / 2.0
            f_local[3] += p_axial_from_w * L / 2.0
            # transform to global and add
            F[np.ix_(dof)] += T.T @ f_local
    # --- point loads ---
    for dof, val in loads:
        F[int(dof)] += val
    # --- solve system ---
    all_dofs = np.arange(n_dofs)
    free = np.setdiff1d(all_dofs, constraints)
    u = np.zeros(n_dofs)
    K_ff = K[np.ix_(free, free)]
    F_ff = F[free]
    u[free] = np.linalg.solve(K_ff, F_ff)
    reactions = (K @ u - F)[constraints]
    # --- postprocess (axial at both ends, shear, moments) ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        C, S = dx / L, dy / L
        prop = elem_props[e]
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        q_local = prop.get('q', 0.0)
        w_global = prop.get('w', 0.0)
        p_axial_from_w = - S * w_global
        q_from_w = - C * w_global
        q_total = q_local + q_from_w
        # rebuild local stiffness
        k_ax = E * A / L
        if I > 0 and prop.get('type', 'beam') == 'beam':
            k_b11 = 12 * E * I / L**3
            k_b12 = 6 * E * I / L**2
            k_b22 = 4 * E * I / L
            k_b22_off = 2 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0
        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ])
        # transform and compute local element displacement
        R = np.array([[C, S, 0],
                      [-S, C, 0],
                      [0, 0, 1]])
        Tm = np.zeros((6,6))
        Tm[:3,:3] = R
        Tm[3:,3:] = R
        u_e = Tm @ u[dof]
        # local f_local (same as in assembly)
        if abs(q_total) > 1e-12 or abs(p_axial_from_w) > 1e-12:
            f_local = np.array([
                0.0,
                q_total * L / 2.0,
                q_total * L**2 / 12.0,
                0.0,
                q_total * L / 2.0,
                - q_total * L**2 / 12.0
            ])
            f_local[0] += p_axial_from_w * L / 2.0
            f_local[3] += p_axial_from_w * L / 2.0
        else:
            f_local = np.zeros(6)
        internal = k_local @ u_e - f_local
        # internal vector components:
        # [N1, V1, M1, N2, V2, M2] in local element coords (N = axial)
        axial_ends[e, 0] = -internal[0]   # axial at node1 end (kN)
        axial_ends[e, 1] = internal[3]   # axial at node2 end (kN)
        #axial_avg[e] = 0.5 * (axial_ends[e,0] + axial_ends[e,1])
        shear_forces[e, 0] = -internal[1]
        shear_forces[e, 1] = internal[4]
        end_moments[e, 0] = -internal[2]
        end_moments[e, 1] = internal[5]
    # --- build vertical N, V, M arrays ---
    N_list = []
    V_list = []
    M_list = []

    for e in range(n_elems):
        # N
        N_list.append(axial_ends[e, 0])         # node 1 end
        N_list.append(axial_ends[e, 1])         # node 2 end

        # V (shear)
        V_list.append(shear_forces[e, 0])      # node 1 end
        V_list.append(shear_forces[e, 1])      # node 2 end

        # M (bending moment)
        M_list.append(end_moments[e, 0])       # node 1 end
        M_list.append(end_moments[e, 1])       # node 2 end

    N_vec = np.array(N_list, dtype=float).reshape(-1, 1)
    V_vec = np.array(V_list, dtype=float).reshape(-1, 1)
    M_vec = np.array(M_list, dtype=float).reshape(-1, 1)

    return u, reactions, N_vec, V_vec, M_vec


def beam_internal_forces(
    L,
    EI,
    node_forces,      # [V_i, M_i, V_j, M_j]
    P=None,           # point load magnitude
    a=None,           # position of point load from left node
    w=0.0,            # distributed load (q) along entire span
    npoints=200       # points for diagram resolution
):
    """
    Computes M(x), V(x), N(x) inside a 2-node Euler-Bernoulli beam.
    Returns vertical numpy arrays.
    """
    # ----------------------------
    # 1. Extract nodal forces
    # ----------------------------
    V_i, M_i, V_j, M_j = node_forces
    # ----------------------------
    # 2. Define local coordinate
    # ----------------------------
    x = np.linspace(0, L, npoints).reshape(-1, 1)
    # ----------------------------------------------
    # 3. Shear force V(x)
    # = interpolation from nodal shear + loads
    # ----------------------------------------------
    V = (V_i * (1 - x/L) + V_j * (x/L))
    # Add distributed load contribution: V -= w*x
    V -= w * x
    # Add point load contribution
    if P is not None and a is not None:
        V -= P * (x >= a)
    # ----------------------------------------------
    # 4. Bending moment M(x)
    # = integral of V(x)
    # ----------------------------------------------
    # Moment from nodal bending interpolation
    M = (M_i * (1 - 3*(x/L)**2 + 2*(x/L)**3) +
         M_j * (3*(x/L)**2 - 2*(x/L)**3))
    # Moment from distributed load
    M -= w * x**2 / 2
    # Moment from point load
    if P is not None and a is not None:
        M -= P * (x - a) * (x >= a)
    # ----------------------------------------------
    # 5. Axial force (zero in this bending-only case)
    # ----------------------------------------------
    N = np.zeros_like(x)
    return x, N, V, M

def generate_grid(nx, ny, Lx, Ly):
    nodes = []

    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id = j * (nx + 1) + i
            x = i * Lx
            y = j * Ly
            nodes.append((node_id, x, y))

    return nodes

def plot_grid(nodes, nx, ny):
    fig, ax = plt.subplots(figsize=(6, 6))

    for node_id, x, y in nodes:
        ax.plot(x, y, 'ks',markersize=8)  # node
        ax.text(x + 0.05, y + 0.05, str(node_id), fontsize=10)

    # draw grid lines
    for j in range(ny + 1):
        ax.plot(
            [nodes[j*(nx+1)][1], nodes[j*(nx+1) + nx][1]],
            [nodes[j*(nx+1)][2], nodes[j*(nx+1) + nx][2]],
            'k--', alpha=0.5
        )

    for i in range(nx + 1):
        ax.plot(
            [nodes[i][1], nodes[i + ny*(nx+1)][1]],
            [nodes[i][2], nodes[i + ny*(nx+1)][2]],
            'k--', alpha=0.5
        )

    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    plt.show()

def plot_grid_elevation_view(nodes, nx, nz):
    fig, ax = plt.subplots(figsize=(6, 6))

    for node_id, x, z in nodes:
        if z == 0:
            ax.plot(x, z, 'ks', markersize=8)  # ground-level column
        else:
            ax.plot(x, z, 'ko', markersize=6)  # upper nodes

        ax.text(x + 0.05, z + 0.05, str(node_id), fontsize=10)

    # horizontal grid lines (constant z)
#    for j in range(nz + 1):
#        ax.plot(
#            [nodes[j*(nx+1)][1], nodes[j*(nx+1) + nx][1]],
#           [nodes[j*(nx+1)][2], nodes[j*(nx+1) + nx][2]],
#            'k--', alpha=0.5
#        )

    # vertical grid lines (constant x)
#    for i in range(nx + 1):
#        ax.plot(
#            [nodes[i][1], nodes[i + nz*(nx+1)][1]],
#            [nodes[i][2], nodes[i + nz*(nx+1)][2]],
#            'k--', alpha=0.5
#        )

    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True)
    plt.show()

