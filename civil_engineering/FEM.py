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
    """
    Visual frame plot:
    - nodes
    - elements
    - supports (auto-detected)
    - nodal loads
    - distributed loads (local or global)
    """

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))

    # -----------------------------------------------------------------
    # 1. Draw Elements
    # -----------------------------------------------------------------
    for e, (n1, n2) in enumerate(elements):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)

    # -----------------------------------------------------------------
    # 2. Draw Nodes
    # -----------------------------------------------------------------
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'ko')
        ax.text(x + 0.05, y + 0.05, f"{i}", fontsize=9)

    # -----------------------------------------------------------------
    # 3. Supports (detected from constraints)
    # -----------------------------------------------------------------
    fixed = set()
    pinned = dict()    # track UX+UY
    roller_y = dict()  # only UY

    for c in constraints:
        node = c // 3
        dof = c % 3  # 0=UX,1=UY,2=RZ

        if node not in pinned:
            pinned[node] = [False, False, False]

        pinned[node][dof] = True

    for node, locked in pinned.items():
        ux, uy, rz = locked
        x, y = nodes[node]

        if ux and uy and rz:
            ax.plot(x, y, 's', color="purple", markersize=10)  # fixed
        elif ux and uy:
            # pinned
            ax.plot(x, y, marker=(3, 0, -90), color="blue", markersize=13)
        elif uy:
            # roller vertical
            ax.plot(x, y, 'o', color="green", markersize=10)

    # -----------------------------------------------------------------
    # 4. Nodal loads
    # -----------------------------------------------------------------
    for dof, P in loads:
        node = dof // 3
        comp = dof % 3
        x, y = nodes[node]

        if comp == 0:  # Fx
            ax.arrow(x, y, scale_load*np.sign(P), 0,
                     head_width=0.1, fc='red', ec='red')
        elif comp == 1:  # Fy
            ax.arrow(x, y, 0, scale_load*np.sign(P),
                     head_width=0.1, fc='red', ec='red')
        elif comp == 2:  # Moment
            r = 0.25
            theta1 = 0
            theta2 = -180 if P < 0 else 180
            arc = Arc((x, y), r, r, theta1=theta1, theta2=theta2,
                      color='red', linewidth=2)
            ax.add_patch(arc)

            # arrowhead
            ax.arrow(x + r/2, y, -0.001*np.sign(P), 0,
                     head_width=0.08, fc='red', ec='red')

    # -----------------------------------------------------------------
    # 5. Distributed Loads (local or global)
    # -----------------------------------------------------------------
    for e, prop in enumerate(elem_props):
        if 'q' not in prop and 'w' not in prop:
            continue

        q_raw = prop.get('q', prop.get('w', 0.0))
        load_type = prop.get('load_type', 'local')

        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        # element direction
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        C, S = dx/L, dy/L

        # local y-axis direction (global)
        e_y_local = np.array([-S, C])

        # evaluate q in GLOBAL coordinates for plotting
        if load_type == "local":
            q = float(q_raw)
            w_vec = q * e_y_local  # arrow direction in global

        else:  # GLOBAL input
            if hasattr(q_raw, "__len__"):
                w_vec = np.array([q_raw[0], q_raw[1]], dtype=float)
            else:
                w_vec = np.array([0.0, q_raw], dtype=float)

        # draw several arrows along the element
        n_arrows = 5
        for i in range(1, n_arrows):
            t = i / n_arrows
            xp = x1 + t * dx
            yp = y1 + t * dy

            ax.arrow(xp,
                     yp,
                     scale_load * w_vec[0],
                     scale_load * w_vec[1],
                     head_width=0.1,
                     fc='red', ec='red')

    # -----------------------------------------------------------------
    # Final formatting
    # -----------------------------------------------------------------
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
    2D frame FEM solver (axial + bending + shear).
    Extended: supports distributed loads given in local or global form.
    - If elem_props[e]['load_type'] == 'local' (default): q is local -Y (scalar).
    - If elem_props[e]['load_type'] == 'global': q can be scalar (qy) or 2-vector (qx,qy).
      It is converted to local q as: q_local = - w_global . e_loc_y, where e_loc_y = [-S, C].
      (This yields a scalar q_local that matches the original element equivalent nodal force routine.)
    Units: keep consistent with your project (original code used kN, kN/m, E in kN/m^2).
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 3 * n_nodes

    K = np.zeros((n_dofs, n_dofs), dtype=float)
    F = np.zeros(n_dofs, dtype=float)

    def T_matrix(C, S):
        R = np.array([[C, S, 0.0],
                      [-S, C, 0.0],
                      [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[:3, :3] = R
        T[3:, 3:] = R
        return T

    axial_forces = np.zeros(n_elems, dtype=float)
    end_moments = np.zeros((n_elems, 2), dtype=float)
    shear_forces = np.zeros((n_elems, 2), dtype=float)

    # --- Assembly loop ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L <= 0:
            raise ValueError(f"Element {e} has zero length (nodes {n1}, {n2}).")
        C, S = dx / L, dy / L

        prop = elem_props[e] if (elem_props is not None and e < len(elem_props)) else {}
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        q_raw = prop.get('q', prop.get('w', 0.0))
        load_type = prop.get('load_type', 'local')
        etype = prop.get('type', 'beam')

        # stiffness
        k_ax = E * A / L
        if I > 0 and etype == 'beam':
            k_b11 = 12.0 * E * I / L**3
            k_b12 = 6.0 * E * I / L**2
            k_b22 = 4.0 * E * I / L
            k_b22_off = 2.0 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0

        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ], dtype=float)

        # transform
        T = T_matrix(C, S)
        k_global = T.T @ k_local @ T

        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        K[np.ix_(dof, dof)] += k_global

        # --- distributed load conversion ---
        # Determine q_local (scalar in local -Y sense) from q_raw and load_type
        q_local = 0.0
        if abs(np.asarray(q_raw, dtype=float).sum()) > 0:  # quick check if non-zero
            if load_type == 'local':
                # q_raw expected to be scalar (local -Y). If array given, take second component.
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    # if vector-like, try to use second component as local-y (fallback)
                    try:
                        q_local = float(q_raw[1])
                    except Exception:
                        q_local = float(q_raw[0])
                else:
                    q_local = float(q_raw)
            elif load_type == 'global':
                # interpret q_raw as either scalar (qy) or 2-vector (qx, qy)
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    wgx = float(q_raw[0])
                    wgy = float(q_raw[1])
                else:
                    # scalar -> treat as global vertical component qy
                    wgx = 0.0
                    wgy = float(q_raw)
                # e_loc_y (local +Y) in global coords is [-S, C], local -Y is negative of that.
                e_loc_y = np.array([-S, C])
                w_global = np.array([wgx, wgy])
                # q_local must be scalar consistent with element convention (positive = local -Y)
                q_local = - np.dot(w_global, e_loc_y)
            else:
                raise ValueError(f"Unknown load_type '{load_type}' for element {e}")
        else:
            q_local = 0.0

        # equivalent nodal forces (for q_local) in local coordinates then transform to global
        if abs(q_local) > 1e-12:
            f_local = np.array([0.0, q_local*L/2.0, q_local*L**2/12.0,
                                0.0, q_local*L/2.0, -q_local*L**2/12.0], dtype=float)
            F[dof] += T.T @ f_local

    # --- point loads ---
    for dof, val in loads:
        F[int(dof)] += val

    # --- solve system ---
    all_dofs = np.arange(n_dofs)
    free = np.setdiff1d(all_dofs, constraints)
    u = np.zeros(n_dofs, dtype=float)
    K_ff = K[np.ix_(free, free)]
    F_ff = F[free]
    try:
        u[free] = np.linalg.solve(K_ff, F_ff)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("Stiffness matrix singular for free DOFs. Check constraints/model connectivity.") from exc

    reactions = (K @ u - F)[constraints]

    # --- postprocess (axial, shear, moments) ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L <= 0:
            raise ValueError(f"Element {e} has zero length (nodes {n1}, {n2}).")
        C, S = dx / L, dy / L

        prop = elem_props[e] if (elem_props is not None and e < len(elem_props)) else {}
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        q_raw = prop.get('q', prop.get('w', 0.0))
        load_type = prop.get('load_type', 'local')
        etype = prop.get('type', 'beam')

        # recompute q_local the same way as in assembly
        if abs(np.asarray(q_raw, dtype=float).sum()) > 0:
            if load_type == 'local':
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    try:
                        q_local = float(q_raw[1])
                    except Exception:
                        q_local = float(q_raw[0])
                else:
                    q_local = float(q_raw)
            else:  # global
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    wgx = float(q_raw[0]); wgy = float(q_raw[1])
                else:
                    wgx = 0.0; wgy = float(q_raw)
                e_loc_y = np.array([-S, C])
                q_local = - np.dot(np.array([wgx, wgy]), e_loc_y)
        else:
            q_local = 0.0

        # same stiffness again
        k_ax = E * A / L
        if I > 0 and etype == 'beam':
            k_b11 = 12.0 * E * I / L**3
            k_b12 = 6.0 * E * I / L**2
            k_b22 = 4.0 * E * I / L
            k_b22_off = 2.0 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0

        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ], dtype=float)

        # transformations
        R = np.array([[C, S, 0.0],
                      [-S, C, 0.0],
                      [0.0, 0.0, 1.0]])
        Tm = np.zeros((6,6))
        Tm[:3,:3] = R
        Tm[3:,3:] = R

        u_e = Tm.T @ u[dof]    # FIXED: use transpose for consistent local displacements

        if abs(q_local) > 1e-12:
            f_local = np.array([0.0, q_local*L/2.0, q_local*(L**2)/12.0,
                                0.0, q_local*L/2.0, -q_local*(L**2)/12.0], dtype=float)
        else:
            f_local = np.zeros(6, dtype=float)

        internal = k_local @ u_e - f_local

        axial_forces[e] = (E * A / L) * (u_e[3] - u_e[0])
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
        N_list.append(axial_forces[e])         # node 1 end
        N_list.append(axial_forces[e])         # node 2 end (same magnitude but opposite sign normally)

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

def frame2d_solver(nodes, elements, E, A, I,
                   point_loads=None, udl=None,   # udl: list of per-element dicts or None
                   constraints=None):
    """
    2D frame solver (Euler-Bernoulli) - linear elastic
    Units must be consistent (e.g. kN and m).
    Nodes : array-like (n_nodes x 2)
    Elements: array-like (n_elems x 2)  (0-based indices)
    E, A, I : scalars or arrays of length n_elems
    point_loads: list of [dof_index, value] OR dict node -> [Fx, Fy, M]
    udl: list of length n_elems with either None or:
         {'type':'local','q':[qx_local,qy_local]} OR
         {'type':'global','q':[qx_global,qy_global]}
         Convention used below: qy_local positive DOWN is typical for gravity.
    constraints: list/array of fixed global DOF indices (0-based),
                 or boolean mask length 3*n_nodes
    Returns: dict with keys:
      'u' : global DOF displacement vector (3*n_nodes,)
      'reactions' : full reactions vector (3*n_nodes,)
      'element_forces' : list of per-element dicts with local f vector [Fx_i,Fy_i,Mi,Fx_j,Fy_j,Mj]
      'info' : summary arrays: N_i, N_j, V_i, V_j, M_i, M_j
    """

    # --- convert inputs ---
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    ndof = 3 * n_nodes

    # expand E,A,I to arrays if scalars
    E_arr = np.full(n_elems, E, dtype=float) if np.isscalar(E) else np.array(E, dtype=float)
    A_arr = np.full(n_elems, A, dtype=float) if np.isscalar(A) else np.array(A, dtype=float)
    I_arr = np.full(n_elems, I, dtype=float) if np.isscalar(I) else np.array(I, dtype=float)

    # prepare global matrices
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # helper: local element stiffness (6x6) for frame element in local coords
    def k_local_frame(E, A, I, L):
        """6x6 local stiffness for beam-column 2D (u_x,u_y,theta per node) in order [ux_i,uy_i,ri, ux_j,uy_j,rj]"""
        k = np.zeros((6,6))
        # axial
        k[0,0] =  A*E / L
        k[0,3] = -A*E / L
        k[3,0] = -A*E / L
        k[3,3] =  A*E / L
        # bending
        k[1,1] =  12*E*I / L**3
        k[1,2] =   6*E*I / L**2
        k[1,4] = -12*E*I / L**3
        k[1,5] =   6*E*I / L**2

        k[2,1] =   6*E*I / L**2
        k[2,2] =   4*E*I / L
        k[2,4] =  -6*E*I / L**2
        k[2,5] =   2*E*I / L

        k[4,1] = -12*E*I / L**3
        k[4,2] =  -6*E*I / L**2
        k[4,4] =  12*E*I / L**3
        k[4,5] =  -6*E*I / L**2

        k[5,1] =   6*E*I / L**2
        k[5,2] =   2*E*I / L
        k[5,4] =  -6*E*I / L**2
        k[5,5] =   4*E*I / L
        return k

    # rotation 6x6
    def T_matrix(C, S):
        R = np.array([[C, S, 0],
                      [-S, C, 0],
                      [0,  0, 1]])
        T = np.zeros((6,6))
        T[0:3,0:3] = R
        T[3:6,3:6] = R
        return T

    # UDL -> equivalent local nodal load vector (6x) for element:
    # q_local = [qx_local (axial per length), qy_local (transverse per length)]
    # Sign convention: local positive y is UP. If you have qy positive DOWN (gravity),
    # pass qy_local as negative value (or code below negates if required).
    def udl_equivalent_local(q_local, L):
        qx, qy = q_local[0], q_local[1]
        f = np.zeros(6)
        # axial contributions (consistently distributed)
        f[0] = qx * L / 2.0
        f[3] = qx * L / 2.0
        # transverse (consistent) for beam: assuming qy is *positive upward*
        # For the usual case qy_positive_down, use qy_local = -q (so qy_local positive upward)
        f[1] = qy * L / 2.0
        f[2] = qy * L**2 / 12.0
        f[4] = qy * L / 2.0
        f[5] = - qy * L**2 / 12.0
        return f

    # assemble element stiffnesses and add equivalent nodal loads
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx = x2 - x1; dy = y2 - y1
        L = np.hypot(dx, dy)
        C = dx / L; S = dy / L

        Ke_local = k_local_frame(E_arr[e], A_arr[e], I_arr[e], L)
        T = T_matrix(C, S)
        Ke_global = T.T @ Ke_local @ T

        # assemble into global K
        dof_map = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        for a in range(6):
            for b in range(6):
                K[dof_map[a], dof_map[b]] += Ke_global[a,b]

        # distributed loads for this element
        if udl is not None and udl[e] is not None:
            entry = udl[e]
            if entry.get('type','local') == 'local':
                q_local = entry['q']  # assume given as [qx_local, qy_local_local_upwards]
                # if user passed qy as positive DOWN (common), they should pass q_local[1] negative.
                f_local = udl_equivalent_local(q_local, L)
            else:
                # global udl specified: convert to local first
                qg = np.array(entry['q'], dtype=float)  # [qx_global, qy_global] global axes (positive right/up)
                # local = R * global where R = [[C,S],[-S,C]]^T? For displacement rotation we used R above.
                # To convert global vector qg to local ql: q_local = [C*S?] Actually treat distributed vector at each point:
                # For forces per length: q_local = [ qx_local, qy_local ] where
                # qx_local =  C * qg[0] + S * qg[1]
                # qy_local = -S * qg[0] + C * qg[1]
                qx_local =  C * qg[0] + S * qg[1]
                qy_local = -S * qg[0] + C * qg[1]
                f_local = udl_equivalent_local([qx_local, qy_local], L)

            # transform to global equivalent nodal loads and add
            f_global = T.T @ f_local
            for a in range(6):
                F[dof_map[a]] += f_global[a]

    # apply point loads
    if point_loads is not None:
        # accept list of [dof,val] or dict node->[Fx,Fy,M]
        if isinstance(point_loads, dict):
            for node, vec in point_loads.items():
                i = int(node)
                F[3*i:3*i+3] += np.array(vec, dtype=float)
        else:
            for dof, val in point_loads:
                F[int(dof)] += float(val)

    # default constraints
    if constraints is None:
        constraints = np.array([], dtype=int)
    else:
        constraints = np.array(constraints, dtype=int)

    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, constraints)
    fixed = np.array(constraints, dtype=int)

    # Partition and solve
    if free.size > 0:
        Kff = K[np.ix_(free, free)]
        Ff = F[free]
        # solve Kff * u_f = Ff
        u = np.zeros(ndof)
        u_f = np.linalg.solve(Kff, Ff)
        u[free] = u_f
    else:
        u = np.zeros(ndof)

    # reactions
    reactions = K @ u - F

# --------------------------------------------------------------
# ELEMENT INTERNAL FORCES (local)
# --------------------------------------------------------------
    element_forces = []

    N_i = np.zeros(n_elems); N_j = np.zeros(n_elems)
    V_i = np.zeros(n_elems); V_j = np.zeros(n_elems)
    M_i = np.zeros(n_elems); M_j = np.zeros(n_elems)

    for e in range(n_elems):
     n1, n2 = elements[e]
     x1, y1 = nodes[n1]; x2, y2 = nodes[n2]
     dx = x2 - x1; dy = y2 - y1
     L = np.hypot(dx, dy)
     C = dx / L; S = dy / L

    # stiffness + rotation
     Ke_local = k_local_frame(E_arr[e], A_arr[e], I_arr[e], L)
     T = T_matrix(C, S)

    # extract local displacement vector
     dof_map = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
     u_e_global = u[dof_map]
     u_e_local  = T @ u_e_global

    # internal nodal forces from stiffness
     f_int_local = Ke_local @ u_e_local

    # distributed fixed-end actions
     f_eq_local = np.zeros(6)
     if udl is not None and udl[e] is not None:
        entry = udl[e]
        if entry.get("type", "local") == "local":
            q_local = entry["q"]
            f_eq_local = udl_equivalent_local(q_local, L)
        else:
            qg = np.array(entry["q"], float)
            qx_local =  C*qg[0] + S*qg[1]
            qy_local = -S*qg[0] + C*qg[1]
            f_eq_local = udl_equivalent_local([qx_local, qy_local], L)

    # total FE nodal forces (usual ones)
     f_total_local = f_int_local - f_eq_local

    # --------------------------------------------------------------
    # TRUE AXIAL FORCE (constant along element) RDM6 STYLE
    # N = EA/L * (u_jx_local - u_ix_local)
    # RDM6 convention: compression = negative
    # --------------------------------------------------------------
    uix = u_e_local[0]   # axial disp i
    ujx = u_e_local[3]   # axial disp j
    # RDM6 uses NEGATIVE for compression → keep that
    N_i[e] = uix
    N_j[e] = ujx     # SAME SIGN, no flip

    # Shear + bending from FEM nodal actions
    V_i[e] = f_total_local[1]
    M_i[e] = f_total_local[2]
    V_j[e] = f_total_local[4]
    M_j[e] = f_total_local[5]

    # --- build vertical N, V, M arrays ---
    N_list = []
    V_list = []
    M_list = []

    for e in range(n_elems):
        # N
        N_list.append(N_i)         # node 1 end
        N_list.append(N_j)         # node 2 end

        # V (shear)
        V_list.append(V_i)      # node 1 end
        V_list.append(V_j)      # node 2 end

        # M (bending moment)
        M_list.append(M_i)       # node 1 end
        M_list.append(M_j)       # node 2 end

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

import numpy as np

def FEM2D_frame2(nodes, elements, elem_props, loads, constraints, default_E=210e6):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 3 * n_nodes

    K = np.zeros((n_dofs, n_dofs), dtype=float)
    F = np.zeros(n_dofs, dtype=float)

    def T_matrix(C, S):
        R = np.array([[C, S, 0.0],
                      [-S, C, 0.0],
                      [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[:3, :3] = R
        T[3:, 3:] = R
        return T

    axial_forces = np.zeros((n_elems, 2), dtype=float)
    end_moments = np.zeros((n_elems, 2), dtype=float)
    shear_forces = np.zeros((n_elems, 2), dtype=float)

    # --- Assembly loop ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L <= 0:
            raise ValueError(f"Element {e} has zero length (nodes {n1}, {n2}).")
        C, S = dx / L, dy / L

        prop = elem_props[e] if (elem_props is not None and e < len(elem_props)) else {}
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        q_raw = prop.get('q', prop.get('w', 0.0))
        load_type = prop.get('load_type', 'local')
        etype = prop.get('type', 'beam')

        # stiffness
        k_ax = E * A / L
        if I > 0 and etype == 'beam':
            k_b11 = 12.0 * E * I / L**3
            k_b12 = 6.0 * E * I / L**2
            k_b22 = 4.0 * E * I / L
            k_b22_off = 2.0 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0

        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ], dtype=float)

        # transform
        T = T_matrix(C, S)
        k_global = T.T @ k_local @ T

        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        K[np.ix_(dof, dof)] += k_global

        # --- distributed load conversion to local q (scalar in local -Y sense) ---
        q_local = 0.0
        try:
            q_arr = np.asarray(q_raw, dtype=float)
            nonzero_q = np.abs(q_arr).sum() > 0
        except Exception:
            nonzero_q = False

        if nonzero_q:
            if load_type == 'local':
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    try:
                        q_local = float(q_raw[1])
                    except Exception:
                        q_local = float(q_raw[0])
                else:
                    q_local = float(q_raw)
            elif load_type == 'global':
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    wgx = float(q_raw[0]); wgy = float(q_raw[1])
                else:
                    wgx = 0.0; wgy = float(q_raw)
                e_loc_y = np.array([-S, C])   # local +Y in global coords
                w_global = np.array([wgx, wgy])
                q_local = - np.dot(w_global, e_loc_y)  # positive = local -Y
            else:
                raise ValueError(f"Unknown load_type '{load_type}' for element {e}")
        else:
            q_local = 0.0

        # equivalent nodal forces (local) then transform to global
        if abs(q_local) > 1e-12:
            f_local = np.array([0.0, q_local*L/2.0, q_local*L**2/12.0,
                                0.0, q_local*L/2.0, -q_local*L**2/12.0], dtype=float)
            F[dof] += T.T @ f_local

    # --- point loads ---
    for dof_idx, val in loads:
        F[int(dof_idx)] += val

    # --- solve system ---
    all_dofs = np.arange(n_dofs)
    free = np.setdiff1d(all_dofs, constraints)
    u = np.zeros(n_dofs, dtype=float)
    K_ff = K[np.ix_(free, free)]
    F_ff = F[free]
    try:
        u[free] = np.linalg.solve(K_ff, F_ff)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("Stiffness matrix singular for free DOFs. Check constraints/model connectivity.") from exc

    # reactions (make them oppose applied loads)
    reactions = (K @ u - F)[constraints]

    # --- postprocess (axial, shear, moments) ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L <= 0:
            raise ValueError(f"Element {e} has zero length (nodes {n1}, {n2}).")
        C, S = dx / L, dy / L

        prop = elem_props[e] if (elem_props is not None and e < len(elem_props)) else {}
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        q_raw = prop.get('q', prop.get('w', 0.0))
        load_type = prop.get('load_type', 'local')
        etype = prop.get('type', 'beam')

        # recompute q_local (same as assembly)
        try:
            q_arr = np.asarray(q_raw, dtype=float)
            nonzero_q = np.abs(q_arr).sum() > 0
        except Exception:
            nonzero_q = False

        if nonzero_q:
            if load_type == 'local':
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    try:
                        q_local = float(q_raw[1])
                    except Exception:
                        q_local = float(q_raw[0])
                else:
                    q_local = float(q_raw)
            else:
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    wgx = float(q_raw[0]); wgy = float(q_raw[1])
                else:
                    wgx = 0.0; wgy = float(q_raw)
                e_loc_y = np.array([-S, C])
                q_local = - np.dot(np.array([wgx, wgy]), e_loc_y)
        else:
            q_local = 0.0

        # local stiffness (same as assembly)
        k_ax = E * A / L
        if I > 0 and etype == 'beam':
            k_b11 = 12.0 * E * I / L**3
            k_b12 = 6.0 * E * I / L**2
            k_b22 = 4.0 * E * I / L
            k_b22_off = 2.0 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0

        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ], dtype=float)

        # transformation (global -> local)
        R = np.array([[C, S, 0.0],
                      [-S, C, 0.0],
                      [0.0, 0.0, 1.0]])
        Tm = np.zeros((6,6))
        Tm[:3,:3] = R
        Tm[3:,3:] = R

        # local equivalent nodal loads (same as assembly)
        if abs(q_local) > 1e-12:
            f_local = np.array([0.0, q_local*L/2.0, q_local*(L**2)/12.0,
                                0.0, q_local*L/2.0, -q_local*(L**2)/12.0], dtype=float)
        else:
            f_local = np.zeros(6, dtype=float)

        # local displacement vector (use transpose to be consistent with assembly)
        u_e = Tm.T @ u[dof]

        # internal vector (authoritative)
        internal = k_local @ u_e - f_local

        # extract axial/shear/moment from internal local vector
        N_i = internal[0]
        V_i = internal[1]
        M_i = internal[2]

        N_j = internal[3]
        V_j = internal[4]
        M_j = internal[5]

        # store axial as average of the two end axials BUT use civil sign: compression NEGATIVE
        axial_forces[e, 0] = N_i
        axial_forces[e, 1] = N_j
        # shears & moments: keep your RDM6-compatible flips
        shear_forces[e, 0] = -V_i
        shear_forces[e, 1] = V_j
        end_moments[e, 0] = -M_i
        end_moments[e, 1] = M_j

    # --- build vertical N, V, M arrays ---
    N_list = []
    V_list = []
    M_list = []

    for e in range(n_elems):
        # N
        N_list.append(axial_forces[e,0])         # node 1 end
        N_list.append(axial_forces[e,1])         # node 2 end

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

def FEM2D_frame3(nodes, elements, elem_props, loads, constraints, default_E=210e6):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 3 * n_nodes

    K = np.zeros((n_dofs, n_dofs), dtype=float)
    F = np.zeros(n_dofs, dtype=float)

    def T_matrix(C, S):
        R = np.array([[C, S, 0.0],
                      [-S, C, 0.0],
                      [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[:3, :3] = R
        T[3:, 3:] = R
        return T

    axial_forces = np.zeros(n_elems, dtype=float)
    end_moments = np.zeros((n_elems, 2), dtype=float)
    shear_forces = np.zeros((n_elems, 2), dtype=float)

    # --- Assembly loop ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L <= 0:
            raise ValueError(f"Element {e} has zero length (nodes {n1}, {n2}).")
        C, S = dx / L, dy / L

        prop = elem_props[e] if (elem_props is not None and e < len(elem_props)) else {}
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        q_raw = prop.get('q', prop.get('w', 0.0))
        load_type = prop.get('load_type', 'local')
        etype = prop.get('type', 'beam')

        # stiffness (local)
        k_ax = E * A / L
        if I > 0 and etype == 'beam':
            k_b11 = 12.0 * E * I / L**3
            k_b12 = 6.0 * E * I / L**2
            k_b22 = 4.0 * E * I / L
            k_b22_off = 2.0 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0

        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ], dtype=float)

        # transform
        T = T_matrix(C, S)
        k_global = T.T @ k_local @ T

        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        K[np.ix_(dof, dof)] += k_global

        # --- distributed load conversion to local q (scalar in local -Y sense) ---
        q_local = 0.0
        try:
            q_arr = np.asarray(q_raw, dtype=float)
            nonzero_q = np.abs(q_arr).sum() > 0
        except Exception:
            nonzero_q = False

        if nonzero_q:
            if load_type == 'local':
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    try:
                        q_local = float(q_raw[1])
                    except Exception:
                        q_local = float(q_raw[0])
                else:
                    q_local = float(q_raw)
            elif load_type == 'global':
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    wgx = float(q_raw[0]); wgy = float(q_raw[1])
                else:
                    wgx = 0.0; wgy = float(q_raw)
                e_loc_y = np.array([-S, C])   # local +Y in global coords
                w_global = np.array([wgx, wgy])
                q_local = - np.dot(w_global, e_loc_y)  # positive = local -Y
            else:
                raise ValueError(f"Unknown load_type '{load_type}' for element {e}")
        else:
            q_local = 0.0

        # equivalent nodal forces (local) then transform to global
        if abs(q_local) > 1e-12:
            f_local = np.array([0.0, q_local*L/2.0, q_local*L**2/12.0,
                                0.0, q_local*L/2.0, -q_local*L**2/12.0], dtype=float)
            F[dof] += T.T @ f_local

    # --- point loads ---
    for dof_idx, val in loads:
        F[int(dof_idx)] += val

    # --- solve system ---
    all_dofs = np.arange(n_dofs)
    free = np.setdiff1d(all_dofs, constraints)
    u = np.zeros(n_dofs, dtype=float)
    K_ff = K[np.ix_(free, free)]
    F_ff = F[free]
    try:
        u[free] = np.linalg.solve(K_ff, F_ff)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("Stiffness matrix singular for free DOFs. Check constraints/model connectivity.") from exc

    # FIXED: reactions sign (reactions oppose applied loads)
    reactions = -(K @ u - F)[constraints]

    # --- postprocess (axial, shear, moments) using consistent local internal vector ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        dof = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L <= 0:
            raise ValueError(f"Element {e} has zero length (nodes {n1}, {n2}).")
        C, S = dx / L, dy / L

        prop = elem_props[e] if (elem_props is not None and e < len(elem_props)) else {}
        A = prop.get('A', 1e-6)
        I = prop.get('I', 0.0)
        E = prop.get('E', default_E)
        q_raw = prop.get('q', prop.get('w', 0.0))
        load_type = prop.get('load_type', 'local')
        etype = prop.get('type', 'beam')

        # recompute q_local (same as assembly)
        try:
            q_arr = np.asarray(q_raw, dtype=float)
            nonzero_q = np.abs(q_arr).sum() > 0
        except Exception:
            nonzero_q = False

        if nonzero_q:
            if load_type == 'local':
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    try:
                        q_local = float(q_raw[1])
                    except Exception:
                        q_local = float(q_raw[0])
                else:
                    q_local = float(q_raw)
            else:
                if hasattr(q_raw, '__len__') and not isinstance(q_raw, (str, bytes)):
                    wgx = float(q_raw[0]); wgy = float(q_raw[1])
                else:
                    wgx = 0.0; wgy = float(q_raw)
                e_loc_y = np.array([-S, C])
                q_local = - np.dot(np.array([wgx, wgy]), e_loc_y)
        else:
            q_local = 0.0

        # local stiffness (same as assembly)
        k_ax = E * A / L
        if I > 0 and etype == 'beam':
            k_b11 = 12.0 * E * I / L**3
            k_b12 = 6.0 * E * I / L**2
            k_b22 = 4.0 * E * I / L
            k_b22_off = 2.0 * E * I / L
        else:
            k_b11 = k_b12 = k_b22 = k_b22_off = 0.0

        k_local = np.array([
            [ k_ax,  0.0,    0.0,   -k_ax,   0.0,    0.0],
            [ 0.0,  k_b11,  k_b12,   0.0,   -k_b11,  k_b12],
            [ 0.0,  k_b12,  k_b22,   0.0,   -k_b12,  k_b22_off],
            [-k_ax,  0.0,    0.0,    k_ax,   0.0,    0.0],
            [ 0.0, -k_b11, -k_b12,   0.0,    k_b11, -k_b12],
            [ 0.0,  k_b12,  k_b22_off, 0.0, -k_b12,  k_b22]
        ], dtype=float)

        # transformation (global -> local)
        R = np.array([[C, S, 0.0],
                      [-S, C, 0.0],
                      [0.0, 0.0, 1.0]])
        Tm = np.zeros((6,6))
        Tm[:3,:3] = R
        Tm[3:,3:] = R

        # local equivalent nodal loads (same as assembly)
        if abs(q_local) > 1e-12:
            f_local = np.array([0.0, q_local*L/2.0, q_local*(L**2)/12.0,
                                0.0, q_local*L/2.0, -q_local*(L**2)/12.0], dtype=float)
        else:
            f_local = np.zeros(6, dtype=float)

        # FIXED: use transpose of Tm here so local disp vector matches assembly convention
        u_e_local = Tm.T @ u[dof]

        # internal force vector in local coords (authoritative source)
        f_internal_local = k_local @ u_e_local - f_local

        # extract consistent end-values from the local internal vector
        N_i = f_internal_local[0]
        V_i = f_internal_local[1]
        M_i = f_internal_local[2]

        N_j = f_internal_local[3]
        V_j = f_internal_local[4]
        M_j = f_internal_local[5]

        # store axial as representative element axial (average of ends)
        axial_forces[e] = 0.5 * (N_i + N_j)

        # store shears (flip second end to match nodal sign continuity)
        shear_forces[e, 0] = V_i
        shear_forces[e, 1] = -V_j

        # store moments (flip second end so nodal sign matches adjacent element)
        end_moments[e, 0] = M_i
        end_moments[e, 1] = -M_j

    # --- build vertical N, V, M arrays ---
    N_list = []
    V_list = []
    M_list = []

    for e in range(n_elems):
        # N
        N_list.append(axial_forces[e])         # node 1 end
        N_list.append(axial_forces[e])         # node 2 end

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
