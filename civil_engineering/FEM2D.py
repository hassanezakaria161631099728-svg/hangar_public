import numpy as np
import matplotlib.pyplot as plt
import os

def FEM2D(A_h, A_v, A_d, nodes, elements, loads, constraints):
    """
    FEM2D - 2D Truss Solver with auto section assignment
    """
    # Young's modulus [kPa]
    E = 210e6  
    n_nodes = nodes.shape[0]
    n_dofs = 2 * n_nodes
    n_elems = elements.shape[0]
    K_global = np.zeros((n_dofs, n_dofs))
    F_global = np.zeros(n_dofs)   # 1D force vector
    # --- Assemble stiffness matrix ---
    for e in range(n_elems):
        n1, n2 = elements[e, :] - 1
        x1, y1 = nodes[n1, :]
        x2, y2 = nodes[n2, :]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        # Detect element type
        if abs(y1 - y2) < 1e-6:
            A = A_h
        elif abs(x1 - x2) < 1e-6:
            A = A_v
        else:
            A = A_d
        # Local stiffness
        k_local = (E * A / L) * np.array([
            [ C**2,  C*S,   -C**2, -C*S ],
            [ C*S,   S**2,  -C*S,  -S**2],
            [-C**2, -C*S,   C**2,  C*S ],
            [-C*S,  -S**2,  C*S,   S**2]
        ])
        dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
                K_global[dof_map[i], dof_map[j]] += k_local[i, j]
    # --- Apply loads ---
    for i in range(loads.shape[0]):
        dof, force = loads[i, :]
        F_global[int(dof) - 1] += force
    # --- Solve system ---
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, np.array(constraints) - 1)
    u = np.zeros(n_dofs)
    K_red = K_global[np.ix_(free_dofs, free_dofs)]
    F_red = F_global[free_dofs]
    u_red = np.linalg.solve(K_red, F_red)
    u[free_dofs] = u_red   # fill free dofs
    # --- Reactions ---
    R = K_global @ u - F_global
    reactions = R[np.array(constraints) - 1]
    # --- Axial forces ---
    axial_forces = np.zeros(n_elems)
    for e in range(n_elems):
        n1, n2 = elements[e, :] - 1
        x1, y1 = nodes[n1, :]
        x2, y2 = nodes[n2, :]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        if abs(y1 - y2) < 1e-6:
            A = A_h
        elif abs(x1 - x2) < 1e-6:
            A = A_v
        else:
            A = A_d
        dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        u_elem = u[dof_map]   # works fine (1D)
        T = np.array([-C, -S, C, S])
        axial_forces[e] = (E * A / L) * (T @ u_elem)
    return u, reactions, axial_forces

def FEM2D_frame(nodes, elements, elem_props, loads, constraints, default_E=210e6):
    """
    2D frame FEM solver (axial + bending + shear forces).
    Parameters
    ----------
    nodes : (n_nodes x 2) array
        Node coordinates [x, y] (m)
    elements : (n_elems x 2) array
        Connectivity (0-based)
    elem_props : list of dict
        Each dict can contain:
          - 'type': 'beam' or 'bar'
          - 'A': area (m^2)
          - 'I': second moment of area (m^4)
          - 'E': Young's modulus (kN/m²)
          - 'q': uniform load (kN/m, local -Y)
    loads : list of [dof_index, value]  (global DOF)
    constraints : list of constrained DOFs (0-based)
    default_E : float, optional
        Default Young's modulus (kN/m²)
    Returns
    -------
    u : ndarray (3*n_nodes,)
        Global displacement vector
    reactions : ndarray
        Reactions at constrained DOFs (kN)
    axial_forces : ndarray (n_elems,)
        Axial forces (kN, +tension)
    shear_forces : ndarray (n_elems, 2)
        Shear forces at node1 and node2 (kN)
    end_moments : ndarray (n_elems, 2)
        End moments at node1 and node2 (kN·m)
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 3 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    F = np.zeros(n_dofs)
    # transformation matrix
    def T_matrix(C, S):
        R = np.array([[C, S, 0],
                      [-S, C, 0],
                      [0, 0, 1]])
        T = np.zeros((6, 6))
        T[:3, :3] = R
        T[3:, 3:] = R
        return T
    axial_forces = np.zeros(n_elems)
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
        q = prop.get('q', 0.0)
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
        # equivalent nodal forces (for q)
        if abs(q) > 1e-12:
            f_local = np.array([0.0, q*L/2.0, q*L**2/12.0,
                                0.0, q*L/2.0, -q*L**2/12.0])
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
    # --- postprocess (axial, shear, moments) ---
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
        q = prop.get('q', 0.0)
        etype = prop.get('type', 'beam')
        # same stiffness again
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
        # transformations
        R = np.array([[C, S, 0],
                      [-S, C, 0],
                      [0, 0, 1]])
        Tm = np.zeros((6,6))
        Tm[:3,:3] = R
        Tm[3:,3:] = R
        u_e = Tm @ u[dof]
        if abs(q) > 1e-12:
            f_local = np.array([0.0, q*L/2.0, q*(L**2)/12.0,
                                0.0, q*L/2.0, -q*(L**2)/12.0])
        else:
            f_local = np.zeros(6)
        internal = k_local @ u_e - f_local
        axial_forces[e] = (E * A / L) * (u_e[3] - u_e[0])
        shear_forces[e, 0] = internal[1]
        shear_forces[e, 1] = internal[4]
        end_moments[e, 0] = internal[2]
        end_moments[e, 1] = internal[5]
    return u, reactions, axial_forces, shear_forces, end_moments


def FEM2D_frame_axial_ends(nodes, elements, elem_props, loads, constraints, default_E=210e6):
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
        p_axial_from_w = - S * w_global  # kN/m along local +x
        # note: the local transverse q_effective = q_local + (C * -w_global)
        # but if you intentionally provide q_local, treat it as local already.
        # Here we treat both: total local transverse load = q_local + (-C * w_global)
        q_from_w = - C * w_global
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
        axial_ends[e, 0] = internal[0]   # axial at node1 end (kN)
        axial_ends[e, 1] = internal[3]   # axial at node2 end (kN)
        axial_avg[e] = 0.5 * (axial_ends[e,0] + axial_ends[e,1])
        shear_forces[e, 0] = internal[1]
        shear_forces[e, 1] = internal[4]
        end_moments[e, 0] = internal[2]
        end_moments[e, 1] = internal[5]
    return u, reactions, axial_avg, axial_ends, shear_forces, end_moments

def organize_NVM(axial_ends, shear_forces, end_moments):
    """
    Builds a (2*n_elems, 3) ndarray with N, V, M for each element end.
    Row pattern:
        element0_start
        element0_end
        element1_start
        element1_end
        ...
    Columns: [N, V, M] in local coordinates.
    """
    n_elems = axial_ends.shape[0]
    table = np.zeros((2 * n_elems, 3))
    for e in range(n_elems):
        # start end
        table[2*e, 0] = axial_ends[e, 0]
        table[2*e, 1] = shear_forces[e, 0]
        table[2*e, 2] = end_moments[e, 0]
        # end end
        table[2*e+1, 0] = axial_ends[e, 1]
        table[2*e+1, 1] = shear_forces[e, 1]
        table[2*e+1, 2] = end_moments[e, 1]
    return table


def plot_frame(node_coords, elements, save_path=None):
    """
    Plot a 2D frame structure showing nodes and elements.
    If save_path is given, the figure will be saved as an image file.

    node_coords : list of (x, y)
    elements    : list of (i, j)
    save_path   : str or None
                  Example: 'frame_plot.png'
                  If only a filename is provided, it is saved in the current folder.
    """

    plt.figure(figsize=(8, 6))

    # Plot elements
    for e, (i, j) in enumerate(elements):
        xi, yi = node_coords[i]
        xj, yj = node_coords[j]

        plt.plot([xi, xj], [yi, yj], 'k-', linewidth=2)

        # Element label at midpoint
        xm = 0.5 * (xi + xj)
        ym = 0.5 * (yi + yj)
        plt.text(xm, ym, f"E{e}", color="blue", fontsize=10)

    # Plot nodes
    for n, (x, y) in enumerate(node_coords):
        plt.plot(x, y, 'ro')
        plt.text(x, y, f" {n}", color="red", fontsize=12)

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Frame Geometry: Nodes and Elements")

    # SAVE THE FIGURE IF REQUESTED
    if save_path is not None:
        # If user gives only filename: save in working folder
        save_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

