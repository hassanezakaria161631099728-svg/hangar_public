import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Arc

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


def plot_frame(nodes, elements, elem_props, loads, constraints, scale_load=0.3):
    """
    Visual frame plot:
    - nodes
    - elements
    - supports
    - nodal loads (offset from node for clarity)
    - distributed loads (suppressed when q=0)
    """

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))

    # ------------------------------------------------------------
    # 1. Draw Elements
    # ------------------------------------------------------------
    for n1, n2 in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)

    # ------------------------------------------------------------
    # 2. Draw Nodes
    # ------------------------------------------------------------
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'ko')
        ax.text(x + 0.05, y + 0.05, f"{i}", fontsize=9)

    # ------------------------------------------------------------
    # 3. Supports
    # ------------------------------------------------------------
    pinned = {}

    for c in constraints:
        node = c // 3
        dof = c % 3

        if node not in pinned:
            pinned[node] = [False, False, False]

        pinned[node][dof] = True

    for node, (ux, uy, rz) in pinned.items():
        x, y = nodes[node]
        if ux and uy and rz:
            ax.plot(x, y, 's', color="purple", markersize=10)  # fixed
        elif ux and uy:
            ax.plot(x, y, marker=(3, 0, -90), color="blue", markersize=13)  # pinned
        elif uy:
            ax.plot(x, y, 'o', color="green", markersize=10)  # roller-Y

    # ------------------------------------------------------------
    # 4. Nodal Loads (offset arrow)
    # ------------------------------------------------------------
    offset = 0.2  # Offset from node

    for dof, P in loads:
        node = dof // 3
        comp = dof % 3
        x, y = nodes[node]

        if comp == 0:  # Fx
            ax.arrow(x + offset*np.sign(P), y,
                     scale_load*np.sign(P), 0,
                     head_width=0.08, fc='red', ec='red')

        elif comp == 1:  # Fy
            ax.arrow(x, y + offset*np.sign(P),
                     0, scale_load*np.sign(P),
                     head_width=0.08, fc='red', ec='red')

        elif comp == 2:  # Moment
            r = 0.25
            theta2 = -180 if P < 0 else 180
            arc = Arc((x, y), r, r, theta1=0, theta2=theta2,
                      color='red', linewidth=2)
            ax.add_patch(arc)

            ax.arrow(x + r/2, y,
                     -0.0001*np.sign(P), 0,
                     head_width=0.08, fc='red', ec='red')

    # ------------------------------------------------------------
    # 5. Distributed Loads (skip if q=0)
    # ------------------------------------------------------------
    for e, prop in enumerate(elem_props):
        if 'q' not in prop and 'w' not in prop:
            continue

        q_raw = prop.get('q', prop.get('w', 0.0))

        # 📌 Skip if scalar or vector q is zero
        if hasattr(q_raw, "__len__"):
            if np.allclose(q_raw, 0):
                continue
        else:
            if abs(q_raw) < 1e-12:
                continue

        load_type = prop.get('load_type', 'local')

        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        C, S = dx/L, dy/L

        e_y_local = np.array([-S, C])

        if load_type == "local":
            q = float(q_raw)
            w_vec = q * e_y_local
        else:
            w_vec = np.array(q_raw, dtype=float)

        n_arrows = 5
        for i in range(1, n_arrows):
            t = i / n_arrows
            xp = x1 + t * dx
            yp = y1 + t * dy

            ax.arrow(xp, yp,
                     scale_load * w_vec[0],
                     scale_load * w_vec[1],
                     head_width=0.08, fc='red', ec='red')

    # ------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Frame Model with Supports & Loads")

    plt.show()

def plot_frame2(nodes, elements, elem_props, loads, constraints, scale_load=0.3):

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


def FEM2D_frame5(nodes, elements, elem_props, loads, constraints):
    """
    FEM2D_frame - 2D frame solver (3 DOF per node)
    ------------------------------------------------
    Converts your 2-DOF truss solver to a general 3-DOF/frame solver.
    This function DOES NOT execute any test cases itself — paste & run locally.
    
    Inputs
    ------
    nodes : array_like (n_nodes x 2)  # [[x,y], ...]
    elements : array_like (n_elems x 2) # [[n1,n2], ...], 0-based
    elem_props : list of dicts (len = n_elems)
                 each dict must have keys: 'A', 'I', 'E'
                 optional: 'q' (UDL, not applied in this version)
    loads : list/array of [global_DOF_index, value]  # global DOF indices, 0-based
            DOF ordering per node: [u, v, theta] -> indices [3*i,3*i+1,3*i+2]
    constraints : list/array of constrained global DOF indices (0-based)
    
    Returns
    -------
    u : ndarray (n_dofs,)       # [u0, v0, th0, u1, v1, th1, ...]
    reactions : ndarray         # reactions corresponding to 'constraints' ordering
    N : ndarray (n_elems,2)     # axial forces at element ends in local coords (kN)
    V : ndarray (n_elems,2)     # shear forces at element ends in local coords (kN)
    M : ndarray (n_elems,2)     # bending moments at element ends in local coords (kN*m)
    
    Conventions
    -----------
    - Local element axis: x' from node1 -> node2, y' positive upwards (right-hand).
    - N positive = TENSION (pulling outward from the node).
    - V positive = upward in local +y.
    - M positive = counterclockwise (CCW) at the node.
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 3 * n_nodes

    # Global matrices
    K_global = np.zeros((n_dofs, n_dofs))
    F_global = np.zeros(n_dofs)

    # Element assembly (store info for post-processing)
    elem_info = [None] * n_elems
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx = x2 - x1
        dy = y2 - y1
        L = np.hypot(dx, dy)
        if L <= 0:
            raise ValueError(f"Element {e} has zero length.")

        C = dx / L
        S = dy / L

        A = float(elem_props[e]['A'])
        I = float(elem_props[e]['I'])
        E = float(elem_props[e]['E'])
        q = float(elem_props[e].get('q', 0.0))  # accepted but not applied here

        # Local stiffness (6x6) [u1 v1 th1 u2 v2 th2] in local coords
        EA_L = E * A / L
        EI = E * I
        k_local = np.zeros((6, 6))
        # axial
        k_local[0, 0] =  EA_L
        k_local[0, 3] = -EA_L
        k_local[3, 0] = -EA_L
        k_local[3, 3] =  EA_L
        # bending/shear (Euler-Bernoulli)
        k_local[1, 1] =  12 * EI / L**3
        k_local[1, 2] =   6 * EI / L**2
        k_local[1, 4] = -12 * EI / L**3
        k_local[1, 5] =   6 * EI / L**2

        k_local[2, 1] =   6 * EI / L**2
        k_local[2, 2] =   4 * EI / L
        k_local[2, 4] =  -6 * EI / L**2
        k_local[2, 5] =   2 * EI / L

        k_local[4, 1] = -12 * EI / L**3
        k_local[4, 2] =  -6 * EI / L**2
        k_local[4, 4] =  12 * EI / L**3
        k_local[4, 5] =  -6 * EI / L**2

        k_local[5, 1] =   6 * EI / L**2
        k_local[5, 2] =   2 * EI / L
        k_local[5, 4] =  -6 * EI / L**2
        k_local[5, 5] =   4 * EI / L

        # transformation matrix (6x6)
        r = np.array([[ C,  S, 0],
                      [-S,  C, 0],
                      [ 0,  0, 1]])
        T = np.zeros((6,6))
        T[0:3,0:3] = r
        T[3:6,3:6] = r

        # map to global and assemble
        k_global_elem = T.T @ k_local @ T
        dof_map = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        for i_loc, i_glob in enumerate(dof_map):
            for j_loc, j_glob in enumerate(dof_map):
                K_global[i_glob, j_glob] += k_global_elem[i_loc, j_loc]

        elem_info[e] = {'L':L, 'C':C, 'S':S, 'T':T, 'k_local':k_local, 'dof_map':dof_map, 'q':q}

    # Apply nodal loads (global DOF indices)
    if loads is not None:
        for dof, val in loads:
            F_global[int(dof)] += float(val)

    # Solve system
    all_dofs = np.arange(n_dofs)
    constrained = np.asarray(constraints, dtype=int)
    free = np.setdiff1d(all_dofs, constrained)

    if free.size == 0:
        raise ValueError("No free DOFs to solve for.")

    K_ff = K_global[np.ix_(free, free)]
    F_f = F_global[free]
    u = np.zeros(n_dofs)
    u[free] = np.linalg.solve(K_ff, F_f)

    # reactions
    R_all = K_global @ u - F_global
    reactions = R_all[constrained]

    # Postprocess internal forces at element ends (local)
    # f_local = k_local @ u_local  -> nodal forces (element on structure).
    # internal section forces are -f_local components:
    N = np.zeros((n_elems, 2))
    V = np.zeros((n_elems, 2))
    M = np.zeros((n_elems, 2))

    for e in range(n_elems):
        info = elem_info[e]
        u_elem_global = u[info['dof_map']]
        u_local = info['T'] @ u_elem_global  # local nodal displacements
        # Note: q-equivalent nodal loads are NOT included; if q != 0, you must
        # add its equivalent nodal vector f_eq_local here and to global F
        f_local = info['k_local'] @ u_local  # nodal forces (element on nodes) [Fx1,Fy1,M1,Fx2,Fy2,M2]
        # internal positive forces (element resisting loads) are the negative of f_local
        N[e,0] = -f_local[0]  # node1 axial (tension +)
        V[e,0] = -f_local[1]  # node1 shear (upward +)
        M[e,0] = -f_local[2]  # node1 moment (CCW +)
        N[e,1] = -f_local[3]  # node2 axial
        V[e,1] = -f_local[4]  # node2 shear
        M[e,1] = -f_local[5]  # node2 moment

    return u, reactions, N, V, M
