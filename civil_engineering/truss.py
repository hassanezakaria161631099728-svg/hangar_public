import pandas as pd
import numpy as np
from .utils import buckling1,beam4,lateral_torsional_buckling,buckling2,find_epf
from.FEM import FEM2D,create_X_horizontal_truss,plot_truss,animate_truss_deformation
def roof_bracing(hangarf, beamT, chI, chII_1):
    # Read building attributes
    ba = pd.read_excel(hangarf, sheet_name='building attributes')
    Lx = ba["Lx_m"].iloc[0]
    Ly = ba["Ly_m"].iloc[0]
    floor_height=ba["floor_height_m"].iloc[0]
    n_floors = ba["n_floors"].iloc[0]
    hp  = ba["hp_m"].iloc[0]
    h=floor_height*n_floors+hp
    l = Lx / 4
    t = Ly / 4
    e = min(Ly, 2*h)
    # Calls to external functions (need definitions)
    di, df, dI, dII, dIII, h1, h2, s1, s2, s3 = roof_bracing1(ba, e, h)
    si, sf, ha, hb, sI, sII, sIII = roof_bracing2(s1, s2, s3, dI, dII, ba, h)
    T1, qw1 = roof_bracing3(si, sf, sI, sII, sIII, di, df, dI, dII, dIII, chI)
    T2, qw2 = roof_bracing4(di, df, si, sf, chI)
    portal_frame = pd.read_excel(hangarf,sheet_name="portal frame")
    Fwi1, Fwi2, hi, ipe_class, heb_class, iymin, Ld = \
    roof_bracing5(qw1, qw2, h, portal_frame, l, t)
    # Beam data
    TIPE = pd.read_excel(beamT, sheet_name='IPE')
    THEB = pd.read_excel(beamT, sheet_name='HEB')
    Tcor = pd.read_excel(beamT, sheet_name='corniere')
    Trafter = beam4(ipe_class, 1, TIPE)
    Tcolumn  = beam4(heb_class, 1, THEB)
    Tdiagonal = beam4(iymin, 12, Tcor)
    T6, q, dF1, dF2 = equivalent_imperfection_load(Trafter, Lx, 'rafter')
    T7 = pd.DataFrame({'h1':[h1], 'h2':[h2], 'ha':[ha], 'hb':[hb], 'q':[q]})
    T8, T9, T10, p1, d1, d2, R1, R2, FV1, FV2 = \
    roof_bracing6(Fwi1, Fwi2,hi, dF1, dF2, qw1, qw2, Lx, Ly, Trafter, Tcolumn, Tdiagonal)
    Tin1    = pd.read_excel(chII_1, sheet_name='T9')
    Tpurlin = pd.read_excel(chII_1, sheet_name='Tpurlin')
    # Vérification des pannes (compression)
    Nsd = min(p1) * -100
    fy  = 2350
    Lfz,laz, lazb, alphaz, phiz, xiz,_,_ = buckling1(t, 'z', Tpurlin, 'double fixed', 1, 1)
    uz, kz = buckling2(lazb, Tpurlin, Nsd, xiz, fy, 'z')
    Mysdn   = Tin1["Mysdn"].iloc[0]
    Mzscorr = Tin1["Mzscorr"].iloc[0]
#    ult, klt = dever(lazb, Nsd, xiz, fy, Tpurlin)
    ult, klt,_,_, zg, k, c1, c2, lalt, laltb,_, philt, xlt, r=\
    lateral_torsional_buckling(t, lazb, 'double fixed', Nsd, xiz, fy, Tpurlin, Mzscorr, kz, Mysdn) 
#    zg, k, c1, c2, lalt, laltb, philt, xlt = dever2(t, Tpurlin, 'double fixed')
#   r = dever3(Nsd, xiz, fy, klt, Mysdn, xlt, kz, Mzscorr, Tpurlin)
    T11 = pd.DataFrame({
        'Lfz':[Lfz],'laz':[laz], 'lazb':[lazb], 'alphaz':[alphaz], 'phiz':[phiz], 'xiz':[xiz],
        'ult':[ult], 'klt':[klt], 'zg':[zg], 'k':[k], 'c1':[c1], 'c2':[c2],
        'lalt':[lalt], 'laltb':[laltb], 'philt':[philt], 'xlt':[xlt]
    })
    T12 = pd.DataFrame({
        'Nsd':[Nsd], 'uz':[uz], 'kz':[kz], 'Mysdn':[Mysdn], 'Mzscorr':[Mzscorr], 'dever':[r]
    })
    # Vérification des diagonales (cornieres)
    Ncsd = min(d1) * -1
    Ntsd = max(d2)
    R11  = R1[1]
    R21  = R2[1] * -1
    Lf,lambda_, lambdabar, alpha, phi, Xi, Nbrd, Ntrd = buckling1(Ld, 'corner', Tdiagonal,
                                                    'double fixed', Ncsd, Ntsd)
    T13 = pd.DataFrame({
       'Lf':[Lf], 'lambda':[lambda_], 'lambdabar':[lambdabar], 'alpha':[alpha], 'phi':[phi], 'Xi':[Xi],
        'Nbrd':[Nbrd], 'Ntrd':[Ntrd], 'Ncsd':[Ncsd], 'Ntsd':[Ntsd], 'R1':[R11], 'R2':[R21]
                       })
    return T1, T2, Trafter, Tcolumn, Tdiagonal, T6, T7, T8, T9, T10, T11, T12, T13,FV1,FV2

def roof_bracing1(ba, e, h3):
    """
    Python translation of MATLAB function roof_bracing1
    Parameters:
        ba : pandas.DataFrame (building attributes)
        e  : float
    Returns:
        di, df, dI, dII, dIII, h1, h2, s1, s2, s3
    """
    Lx = ba["Lx_m"].iloc[0]   # corresponds to ba{1,3}
    h0 = h3 - 2  # h3 full height of hangar on the summet
    a  = ba["slope_angle_deg"].iloc[0] 
    lf = Lx / 4
    d1 = lf / 2
    h1 = h0 + d1 * np.sin(np.deg2rad(a))
    d2 = d1 + lf
    h2 = h0 + d2 * np.sin(np.deg2rad(a))
    d3 = d2 + lf
    d4 = d3 + lf
    d5 = d4 + lf / 2
    di = [0, d1, d2, d3, d4]
    df = [d1, d2, d3, d4, d5]
    dI   = e / 5
    dII  = e
    dIII = Lx
    s1 = (h0 + h1) * lf / 4
    s2 = (h1 + h2) * lf / 2
    s3 = h2 * lf + (h3 - h2) * lf / 2
    return di, df, dI, dII, dIII, h1, h2, s1, s2, s3

def roof_bracing2(sin1, sin2, sin3, dI, dII, ba, h2):
    """
    Python translation of MATLAB function roof_bracing2
    Parameters:
        sin1, sin2, sin3 : float
        dI, dII : float
        ba : pandas.DataFrame (building attributes)
    Returns:
        si, sf, ha, hb, sI, sII, sIII
    """
    h0 = h2-2   # ba{1,2}
    a  = ba["slope_angle_deg"].iloc[0] 
    Lx = ba["Lx_m"].iloc[0]   
    # cumulative distances
    s1 = sin1
    s2 = s1 + sin2
    s3 = s2 + sin3
    s4 = s3 + sin2
    s5 = s4 + sin1
    si = [0, s1, s2, s3, s4]
    sf = [s1, s2, s3, s4, s5]
    # heights
    ha = h0 + dI * np.sin(np.deg2rad(a))
    hb = h0 + (Lx - dII) * np.sin(np.deg2rad(a))
    # surface areas (trapezoid approximations)
    sI   = (h0 + ha) * dI / 2
    sII  = sI + (ha + h2) * (Lx/2 - dI) / 2 + (h2 + hb) * (dII - Lx/2) / 2
    sIII = sII + (hb + h0) * (Lx - dII) / 2
    return si, sf, ha, hb, sI, sII, sIII

def roof_bracing3(si, sf, sI, sII, sIII, di, df, dI, dII, dIII, chI):
    """
    Python translation of MATLAB function roof_bracing3
    """
    # call helper functions
    sa, sb, sc,_ = roof_bracing31(si, sf, sI, sII, sIII)
    ba, bb, bc,_ = roof_bracing31(di, df, dI, dII, dIII)
    epfa, epfb, epfc, qwa, qwb, qwc, qw = roof_bracing32(sa, sb, sc, ba, bb, bc, chI)
    # labels
    c = [
        'si','sf','di','df',
        'sa','sb','sc',
        'ba','bb','bc',
        'epfa','epfb','epfc',
        'qwa','qwb','qwc','qw'
    ]
    var = ['attributes','M1','M2','M3','M4','M5']
    # assemble into a matrix (rows = attributes, cols = 5 values)
    M = np.vstack([
        si, sf, di, df,
        sa, sb, sc,
        ba, bb, bc,
        epfa, epfb, epfc,
        qwa, qwb, qwc, qw
    ])
    # build DataFrame similar to MATLAB table
    T = pd.DataFrame({
        'attributes': c,
        'M1': M[:,0],
        'M2': M[:,1],
        'M3': M[:,2],
        'M4': M[:,3],
        'M5': M[:,4]
    })
    return T, qw

def roof_bracing31(si, f, I, II, III):
    """
    Python translation of MATLAB function roof_bracing31
    Parameters:
        si, f : list or array (length 5)
        I, II, III : float
    Returns:
        a, b, c : numpy arrays (length 5)
        s : float (sum of all a+b+c)
    """
    a = np.zeros(5)
    b = np.zeros(5)
    c = np.zeros(5)
    for i in range(5):
        if f[i] < I:
            a[i] = f[i]
            b[i] = 0
            c[i] = 0
        elif si[i] < I and f[i] > I:
            a[i] = I - si[i]
            b[i] = f[i] - I
            c[i] = 0
        elif si[i] > I and f[i] < II:
            a[i] = 0
            b[i] = f[i] - si[i]
            c[i] = 0
        elif si[i] < II and f[i] > II:
            a[i] = 0
            b[i] = II - si[i]
            c[i] = f[i] - II
        elif si[i] > II and f[i] <= III:
            a[i] = 0
            b[i] = 0
            c[i] = f[i] - si[i]
    s = np.sum(a) + np.sum(b) + np.sum(c)
    return a, b, c, s

def roof_bracing32(sa, sb, sc, ba, bb, bc, chI):
    """
    Python translation of MATLAB function roof_bracing32
    Parameters:
        sa, sb, sc, ba, bb, bc : lists/arrays (length 5)
        chI : str (Excel file path)
    Returns:
        epfa, epfb, epfc, qwa, qwb, qwc, qw : numpy arrays (length 5)
    """
    # read Excel sheets
    Tin1  = pd.read_excel(chI, sheet_name='T2')
    Tin2  = pd.read_excel(chI, sheet_name='T3')
    qpze1 = Tin1["qpze_N_m_2"].iloc[0]   
    wi1   = Tin2["wi1"].iloc[0]   
    # coefficients
    epf1a  = np.array([-1.3] * 5)
    epf10a = np.array([-1.0] * 5)
    epf1b  = np.array([-1.0] * 5)
    epf10b = np.array([-0.8] * 5)
    epf1c  = np.array([-0.5] * 5)
    qpze = np.array([qpze1] * 5)
    wi   = np.array([wi1] * 5)
    # initialize arrays
    epfa = np.zeros(5); epfb = np.zeros(5); epfc = np.zeros(5)
    wea  = np.zeros(5); web  = np.zeros(5); wec  = np.zeros(5)
    qwa  = np.zeros(5); qwb  = np.zeros(5); qwc  = np.zeros(5); qw = np.zeros(5)
    # main loop
    for i in range(5):
        epfa[i] = find_epf(epf1a[i], epf10a[i], sa[i])
        wea[i]  = qpze[i] * epfa[i]
        qwa[i]  = (wea[i] - wi[i]) * ba[i]
        epfb[i] = find_epf(epf1b[i], epf10b[i], sb[i])
        web[i]  = qpze[i] * epfb[i]
        qwb[i]  = (web[i] - wi[i]) * bb[i]
        epfc[i] = find_epf(epf1c[i], epf1c[i], sc[i])
        wec[i]  = qpze[i] * epfc[i]
        qwc[i]  = (wec[i] - wi[i]) * bc[i]
        qw[i]   = (qwa[i] + qwb[i] + qwc[i]) / 10
    return epfa, epfb, epfc, qwa, qwb, qwc, qw

def roof_bracing4(di, df, si, sf, chI):
    """
    Python translation of MATLAB function roof_bracing4
    Parameters:
        di, df, si, sf : lists/arrays (length 5)
        chI : str (Excel file path)
    Returns:
        T : pandas.DataFrame
        qw : numpy array (length 5)
    """
    # read Excel sheets
    Tin  = pd.read_excel(chI, sheet_name='T7')
    Tin2  = pd.read_excel(chI, sheet_name='T8')
    qpze1 = Tin.iloc[0, 2]   # in{1,3}
    wi1   = Tin2.iloc[0, 4]   # in2{1,5}
    # initialize arrays
    s  = np.zeros(5)
    b  = np.zeros(5)
    epf = np.zeros(5)
    we  = np.zeros(5)
    qw  = np.zeros(5)
    for i in range(5):
        s[i] = sf[i] - si[i]
        b[i] = df[i] - di[i]
    # constants
    epf1  = np.array([1.0] * 5)
    epf10 = np.array([0.8] * 5)
    qpze = np.array([qpze1] * 5)
    wi   = np.array([wi1] * 5)
    # loop to compute epf, we, qw
    for i in range(5):
        epf[i] = find_epf(epf1[i], epf10[i], s[i])
        we[i]  = qpze[i] * epf[i]
        qw[i]  = (we[i] - wi[i]) * b[i] / 10
    # build DataFrame equivalent to MATLAB table
    c = ['di','df','si','sf','s','b','epf','qw']
    var = ['attributes','M1','M2','M3','M4','M5']
    M = np.vstack([di, df, si, sf, s, b, epf, qw])
    T = pd.DataFrame({
        'attributes': c,
        'M1': M[:,0],
        'M2': M[:,1],
        'M3': M[:,2],
        'M4': M[:,3],
        'M5': M[:,4]
    })
    return T, qw

def roof_bracing5(qw1, qw2, h3, portal_frame, l, t):
    """
    Python translation of MATLAB function roof_bracing5
    Parameters:
        qw1, qw2 : lists/arrays (length 5)
        ba : pandas.DataFrame (building attributes)
        key1, key2 : int (keys for section selection)
        l, t : float (geometry parameters)
    Returns:
        Fwi1, Fwi2 : numpy arrays (length 5)
        h : numpy array (length 5)
        val1, val2 : float
        iymin : float
        Ld : float
    """
    h1 = h3 - 2
    h2 = h3 - 1
    h  = np.array([h1, h2, h3, h2, h1])
    Fwi1 = np.zeros(5)
    Fwi2 = np.zeros(5)
    for i in range(5):
        Fwi1[i] = qw1[i] * h[i] / 2 / 100
        Fwi2[i] = qw2[i] * h[i] / 2 / 100
    # key-value lookup (MATLAB keyval equivalent)
    ipe_class, heb_class,_,_=get_beam_class( portal_frame,L_value=18, h_value=9)
    # diagonals
    Ld    = (l**2 + t**2) ** 0.5   # truss diagonal length
    iymin = Ld / 3                 # cm
    return Fwi1, Fwi2, h, ipe_class, heb_class, iymin, Ld

def get_beam_class(portal_frame, L_value=None, h_value=None):
    """
    Look up IPE and HEB classes for given L and h values in a portal_frame table.
    Returns both numeric classes and formatted names.
    """
    ipe_class = None
    heb_class = None
    IPE_name = None
    HEB_name = None
    # Lookup IPE for given L
    if L_value is not None:
        row = portal_frame.loc[portal_frame["L"] == L_value]
        if not row.empty:
            ipe_class = int(row["IPE"].values[0])
            IPE_name = f"IPE{ipe_class}"
        else:
            raise ValueError(f"L={L_value} not found in table")
    # Lookup HEB for given h
    if h_value is not None:
        row = portal_frame.loc[portal_frame["h"] == h_value]
        if not row.empty:
            heb_class = int(row["HEB"].values[0])
            HEB_name = f"HEB{heb_class}"
        else:
            raise ValueError(f"h={h_value} not found in table")
    # Print only if available
    print(f"{IPE_name or ''} {HEB_name or ''}".strip())
    return ipe_class, heb_class, IPE_name, HEB_name

def roof_bracing6(Fwi1, Fwi2, hi, dF1, dF2, qw1, qw2, Lx, Ly, Trafter, Tcolumn, Tdiagonal):
    """
    Equivalent Python translation of roof_bracing6 (MATLAB version).
    """
    # Initialize arrays
    F1 = np.zeros(5)
    F2 = np.zeros(5)
    dF = [dF2, dF1, dF1, dF1, dF2]
    # Compute forces F1, F2
    for i in range(5):
        F1[i] = 1.5 * Fwi1[i] - dF[i]
        F2[i] = 1.5 * Fwi2[i] + dF[i]
    # Build table T8
    c = ['hi', 'qw1', 'qw2', 'Fwi1', 'Fwi2', 'dF', 'F1', 'F2']
    var = ['attributes', 'M1', 'M2', 'M3', 'M4', 'M5']
    M = np.vstack([hi, qw1, qw2, Fwi1, Fwi2, dF, F1, F2])
    T8 = pd.DataFrame(M, index=c, columns=var[1:])
    T8.insert(0, 'attributes', c)
    # Geometry of truss
    l = Lx / 4
    t = Ly / 4
    n_panel=4 
    nodes, elements = create_X_horizontal_truss(n_panel, l, t)
    # Loads for each truss case
    loads1 = np.array([[1,F1[0]], [3,F1[1]], [5,F1[2]], [7,F1[3]], [9,F1[4]]]) 
    loads2 = np.array([[1,F2[0]], [3,F2[1]], [5,F2[2]], [7,F2[3]], [9,F2[4]]]) 
    constraints = np.array([0, 1, 8, 9])
    plot_truss(nodes, elements, loads1, constraints,load_scale=0.02)
    plot_truss(nodes, elements, loads2, constraints,load_scale=0.02)     
    # Areas *1e-4 convert cm2 to m2
    A_h = Trafter["A"].iloc[0] * 1e-4   # horizontal 
    A_v = Tcolumn["A"].iloc[0] * 1e-4    # vertical
    A_d = Tdiagonal["A"].iloc[0] * 2 * 1e-4   # diagonal two corners forming one diagonal
    # FEM solver (you must already have FEM2D4 translated into Python)
    u1, R1, FV1,elem_types1 = FEM2D(A_h, A_v, A_d, nodes, elements, loads1, constraints)
    _, R2, FV2,elem_types2 = FEM2D(A_h, A_v, A_d, nodes, elements, loads2, constraints)
 # --- Extract vertical (purlins) and diagonal elements automatically ---
    elem_types1 = np.array(elem_types1)  # ensure it's an array for easy filtering
    elem_types2 = np.array(elem_types2)
# Vertical elements (purlins)
    vert_idx = np.where(elem_types1 == 'V')[0]
    c1 = [f'P{i+1}' for i in range(len(vert_idx))]
    p1 = FV1[vert_idx]
    p2 = FV2[vert_idx]
    T9 = pd.DataFrame({'purlin': c1, 'wind1': p1, 'wind2': p2})
# Diagonal elements
    diag_idx = np.where(elem_types1 == 'D')[0]
    c2 = [f'D{i+1}' for i in range(len(diag_idx))]
    d1 = FV1[diag_idx]
    d2 = FV2[diag_idx]
    T10 = pd.DataFrame({'diagonal': c2, 'wind1': d1, 'wind2': d2})   
    return T8, T9, T10, p1, d1, d2, R1, R2,FV1,FV2

def equivalent_imperfection_load(T, L, cas):
    """
    Compute the equivalent imperfection load.
    Parameters:
        T   : pandas.DataFrame or list (input data, similar to MATLAB table)
        L   : float - length [m]
        cas : str   - 'traverse' (beam) or 'poteau' (column) 
    Returns:
        T2  : pandas.DataFrame - result summary
        q   : float             - equivalent imperfection load [kN/m]
        dF1 : float             - derived force 1 [kN]
        dF2 : float             - derived force 2 [kN]
    """
    fy = 2350  # yield strength (same as MATLAB)
    if cas == 'rafter':
        # Extract values (adjust indices to match MATLAB’s 1-based indexing)
        h = T["h"].iloc[0] / 100.0
        wply = T["wply"].iloc[0]  # column 18 in MATLAB → index 17 in Python
        Mcrd = wply * fy / 1.1 * 10**-4  # [kN·m]
        N = Mcrd / h
        T2 = pd.DataFrame({'h': [h], 'Mcrd': [Mcrd], 'N': [N]})
    elif cas == 'column':
        # Assume buckling1 is defined elsewhere and returns 6 values
        Lf,laz, labz, alphaz, phiz, Xiz, N,_ = buckling1(L, 'z', T, 'simplement appuye', 0, 0)
        T2 = pd.DataFrame({
            'Lf':[Lf],
            'laz': [laz],
            'labz': [labz],
            'alphaz': [alphaz],
            'phiz': [phiz],
            'Xiz': [Xiz],
            'N': [N]
        })
    else:
        raise ValueError("Invalid case. Must be 'rafter' or 'column'.")
    # Common part
    kr = np.sqrt(0.2 + 1/3)
    q = 3 * N * (kr + 0.2) / (60 * L)  # [kN/m]
    if cas == 'rafter':
        lf = L / 4
        dF1 = q * lf
        dF2 = dF1 / 2
    elif cas == 'column':
        dF1 = q * L / 2
        dF2 = dF1    
    return T2, q, dF1, dF2

def wall_bracing(beamT, chIII_1, hangarf):
    # Read Excel sheets
    ba = pd.read_excel(hangarf, sheet_name='building attributes')
    Tin1 = pd.read_excel(chIII_1, sheet_name='Tcolumn')
    THEA = pd.read_excel(beamT, sheet_name='HEA')
    Tin2 = pd.read_excel(chIII_1, sheet_name='T13')
    Tcorner = pd.read_excel(beamT, sheet_name='corniere')
    # Extract key values
    floor_height=ba["floor_height_m"].iloc[0]
    n_floors=ba["n_floors"].iloc[0]
    hp=ba["hp_m"].iloc[0] 
    h = floor_height*n_floors+hp
    Ly = ba["Ly_m"].iloc[0]
    # Equivalent of: [T1, q, df1, df2] = chaeqimp(in, h-2, 'poteau')
    T1, q, df1, df2 = equivalent_imperfection_load(Tin1, h - 2, 'column')
    # Build T2 table
    T2 = pd.DataFrame({'q': [q], 'df1': [df1], 'df2': [df2]})
    # Eave purlin class determination
    t = Ly / 4
    izmin = t / 300  # slenderness check
    TEave_purlin = beam4(izmin, 16, THEA)  # HEA beam
    # Diagonal braces class determination
    L = np.sqrt((h - 2)**2 + (t / 2)**2)
    iymin = L / 3
    Tdiagonal = beam4(iymin, 12, Tcorner)  # corniere
    # Vertical beam (column) data again
    Tcolumn=Tin1 
    #Tcolumn = pd.read_excel(chIII_1, sheet_name='Tcolumn')
    # Geometry (nodes and elements)
    nodes = np.array([
        [0, 0],
        [t, 0],
        [0, h - 2],
        [t / 2, h - 2],
        [t, h - 2]
    ])
    elements = np.array([
        [2, 3], [3, 4],  # top
        [0, 2], [1, 4],  # verticals
        [0, 3], [3, 1]   # diagonals
    ])
    R1 = Tin2["R1"].iloc[0]
    R2 = Tin2["R2"].iloc[0]
    loads1 = np.array([
        [0, -1 * df2],
        [4, -1 * (df1 + R1)]
    ])
    loads2 = np.array([
        [0, df2],
        [4, R2 + df1]
    ])
    constraints = [0, 1, 2, 3]
    plot_truss(nodes,elements,loads1,constraints,load_scale=0.01)
    plot_truss(nodes,elements,loads2,constraints,load_scale=0.01)
    # Cross-sectional areas (m²)
    A_h = TEave_purlin["A"].iloc[0] * 1e-4
    A_v = Tcolumn["A"].iloc[0] * 1e-4 # section of column
    A_d = Tdiagonal["A"].iloc[0] * 1e-4
    # Finite element results
    _, _, FV1,_ = FEM2D(A_h, A_v, A_d, nodes, elements, loads1, constraints)
    _, _, FV2,_ = FEM2D(A_h, A_v, A_d, nodes, elements, loads2, constraints)
    c = ['s1', 's2', 'poteau', 'potelet', 'd1', 'd2']
    T3 = pd.DataFrame({'element': c, 'vent1': FV1, 'vent2': FV2})
    Ncsdps = -FV2[0]
    Ncsdd = FV2[4]
    # Verification — Eave purlin
    Lfh,laz, lazb, alphaz, phiz, Xiz, Nbrdz,_ = buckling1(t, 'z', TEave_purlin, 'simplement appuye', Ncsdps, 1)
    # Verification — Diagonals
    Lfd,lay, layb, alphay, phiy, Xiy, Nbrdy,_ = buckling1(L, 'corner', Tdiagonal, 'simplement appuye', Ncsdd, 1)
    T4 = pd.DataFrame({
        't': [t],
        'Lfh':[Lfh], 
        'laz': [laz],
        'lazb': [lazb],
        'alphaz': [alphaz],
        'phiz': [phiz],
        'Xiz': [Xiz],
        'Nbrdz': [Nbrdz],
        'Ncsdps': [Ncsdps]
    })
    T5 = pd.DataFrame({
        'L': [L],
        'Lfd':[Lfd],
        'lay': [lay],
        'layb': [layb],
        'alphay': [alphay],
        'phiy': [phiy],
        'Xiy': [Xiy],
        'Nbrdy': [Nbrdy],
        'Ncsdd': [Ncsdd]
    })
    return T1, T2, TEave_purlin, Tdiagonal, T3, T4, T5
