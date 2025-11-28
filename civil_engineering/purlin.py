import numpy as np
import pandas as pd
from .utils import (comb,snow,beam3,find_vplrd,bending,resistance,lateral_torsional_buckling,find_epf,
moment_shear_defection)

def purlin(b1,b2,hangarf,chI,beamT):
    # === Call purlin0 ===
    ba,geo,Troof1,Troof2,in1,in2,in3,in6,in7,in8=purlin0(hangarf,chI)
    bt2 = ba["bt2"].iloc[0]   
    Lx = ba["Lx_m"].iloc[0]   
    Ly = ba["Ly_m"].iloc[0]   
    # === Snow loads ===
    s = snow(geo,ba,Ly,Lx,Ly)
    bp = 1.355  # largeur d'influence m
    # === Wind loads from panne1 ===
    qw1 = purlin1(ba,bt2,b1,Troof1,in1,in2,in3,bp)
    qw2 = purlin1(ba,bt2,b2,Troof2,in6,in7,in8,bp)
    qws = np.concatenate([np.atleast_1d(qw1), np.atleast_1d(qw2)])/10#convert to daN/m
    qw = np.max(np.abs(qws))    
    t = Ly / 4.0
    alpha = ba["slope_angle_deg"].iloc[0]   # ba{1,5}
    # === Profile characteristics ===
    indata = pd.read_excel(beamT,sheet_name="IPE")
    Tpanne,Iymin,lb = beam3(t,qw,indata)
    # === Loads on the most unfavorable panne ===
    qgc = 15.21        # daN/m (covering weight)
    qgp = Tpanne.iloc[0, 7]  # Tpanne{1,8}
    qg = qgp + qgc * bp
    qs = s * bp
    Q = 100.0  # daN live load (not combined with climate)
    T2 = pd.DataFrame({"bp": [bp],"Iymin": [Iymin],"lb": [lb],
    "qw1_1": [qw1[0]],"qw1_2": [qw1[1]],"qw2": [qw2],"qg": [qg],"qs": [qs],"Q": [Q]})
    # === Load components along y and z ===
    qwp = qw1[1] / 10.0
    qsy = qs * np.cos(np.deg2rad(alpha))
    qgy = qg * np.cos(np.deg2rad(alpha))
    Qy = Q * np.cos(np.deg2rad(alpha))
    qsz = qs * np.sin(np.deg2rad(alpha))
    qgz = qg * np.sin(np.deg2rad(alpha))
    Qz = Q * np.sin(np.deg2rad(alpha))
    loads = pd.DataFrame({"axis": ["y", "z"],
        "Q": [Qy, Qz],"qw": [qw, 0],"qwp": [qwp, 0],"qs": [qsy, qsz],"qg": [qgy, qgz]})
    # === Call purlin2 ===
    Tcomb = pd.read_excel(beamT,sheet_name="comb_psi")
    acp,combdel,combV,combM,delmax,del2,Vysdn,Vzsdp,Mysdn,Mzscorr,Mysdp,Mzsdp=purlin2(
    Tpanne,qgy,qgz,qw,Qy,Qz,qwp,qsy,qsz,Ly,Tcomb)
    # === Verification (ELU & ELS) ===
    Avy, Vplrdy = find_vplrd("y", Tpanne, "IPE", Vysdn)
    Avz, Vplrdz = find_vplrd("z", Tpanne, "IPE", Vzsdp)
    lbmax, lb2 = bending(del2, delmax, t)
    r1 = resistance(Tpanne, Mysdp, Mzsdp)
    T8 = pd.DataFrame({"delmax": [delmax],"L/200": [lbmax],"del2": [del2],
        "L/250": [lb2],"Vysd": [Vysdn],"Avy": [Avy],"Vplrdy": [Vplrdy],"Vzsd": [Vzsdp],
        "Avz": [Avz],"Vplrdz": [Vplrdz],"Mysdp": [Mysdp],"Mzsdp": [Mzsdp],"resis": [r1]})
    # === Dever ===
    fy = 2350
    ult,klt,L1,Lfz,zg,k,c1,c2,lalt,laltb,alphalt,philt,xlt,rd=lateral_torsional_buckling(
    t,0,"double fixed",0,1,fy,Tpanne,Mzscorr,1,Mysdn)
    T9 = pd.DataFrame({
        "ult": [ult],"klt": [klt],"L1": [L1],"Lfz": [Lfz],"zg": [zg],"k": [k],
        "c1": [c1],"c2": [c2],"lalt": [lalt],"laltb": [laltb],"alphalt": [alphalt],
        "philt": [philt],"xlt": [xlt],"Mysdn": [Mysdn],"Mzscorr": [Mzscorr],"rd": [rd]})
    # === Lierne ===
    T10 = purlin_tie_rod(t,qgp,qgc,s,alpha,Lx,Tcomb)
    return Tpanne, T2, loads, acp, combdel, combV, combM, T8, T9, T10

def purlin0(hangarf, chapterI):
    geo = pd.read_excel(hangarf, sheet_name="geography attributes")
    ba = pd.read_excel(hangarf, sheet_name="building attributes")
    in1 = pd.read_excel(chapterI, sheet_name="T1")
    in2 = pd.read_excel(chapterI, sheet_name="T2")
    in3 = pd.read_excel(chapterI, sheet_name="T3")
    in6 = pd.read_excel(chapterI, sheet_name="T6")
    in7 = pd.read_excel(chapterI, sheet_name="T7")
    in8 = pd.read_excel(chapterI, sheet_name="T8")
    Troof1 = pd.read_excel(chapterI, sheet_name="Troof1")
    Troof2 = pd.read_excel(chapterI, sheet_name="Troof2")
    return ba,geo,Troof1,Troof2,in1,in2,in3,in6,in7,in8

def purlin1(ba, bt, b, Troof1, T1, T2, T3, bp):
    # Extract values from input tables (adjusted for 0-based indexing in Python)
    L = ba["Lx_m"].iloc[0]   
    T = ba["Ly_m"].iloc[0]   
    t = T / 4.0
    e   = T1["e_m"].iloc[0]          # table1{1,4}
    qpze = T2["qpze_N_m_2"].iloc[0]  # table2{1,3}
    ipf = T3["ipf1"].iloc[0]         # table3{1,3}
    if bt == 'gable':
        if b == T:  # wind 1 direction (theta = 0)
            bpf = bp
            bpg = bpf
            bph = 0
            bf = e / 4.0
            if t <= bf:
                tf = t
                tg = 0
                th = t
            else:
                tf = bf
                tg = t - bf
                th = t
        elif b == L:  # wind 2 direction (theta = 90)
            df = e / 10.0
            tf = df
            th = t - df
            tg = df
            bpf = bp
            bpg = 0
            bph = bp
        else:
            raise ValueError("b has to be either L or T")
        # --- Surface forces ---
        sf = bpf * tf
        sg = bpg * tg
        sh = bph * th
        epff = find_epf(Troof1.iloc[0, 1], Troof1.iloc[1, 1], sf)
        epfg = find_epf(Troof1.iloc[0, 2], Troof1.iloc[1, 2], sg)
        epfh = find_epf(Troof1.iloc[0, 3], Troof1.iloc[1, 3], sh)
        qwf = qpze * (epff - ipf) * bpf
        qwg = qpze * (epfg - ipf) * bpg
        qwh = qpze * (epfh - ipf) * bph
        if b == T:  # wind 1 direction
            qw1 = qwf
            t1 = tf
            qw2 = qwg
            t2 = tg
            epfJp = Troof1.loc[Troof1["factors"] == "epf+", "J"].values[0]
            qwJp = qpze * (epfJp - ipf) * bp
        elif b == L:  # wind 2 direction
            qw1 = qwf
            t1 = tf
            qw2 = qwh
            t2 = th
        else:
            raise ValueError("b has to be either L or T")
        # --- Equivalent load ---
        qweq = (qw1 * t1 + qw2 * t2) / t
        if b == T:
           qw = np.array([qweq, qwJp])
        elif b == L:
           qw = np.array([qweq])
        else:
            raise ValueError("b has to be either L or T")
    else:
        raise ValueError("unexpected bt")
    return qw

def purlin2(T1,qgy,qgz,qw,Qy,Qz,qwp,qsy,qsz,Ly,Tcomb):
    Iy = T1["Iy"].iloc[0]  # cm^4
    # --- Call purlin21 ---
    (Mgy, Vgy, delgy, Mw, Vw, delw, MQy, VQy, delQy,
     Mgz, Vgz, MQz, VQz, Mwp, Vwp, delwp,
     Msy, Vsy, delsy, Msz, Vsz) = purlin21(qgy, qgz, qw, Qy, Qz,qwp, qsy, qsz, Iy, Ly)
    # --- Call purlin22 ---
    acp = purlin22(Mgy, Vgy, delgy,
                  Mw, Vw, delw,
                  MQy, VQy, delQy,
                  Mgz, Vgz, MQz, VQz,
                  Mwp, Vwp, delwp,
                  Msy, Vsy, delsy,
                  Msz, Vsz,
                  qgy, qgz, qw, Qy, Qz, qwp, qsy, qsz)
    # --- Call purlin23 (ELS) ---
    combdel, delmax, del2 = purlin23(delgy, delw, delQy, delwp, delsy,Tcomb)
    # --- Call purlin24 (ELU, shear) ---
    combV, Vysdn, _, _, Vzsdp = purlin24(Vgy, Vgz, Vw, VQy, Vsy, Vwp, VQz, Vsz,Tcomb)
    # --- Call purlin24 (ELU, moment) ---
    combM, Mysdn, Mzscorr, Mysdp, Mzsdp = purlin24(Mgy,Mgz,Mw,MQy,Msy,Mwp,MQz,Msz,Tcomb)
    return acp, combdel, combV, combM, delmax, del2, Vysdn, Vzsdp, Mysdn, Mzscorr, Mysdp, Mzsdp

def purlin21(qgy,qgz,qw,Qy,Qz,qwp,qsy,qsz,Iy,Ly):
    t = Ly / 4.0
    L = t
    lt1 = "uniform"
    lt2 = "double ponctuel"
    # Loads on y-axis
    Mgy, Vgy, delgy = moment_shear_defection(lt1, "y", qgy * 1e-5, L, Iy * 1e-8)
    Mw, Vw, delw    = moment_shear_defection(lt1, "y", qw   * 1e-5, L, Iy * 1e-8)
    MQy, VQy, delQy = moment_shear_defection(lt2, "y", Qy   * 1e-5, L, Iy * 1e-8)
    # Loads on z-axis
    Mgz, Vgz, _     = moment_shear_defection(lt1, "z", qgz * 1e-5, L, Iy)
    MQz, VQz, _     = moment_shear_defection(lt2, "z", Qz * 1e-5, L, Iy)
    # Wind punctual load
    Mwp, Vwp, delwp = moment_shear_defection(lt1, "y", qwp * 1e-5, L, Iy * 1e-8)
    # Snow loads
    Msy, Vsy, delsy = moment_shear_defection(lt1, "y", qsy * 1e-5, L, Iy * 1e-8)
    Msz, Vsz, _     = moment_shear_defection(lt1, "y", qsz * 1e-5, L, Iy)
    return (Mgy, Vgy, delgy,
            Mw, Vw, delw,
            MQy, VQy, delQy,
            Mgz, Vgz, MQz, VQz,
            Mwp, Vwp, delwp,
            Msy, Vsy, delsy,
            Msz, Vsz)

def purlin22(Mgy, Vgy, delgy,
            Mw, Vw, delw,
            MQy, VQy, delQy,
            Mgz, Vgz, MQz, VQz,
            Mwp, Vwp, delwp,
            Msy, Vsy, delsy,
            Msz, Vsz,
            qgy, qgz, qw, Qy, Qz, qwp, qsy, qsz):
    # --- First set (ci = {'G','W-','Q','G','Q'}) ---
    c1 = ["y", "y", "y", "z", "z"]
    ci = ["G", "W-", "Q", "G", "Q"]
    loadi = np.array([qgy, qw, Qy, qgz, Qz])
    Mi    = np.array([Mgy, Mw, MQy, Mgz, MQz]) * 1e5
    Vi    = np.array([Vgy, Vw, VQy, Vgz, VQz]) * 1e5
    deli  = np.array([delgy, delw, delQy, 0, 0]) * 100
    # --- Second set (cj = {'G','W+','S','G','S'}) ---
    cj = ["G", "W+", "S", "G", "S"]
    loadj = np.array([qgy, qwp, qsy, qgz, qsz])
    Mj    = np.array([Mgy, Mwp, Msy, Mgz, Msz]) * 1e5
    Vj    = np.array([Vgy, Vwp, Vsy, Vgz, Vsz]) * 1e5
    delj  = np.array([delgy, delwp, delsy, 0, 0]) * 100
    # --- panne identifiers (t1, t2 in MATLAB) ---
    t1 = ["i"] * 5
    t2 = ["j"] * 5
    # --- Build combined arrays (stack vertically like MATLAB [a;b]) ---
    panne  = t1 + t2
    axis   = c1 + c1
    comb   = ci + cj
    load   = np.concatenate([loadi, loadj])
    M      = np.concatenate([Mi, Mj])
    V      = np.concatenate([Vi, Vj])
    delt   = np.concatenate([deli, delj])
    # --- Create DataFrame ---
    acp = pd.DataFrame({
        "panne": panne,"axis": axis,"comb": comb,"load": load,"M": M,"V": V,"del": delt})
    return acp

def purlin23(gy, w, Qy, wp, sy,Tcomb):
    # Base deflections
    del1 = np.array([gy, gy, gy, gy]) * 100
    del0 = np.zeros(4)
    del2 = np.array([w, Qy, sy, wp]) * 100
    del3 = np.array([0, 0, 0, sy]) * 100
    n = 8
    a = np.ones(n)  # vector of ones
    cas = ["ELS"] * n
    psi0 = Tcomb.loc[Tcomb["loads"] == "snow load", "psi0"].values[0]
    psi0_array=[psi0]*n    
    # Expand vectors 
    V1 = np.concatenate([del1, del0])
    V2 = np.concatenate([del2, del2])
    V3 = np.concatenate([del3, del3])
    # Calculate combination deflections
    delvals = np.zeros(n)
    for i in range(n):
        delvals[i] = comb(V1[i],V2[i],V3[i],cas[i],a[i],psi0_array[i])
                     #comb(G,Q1,Qi,cas,a,psi0): 
    # Combination labels
    c = [
        "G+W-", "G+Q", "G+S", "G+W++0.6*(S/2)",
        "W-", "Q", "S", "W++0.6*(S/2)"
    ]
    # Build DataFrame
    combdel = pd.DataFrame({"comb": c,"Gk": V1,"Qk1": V2,"Qki": V3,"Sser": delvals})
    # Extract max values
    delmax = np.max(delvals[0:4])
    d2 = np.max(delvals[4:8])
    return combdel, delmax, d2

def purlin24(gy,gz,w,Qy,sy,wp,Qz,sz,Tcomb):
    # Define load effect vectors (scaled like MATLAB *1e5)
    V1 = np.array([gy, gy, gy, gy, gz, gz]) * 1e5
    V2 = np.array([-w, Qy, sy, wp, Qz, sz]) * 1e5
    V3 = np.array([0, 0, 0, sy, 0, 0]) * 1e5
    n = len(V1)
    combVvals = np.zeros(n)
    a = np.ones(n)        # like MATLAB zeros then set to 1
    cas = ["ELU"] * n
    psi0 = Tcomb.loc[Tcomb["loads"] == "snow load", "psi0"].values[0]
    psi0_array=[psi0]*n    
    # Apply combination function
    for i in range(n):
        combVvals[i] = comb(V1[i],V2[i],V3[i],cas[i],a[i],psi0_array[i])
    # Axis labels
    c = ["y", "y", "y", "y", "z", "z"]
    # Combination labels
    c2 = [
        "G+1.5W-",
        "1.35G+1.5Q",
        "1.35G+1.5S",
        "1.35G+1.5W++1.5*0.6*(S/2)",
        "1.35G+1.5Q",
        "1.35G+1.5S"
    ]
    # Build DataFrame
    com = pd.DataFrame({"axis": c,"comb": c2,"Gk": V1,"Qk1": V2,"Qki": V3,"Sd": combVvals})
    # Extract maxima
    ysdn = np.max(combVvals[0:4])   # y shear (design)
    zsdp = np.max(combVvals[4:6])   # z shear
    ysdp = np.max(combVvals[1:3])   # y moment
    zscorr = gz * 1e5               # corrected z shear
    return com, ysdn, zscorr, ysdp, zsdp

def purlin_tie_rod(t,qgp,qgc,s,alpha,L,Tcomb):
    np_val = 7                 # number of purlins
    d = 1.41                   # spacing (m)
    B = (L / 2) / np.cos(np.radians(alpha)) - 0.6 - d / 2
    # Loads
    G = (np_val - 1) * qgp + qgc * B
    Q1 = s * B
    # Combination (ELU, snow case)
          #comb(G,Q1,Qi,cas,a,psi0):
    psi0 = Tcomb.loc[Tcomb["loads"] == "snow load", "psi0"].values[0]
    qsdt = comb(G,Q1,0,"ELU",1,psi0)
    # Tie force
    NT = 1.25 * qsdt * np.sin(np.radians(alpha)) * t / 2
    beta = np.degrees(np.arctan(d / (t / 2)))
    Nsd = NT / (2 * np.sin(np.radians(beta)))
    # Build result table
    T = pd.DataFrame({
        "np": [np_val],"B": [B],"qsdt": [qsdt],"NT": [NT],"beta": [beta],"Nsd": [Nsd]})
    return T
