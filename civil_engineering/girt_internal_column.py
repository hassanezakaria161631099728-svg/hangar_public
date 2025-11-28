import pandas as pd
import numpy as np
from shared_functions.utils import (beam3,moment_shear_defection,find_vplrd,resistance,
lateral_torsional_buckling,find_epf,buckling1,buckling2,buckling3)
from shared_functions.wind.cpe import wall,cpe_from_s
from shared_functions.wind.action_of_set import fwe
def girt(hangarf, chI, beamT):
    # call to girt0
    Lx, Ly, ba, Tin1, Tin2, Tin3, Tin6, Tin7, Tin8 = girt0(hangarf, chI)    
    e1 = Tin1["e_m"].iloc[0]   
    e2 = Tin6["e_m"].iloc[0]
    b1 = Lx
    b2 = Ly
    lf = Lx / 4
    t = Ly / 4
    floor_height=ba["floor_height_m"].iloc[0]
    n_floors=ba["n_floors"].iloc[0]
    h2 = floor_height*n_floors    
    qgb = 11.89  # daN/m    
    # call to girt1
    qws1, qwf1 = girt1(Lx, Ly, e1, b1, Tin2, Tin3)
    qws2, qwf2 = girt1(Lx, Ly, e2, b2, Tin7, Tin8)    
    # determining the fit UPE for girts
    qws = max(qws1, -qws2) / 10
    qwf = max(-qwf1, qwf2) / 10    
    TUPE = pd.read_excel(beamT, sheet_name='UPE')    
    sls,_,_ = beam3(t, qws, TUPE) # sgs selelected girt side
    slf,_,_ = beam3(lf, qwf, TUPE) # sgf selelected girt front
    if sls["h"].iloc[0] < slf["h"].iloc[0]:
        Tgirt = slf
        ll = lf
        qw = qwf
        print("girt of front has more load")
    elif slf[0][1] < sls[0][1]:
        Tgirt = sls
        ll = t
        qw = qws
        print("girt of side has more load")
    else:
        Tgirt = slf
        ll = lf
        qw = qwf    
    qgl = Tgirt["P"].iloc[0]   
    qg = qgl + qgb       # daN/ml    
    # T2 table
    T2 = pd.DataFrame({
        "qws1": [qws1], "qwf1": [qwf1], "qws2": [qws2], "qwf2": [qwf2],
        "qws": [qws], "qwf": [qwf], "ll": [ll], "qw": [qw],
        "qgl": [qgl], "qgb": [qgb], "qg": [qg]})    
    # actions on the girt
    lt = 'uniform'
    Mw, Vw,_ = moment_shear_defection(lt, 'y', qw, ll, 1)
    Mg, Vg,_ = moment_shear_defection(lt, 'z', qg, ll, 1)    
    # verifications a ELU
    Vysd = 1.5 * Vw
    Vzsd = 1.35 * Vg
    Avy, Vplrdy = find_vplrd('y', Tgirt, 'UPE', Vysd)
    Avz, Vplrdz = find_vplrd('z', Tgirt, 'UPE', Vzsd)    
    Mysd = 1.5 * Mw
    Mzsd = 1.35 * Mg
    r = resistance(Tgirt, Mysd, Mzsd)    
    T3 = pd.DataFrame({
        "Vysd": [Vysd], "Vzsd": [Vzsd],
        "Avy": [Avy], "Vplrdy": [Vplrdy],
        "Avz": [Avz], "Vplrdz": [Vplrdz],
        "Mysd": [Mysd], "Mzsd": [Mzsd], "r": [r]
    })
    # girt tie rods                 
    G1, qsdt1, L1, NT1, Nsd1, beta = girt_tie_rod(h2, qgl, qgb, lf, t, 'side')
    G2, qsdt2, L2, NT2, Nsd2,_ = girt_tie_rod(h2, qgl, qgb, lf, t, 'front')
    Nsd = max(Nsd1,Nsd2)    
    T4 = pd.DataFrame({
        "G1": [G1], "qsdt1": [qsdt1], "L1": [L1], "NT1": [NT1], "Nsd1": [Nsd1], "beta": [beta],
        "G2": [G2], "qsdt2": [qsdt2], "L2": [L2], "NT2": [NT2], "Nsd2": [Nsd2], "Nsd": [Nsd]
    })    
    return Tgirt, T2, T3, T4

def girt0(hangarf, chI):
    # Read Excel sheets
    ba = pd.read_excel(hangarf, sheet_name='building attributes')
    in1 = pd.read_excel(chI, sheet_name='T1')
    in2 = pd.read_excel(chI, sheet_name='T2')
    in3 = pd.read_excel(chI, sheet_name='T3')
    in6 = pd.read_excel(chI, sheet_name='T6')
    in7 = pd.read_excel(chI, sheet_name='T7')
    in8 = pd.read_excel(chI, sheet_name='T8')
    Lx = ba["Lx_m"].iloc[0]
    Ly = ba["Ly_m"].iloc[0]
    return Lx, Ly, ba, in1, in2, in3, in6, in7, in8

def girt1(Lx, Ly, e, b, T2, T3):
    """
    qws: side wall
    qwf: front (gable)
    """
    lf = Lx / 4
    t = Ly / 4
    bl = 1.0  # m
    La = e / 5.0
    if b == Ly:
        L = lf
    elif b == Lx:
        L = t
    else:
        raise ValueError("b must be either Ly or Lx")
    Lb = L - La
    sd = L * bl
    sa = La * bl
    sb = Lb * bl
    # External call
    Twall = wall()
    sw = [sd, 0, 0, sa, sb]
    n = len(sw) + 1
    cd = np.array([1, 0, 0, 1, 1])
    blm = np.array([1, 0, 0, 1, 1])
    qw = np.zeros(n - 1)
    epf = cpe_from_s(n, Twall, sw)
    qpze1=T2["qpze_N_m_2"].iloc[0]
    qpze2=0
    we = fwe(epf, sw, qpze1, qpze2)
    # MATLAB table3{1:n-1,5} → pandas iloc[0:n-1, 4]
    wi = T3.iloc[0:n-1, 4].to_numpy()
    for i in range(n - 1):
        qw[i] = (cd[i] * we[i] - wi[i]) * blm[i]  # daN/ml
    qwa = qw[3]
    qwb = qw[4]
    qwd = qw[0]
    qeq = (qwa * La + qwb * Lb) / L
    if L == t:  # qws side, qwf gable
        qws, qwf = qwd, qeq
    elif L == lf:
        qws, qwf = qeq, qwd
    else:
        raise ValueError("Unexpected case for L")
    return qws, qwf
   
def girt_tie_rod(hp, qgl, qgb, lf, t, cas):
    nl = hp - 1
    if cas == "side":
        n = nl - 1
        k = -1
        L = t
    elif cas == "front":
        n = nl
        k = 1
        L = lf
    else:
        raise ValueError("cas must be 'side' or 'front'")    
    G = n * qgl + qgb * (hp - 2 + k * 0.5)
    qsdt = 1.35 * G
    NT = 1.25 * qsdt * L / 2.0
    if cas == "side":
        beta = np.degrees(np.arctan(1 / (t / 2.0)))  # atand
        Nsd = NT / (2 * np.sin(np.radians(beta)))    # sind
    elif cas == "front":
        beta=0
        Nsd = NT    
    else: raise ValueError("cas has to be either side or front")
    return G, qsdt, L, NT, Nsd, beta

def internal_column(Tgirt, hangarf, chI, beamT):
    # call girt0
    Lx, Ly, ba, Tin1, Tin2, Tin3, Tin6, Tin7, Tin8 = girt0(hangarf, chI)
    # extract values
    e1 = Tin1["e_m"].iloc[0]
    e2 = Tin6["e_m"].iloc[0]
    floor_height=ba["floor_height_m"].iloc[0] 
    n_floors= ba["n_floors"].iloc[0]
    hp= ba["hp_m"].iloc[0]
    h1 = floor_height*n_floors+hp
    h2 = floor_height*n_floors
    b1 = Ly
    b2 = Lx
    qpze1 = Tin2["qpze_N_m_2"].iloc[0]
    qpze2 = Tin7["qpze_N_m_2"].iloc[0]
    wi1 = Tin3["wi1"].iloc[0]
    wi2 = Tin8["wi1"].iloc[0]
    qgb = 11.89  # daN
    # inter_columns
    #qw1, s1, s2, bpa, ha
    qwn, sa, sb, bpa, ha = inter_column1(b1, Lx, Ly, e1, qpze1, wi1, h2)
    qwp, sd, _ , _ , _  = inter_column1(b2, Lx, Ly, e2, qpze2, wi2, h2)
    qw = max(-qwn, qwp)  # T6, T7, T8
    # read HEB
    THEB = pd.read_excel(beamT, sheet_name="HEB")
    Tinter_column, Iymin, lb = beam3(h1, qw, THEB)
    # T5 table
    T5 = pd.DataFrame([{
        "qwn": qwn, "bpa": bpa, "ha": ha, "sa": sa, "sb": sb,
        "qwp": qwp, "sd": sd, "qw": qw, "Iymin": Iymin, "lb": lb
    }])
    # shear & normal forces
    Vw = qw * h1 / 2
    Vysd = 1.5 * Vw
    Avy, Vplrdy = find_vplrd("y", Tinter_column, "HEB", Vysd)
    nl = h2
    bp = Lx / 4
    pl = Tgirt["P"].iloc[0]   # poids linéaire lisse
    NGl = nl * bp * pl
    NGb = (sd - bp * 2) * qgb
    plp = Tinter_column.iloc[0, 7]   # poids linéaire potelet
    NGp = plp * h1
    NG = NGl + NGb + NGp
    Nsd = 1.35 * NG
    Mwn = -qwn * (h1 - 1) ** 2 / 8
    Mwp = qw * h1 ** 2 / 8
    Mysdn = 1.5 * Mwn
    Mysdp = 1.5 * Mwp
    T6 = pd.DataFrame([{
        "Vw": Vw, "Vysd": Vysd, "Avy": Avy, "Vplrdy": Vplrdy,
        "NGl": NGl, "NGb": NGb, "NGp": NGp, "NG": NG, "Nsd": Nsd,
        "Mysdn": Mysdn, "Mysdp": Mysdp
    }])
    fy = 2350
    # buckling
    lay, layb, alphay, phiy, xiy,_,_ = buckling1(h1, "y", Tinter_column, "simplement appuye", 1, 1)
    laz, lazb, alphaz, phiz, xiz,_,_ = buckling1(h1, "z", Tinter_column, "simplement appuye", 1, 1)
    uy, ky = buckling2(layb, Tinter_column, Nsd, xiy, fy, "y")
    xmin, rf = buckling3(Nsd, fy, xiy, xiz, ky, Mysdp, Tinter_column)
    T7 = pd.DataFrame([{
        "h1": h1, "lay": lay, "layb": layb, "alphay": alphay, "phiy": phiy, "xiy": xiy,
        "laz": laz, "lazb": lazb, "alphaz": alphaz, "phiz": phiz, "xiz": xiz,
        "Nsd": Nsd, "uy": uy, "ky": ky, "xmin": xmin, "rf": rf
    }])
    # lateral_torsional_buckling
    lazd, lazbd, alphazd, phizd, xizd,_,_ = buckling1(h1 - 1, "z", Tinter_column,
                                               "simplement appuye", 1, 1)
    ult, klt, L1, Lfz, zg, k, c1, c2, lalt, laltb, alphalt, philt, xlt, rd = \
        lateral_torsional_buckling(h1 - 1, lazbd, "simplement appuye", 
                                   Nsd, 1, fy, Tinter_column, 0, 1, Mysdn)
    T8 = pd.DataFrame([{
        "h": h1 - 1, "lazd": lazd, "lazbd": lazbd,
        "alphazd": alphazd, "phizd": phizd, "xizd": xizd
    }])
    T9 = pd.DataFrame([{
        "Nsd": Nsd, "Mysdn": Mysdn, "xizd": xizd,
        "ult": ult, "klt": klt, "L1": L1, "Lfz": Lfz, "zg": zg,
        "k": k, "c1": c1, "c2": c2,
        "lalt": lalt, "laltb": laltb, "alphalt": alphalt, "philt": philt,
        "xlt": xlt, "rd": rd
    }])
    return T5, Tinter_column, T6, T7, T8, T9

def inter_column1(b, Lx, Ly, e, qpze, wi, hp):
    """
    Compute loads for internal column (potelet).    
    Parameters
    ----------
    b : float
        Reference dimension (Lx or Ly depending on case).
    Lx, Ly : float
        Building dimensions.
    e : float
        Some distance parameter.
    qpze : float
        External pressure.
    wi : float
        Internal pressure (from table).
    hp : float
        Column height.
    Returns
    -------
    qw1 : float
        Load per unit length (daN/ml).
    s1, s2 : float
        Sectional areas / surfaces (depending on case).
    bpa : float
        Effective base width.
    ha : float
        Column height adjustment.
    """
    epfD = 0.8
    weD = epfD * qpze
    bp = Lx / 4.0
    k = Lx / 2.0
    h1 = (2 / k) * 0.5 * bp + hp
    h2 = (2 / k) * 1.5 * bp + hp
    s1 = s2 = bpa = ha = 0.0  # initialize
    qw = 0.0
    if b == Ly:
        ba = e / 5.0
        if ba < 1.5 * bp:
            bpa = ba - 0.5 * bp
            bpb = bp - bpa
            ha = (2 / k) * ba + hp
            sa = (h1 + ha) * (bpa / 2.0)
            sb = (ha + h2) * (bpb / 2.0)
        else:
            bpa = bp
            bpb = 0
            ha=0 
            sa = (h1 + h2) * (bp / 2.0)
            sb = 0
        epfA = find_epf(-1.3, -1, sa)
        epfB = find_epf(-1, -0.8, sb)
        weA = epfA * qpze
        weB = epfB * qpze
        qwA = (weA - wi) * bpa
        qwB = (weB - wi) * bpb
        qw = qwA + qwB
        print("inter_column 1 has more load for wind1 qw<0")
        s1, s2 = sa, sb
    elif b == Lx:
        sd = (h2 + 0.25) * bp
        qw = (weD - wi) * bp
        s1,s2 = sd, 0
        print("inter_column 2 has more load for wind2 qw<0")
    else: raise ValueError("b has to be either Ly or Lx")
    qw1 = qw / 10.0  # daN/ml
    return qw1, s1, s2, bpa, ha

