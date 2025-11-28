import numpy as np

def wall_perpendicular(ba,b):
    f_height = ba["floor_height_m"].iloc[0]
    hp=ba["hp_m"].iloc[0]
    n_floors = ba["n_floors"].iloc[0]
    h1 = f_height*n_floors+hp
    h2 = f_height*n_floors
    Lx=ba["Lx_m"].iloc[0]# symbole L if hangar
    Ly=ba["Ly_m"].iloc[0]# symbole T if hangar
    if h1 <= b:
        if b == Lx:  # direction y (vent 2)
            sd1 = b * h1
            sd2 = 0
        elif b == Ly:  # direction x (vent 1)
            sd1 = b * h2
            sd2 = 0
        else:
            sd1, sd2 = 0, 0  # safeguard if dimensions mismatch
        se = sd1 + sd2   # ✅ ensure se is always defined here
    elif b < h1:
     sd1 = b * b
     sd2 = b * (h1 - b)
     se = sd1 + sd2
    return h1,h2,sd1,sd2,se

def wall_parallel(h1,h2,b,d,bt,e):
    # paroi verticale
    cd = 1 if h1 < 4 * b else 0
    if bt == 'flat roof':
        h = h1
    elif bt == 'hangar':
        h = h2
    else: h=0
    if e < d:
        sa = (e / 5) * h
        sb = (4 * e / 5) * h
        sc = (d - e) * h
    else:
        sa = (e / 5) * h
        sb = (d - e / 5) * h
        sc = 0
    return cd, sa, sb, sc

def roof_list(b,d,ba,bt,e):
    sf = (e / 4) * (e / 10)  # m^2
    sg = (b - e / 2) * e / 10  # m^2
    if bt == 'flat roof':
        if d < e / 2:
         sh = b * (d - e / 10)  # m^2
         sI = 0
        else:  
         sh = b * (e / 2 - e / 10)  # m^2
         sJ = 0
         sI = b * (d - e / 2)  # m^2
    elif bt == 'gable':
        L = ba[0][2] if isinstance(ba, (list, tuple)) else ba.iloc[0, 2]
        if b == L:  # angle 90 of the wind
            sg = (b - e / 2) / 2 * (e / 10)
            sh = (b / 2) * (e / 2 - e / 10)
            sJ = 0
            sI = b * (d - e / 2)
            if d < e / 2:
                sh = (b / 2) * (d - e / 10)
                sI = 0
        else:
         sh = b * (d / 2 - e / 10)  # m^2
         sJ = 2 * sf + sg
         sI = sh    
    elif bt == 'shed':
        L = ba[0][2] if isinstance(ba, (list, tuple)) else ba.iloc[0, 2]
        if L == b:
            if d < e / 2:
             sh = b * (d - e / 10)
             sI = 0
             sJ=0
            else:
             sh = b * (e / 2)
             sJ = 0
             sI = b * (d - e / 2)
        else:
         sh = b * (d - e / 10)
         sJ = 0
         sI = 0
    else: sf,sg,sh,sJ,sI=0,0,0,0,0
    return sf,sg,sh,sJ,sI

def roof_array(sf,sg,sh,sJ,sI,bt,bt2,b,ba,h1,h2):
    if bt == 'flat roof':
        st = np.array([sf, sg, sh, sI, sI])
        ct = ['F-', 'G-', 'H-', 'I-', 'I+']
    elif bt == 'hangar':
        # Extract values from ba (assume list of lists like [[h1,h2,L,T]])
        L= ba["Lx_m"].iloc[0]
        T=ba["Ly_m"].iloc[0]
        alpha = np.degrees(np.arctan((h1 - h2) / (L / 2)))
        if bt2 == 'shed':
            if b == T:
                s1 = np.array([sf, sg, sf, sh])
                ct = ['F', 'G', 'H']
            elif b == L:
                s1 = np.array([sf, sg, sf, sh, sI])
                ct = ['Fup', 'G', 'Flow', 'H', 'I']
            else:
                raise ValueError("Unexpected b vs T/L relation for shed")
        elif bt2 == 'gable':
            if b == T:
                s1 = np.array([sf, sg, sh, sJ, sI])
                ct = ['F', 'G', 'H', 'J', 'I']
            elif b == L:
                s1 = np.array([sf, sg, sh, sI])
                ct = ['F', 'G', 'H', 'I']
            else:
                raise ValueError("Unexpected b vs T/L relation for gable")
        else:
            raise ValueError(f"Unknown subtype bt2: {bt2}")
        # Apply geometric scaling factor
        st = s1 * (1 / np.cos(np.radians(alpha)))
    else:
        raise ValueError(f"Unknown building type: {bt}")
    return st,ct
