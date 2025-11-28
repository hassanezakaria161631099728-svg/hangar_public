import numpy as np
import pandas as pd
from typing import Tuple, Any
def fwe(pf, sw, qpze1, qpze2):
    sd2 = sw[1]
    pf = np.asarray(pf)
    w = np.zeros_like(pf, dtype=float)

    if sd2 == 0:
        w = pf * qpze1
    else:
        w[0] = pf[0] * qpze1
        w[1:] = pf[1:] * qpze2
    return w


def xyz_wall(b:float,d:float,sd2:float,Lx:float,Ly:float,h:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # initial definitions
    par = [b / 2]
    per = [0, d]
    z_vals = [h / 2, b / 2, (h - b) / 2 + b / 2]
    if sd2 == 0:
        par = [par[0], par[0]]
        per = [per[0], per[1]]
        z = [z_vals[0], z_vals[0]]
    elif sd2 > 0:
        par = [par[0], par[0], par[0]]
        per = [per[0], per[0], per[1]]
        z = [z_vals[1], z_vals[2], z_vals[0]]
    else:
        raise ValueError("sd2 must be >= 0")
    # orientation depending on wind direction
    if b == Lx:
        x, y = per, par
    elif b == Ly:
        y, x = per, par
    else:
        raise ValueError("b must equal either Lx or Ly")
    return np.array(x), np.array(y), np.array(z)

def xyz_roof(e: float, b: float, d: float, bt: str, bt2: str, Lx: float, Ly: float, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # perpendicular direction
    per = [e / 8, b / 2, b - e / 8]  # f1, g, f2
    # parallel direction
    par = [e / 20, (e / 2 - e / 10) / 2 + e / 10, (d - e / 2) / 2 + e / 2]  # f1, g, f2/h/I
    if bt == "flat roof":
        per = [per[0], per[1], per[2], per[1], per[1], per[1]]
        par = [par[0], par[0], par[0], par[1], par[1], par[1]]
        if d < e / 2:
            per = [per[0], per[1], per[2], per[1]]  # f1 g f2 h
            par[1] = d - e / 10
            par = [par[0], par[0], par[0], par[1]]
    elif bt == "hangar":
        if bt2 == "shed":
            if b == Ly:
                per = [per[0], per[1], per[2], per[1], per[0], per[1], per[2], per[1]]
                par = [par[0], par[0], par[0], par[1], par[0], par[0], par[0], par[1]]
            elif b == Lx:
                per = [per[0], per[1], per[2], per[1], per[1]]
                par = [par[0], par[0], par[0], par[1], par[2]]
        elif bt2 == "gable":
            if b == Lx:
                per[1] = (b / 2 - e / 4) / 2
                per[2] = (b / 2 - e / 4) / 2 + b / 2
                per.extend([b - e / 8, b / 4, b / 2 + b / 4, b / 4, b / 2 + b / 4])
                per = [per[0], per[1], per[2], per[3], per[4], per[5], per[6], per[7]]
                par = [par[0], par[0], par[0], par[0], par[1], par[1], par[2], par[2]]
            elif b == Ly:
                par[1] = (d / 2 - e / 10) / 2
                par[2] = e / 20 + d / 2
                par.extend([(d / 2 - e / 10) / 2 + d / 2])
                per = [per[0], per[1], per[2], per[1], per[1], per[1],
                       per[0], per[1], per[2], per[1], per[1], per[1]]
                par = [par[0], par[0], par[0], par[1], par[2], par[3],
                       par[0], par[0], par[0], par[1], par[2], par[3]]
    else:
        raise ValueError(f"Unknown roof type: {bt}")
    # orientation
    if b == Lx:
        x, y = per, par
    elif b == Ly:
        y, x = per, par
    else:
        raise ValueError("b must equal either Lx or Ly")
    n = len(x)
    z = [h] * n
    return np.array(x), np.array(y), np.array(z)

def moment_renversant(Rhor,ZRhor,Rz,d,horRz):
# moment renversant
 Mrv=1.5*(Rhor+ZRhor+Rz*(d-horRz))
 return Mrv

def reactions(k, bt, bt2, ba, ed,Fwehor, Fwi1hor, Fwi2hor,Fwez, Fwi1z, Fwi2z,
          b, srt, x, y, z, Lx, Ly, d):
    # Initialize arrays
    Rhor1i = np.zeros(k)
    Rhor2i = np.zeros(k)
    Rz1i = np.zeros(k)
    Rz2i = np.zeros(k)
    Rhorzi1 = np.zeros(k)
    Rhorzi2 = np.zeros(k)
    Rzhori1 = np.zeros(k)
    Rzhori2 = np.zeros(k)
    # First loop
    for i in range(ed - 1):
        Rhor1i[i] = Fwehor[i] + Fwi1hor[i]
        Rhor2i[i] = Fwehor[i] + Fwi2hor[i]
        Rhorzi1[i] = Rhor1i[i] * z[i]
        Rhorzi2[i] = Rhor2i[i] * z[i]
    # Horizontal axis choice
    if b == Lx:
        hor = y
    elif b == Ly:
        hor = x
    else:
        hor = np.zeros_like(x)  # fallback
    # Vertical forces
    for i in range(ed - 1, k):
        Rz1i[i] = Fwez[i] + Fwi1z[i]
        Rz2i[i] = Fwez[i] + Fwi2z[i]
    # Moments wrt hor
    for i in range(ed - 1, k):
        Rzhori1[i] = Rz1i[i] * hor[i]
        Rzhori2[i] = Rz2i[i] * hor[i]
    # Sums
    Rhor1 = np.sum(Rhor1i[:ed - 1])
    ZRhor1 = np.sum(Rhorzi1[:ed - 1]) / Rhor1 if Rhor1 != 0 else 0
    Rhor2 = np.sum(Rhor2i[:ed - 1])
    ZRhor2 = np.sum(Rhorzi2[:ed - 1]) / Rhor2 if Rhor2 != 0 else 0
    Rhor = [Rhor1, Rhor2]
    ZRhor = [ZRhor1, ZRhor2]
    Rz1 = np.sum(Rz1i[ed - 1:])
    horRz1 = np.sum(Rzhori1[ed - 1:]) / Rz1 if Rz1 != 0 else 0
    Rz2 = np.sum(Rz2i[ed - 1:])
    horRz2 = np.sum(Rzhori2[ed - 1:]) / Rz2 if Rz2 != 0 else 0
    Rz = [Rz1, Rz2]
    horRz = [horRz1, horRz2]
    row = ["wi1-", "wi2+"]
    # --- Flat roof case ---
    if bt == "flat roof":
        sI = srt[3]  # MATLAB srt(4)
        if sI > 0:
            Rz1 = np.sum(Rz1i[ed - 1:k - 1])
            horRz1 = np.sum(Rzhori1[ed - 1:k - 1]) / Rz1 if Rz1 != 0 else 0
            Rz2 = np.sum(Rz2i[ed - 1:k - 1])
            horRz2 = np.sum(Rzhori2[ed - 1:k - 1]) / Rz2 if Rz2 != 0 else 0
            Rver3 = np.sum(Rz1i[ed - 1:k - 2]) + Rz1i[k - 1]
            horRver3 = (np.sum(Rzhori1[ed - 1:k - 2]) + Rzhori1[k - 1]) / Rz1 if Rz1 != 0 else 0
            Rver4 = np.sum(Rz2i[ed - 1:k - 2]) + Rz2i[k - 1]
            horRver4 = (np.sum(Rzhori2[ed - 1:k - 2]) + Rzhori2[k - 1]) / Rz2 if Rz2 != 0 else 0
            Rz = [Rz1, Rz2, Rver3, Rver4]
            horRz = [horRz1, horRz2, horRver3, horRver4]
            Rhor = [Rhor1, Rhor2, Rhor1, Rhor2]
            ZRhor = [ZRhor1, ZRhor2, ZRhor1, ZRhor2]
            row = ["wi- I-", "wi+ I-", "wi- I+", "wi+ I+"]
    # --- Hangar case ---
    if bt == "hangar":
        Rhor = [Rhor1]
        Rz = [Rz1]
        T = ba[0][3]  # ba{1,4}
        horRz = [horRz1]
        ZRhor = [ZRhor1]
        row = ["row"]
        if bt2 == "gable" and b == T:
            Rz1 = np.sum(Rz1i[ed - 1:k - 6])
            horRz1 = np.sum(Rzhori1[ed - 1:k - 6]) / Rz1 if Rz1 != 0 else 0
            Rz2 = np.sum(Rz1i[ed - 1:k - 7]) + Rz1i[k - 1]
            horRz2 = (np.sum(Rzhori1[ed - 1:k - 7]) + Rzhori1[k - 1]) / Rz1 if Rz1 != 0 else 0
            Rver3 = np.sum(Rz1i[ed + 3:k - 1])
            horRver3 = np.sum(Rzhori1[ed + 3:k - 1]) / Rz1 if Rz1 != 0 else 0
            Rver4 = np.sum(Rz1i[k - 6:k])
            horRver4 = np.sum(Rzhori1[k - 6:k]) / Rz1 if Rz1 != 0 else 0
            Rz = [Rz1, Rz2, Rver3, Rver4]
            horRz = [horRz1, horRz2, horRver3, horRver4]
            Rhor = [Rhor1] * 4
            ZRhor = [ZRhor1] * 4
            row = ["--", "-+", "+-", "++"]
    # --- Final moment calc ---
    o = len(Rhor)
    Mrv = np.zeros(o)
    d1 = np.full(o, d)
    for i in range(o):
        Mrv[i] =moment_renversant(Rhor[i], ZRhor[i], Rz[i], d1[i], horRz[i])
    # Build DataFrame
    T5 = pd.DataFrame({"row": row,"Rhor": Rhor,"ZRhor": ZRhor,"Rz": Rz,"d": d1,"horRz": horRz,"Mrv": Mrv})
    return T5

def xyz(e, b, d, bt, bt2, Lx, Ly, sd2, ba) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # --- Determine building height h ---
    if bt == "flat roof":
        # MATLAB: fh = ba{1,1}; nf = ba{1,2}; h = fh*nf
        try:
            # Try pandas DataFrame
            import pandas as pd
            if isinstance(ba, pd.DataFrame):
                fh = ba.iloc[0, 0]
                nf = ba.iloc[0, 1]
            else:
                fh = ba[0][0]
                nf = ba[0][1]
        except Exception:
            arr = np.asarray(ba)
            fh = arr[0, 0]
            nf = arr[0, 1]
        h = fh * nf
    elif bt == "hangar":
        try:
            import pandas as pd
            if isinstance(ba, pd.DataFrame):
                h = ba.iloc[0, 0]
            else:
                h = ba[0][0]
        except Exception:
            arr = np.asarray(ba)
            h = arr[0, 0]
    else:
        raise ValueError(f"Unknown roof type: {bt}")
    # --- Call xyz1 for wall coordinates ---
    xm,yt,zt =xyz_wall(b, d, sd2, Lx, Ly, h)
    # --- Call xyz2 for roof coordinates ---
    xt,yt,zt =xyz_roof(e,b,d,bt,bt2,Lx,Ly,h)
    # --- Concatenate results ---
    x = np.concatenate([np.atleast_1d(xm), np.atleast_1d(xt)])
    y = np.concatenate([np.atleast_1d(yt), np.atleast_1d(yt)])
    z = np.concatenate([np.atleast_1d(zt), np.atleast_1d(zt)])
    return x, y, z

def actions_ensemble(crt2, sw, we, wi1, wi2, cd, s, e, b, d, ba, bt, bt2, Lx, Ly, srt):
    n = len(we)
    Fwe = np.zeros(n)
    Fwi1 = np.zeros(n)
    Fwi2 = np.zeros(n)
    sd2 = sw[1]   # sw(2) in MATLAB (1-based)
    sc = sw[5]    # sw(6) in MATLAB
    # --- Decide ed, in, cw ---
    if sd2 == 0 and sc == 0:
        ed, in_ = 3, 5
        cw = ["D", "E"]
    elif sd2 == 0 and sc > 0:
        ed, in_ = 3, 6
        cw = ["D", "E"]
    elif sd2 > 0 and sc == 0:
        ed, in_ = 4, 6
        cw = ["D1", "D2", "E"]
    elif sd2 > 0 and sc > 0:
        ed, in_ = 4, 7
        cw = ["D1", "D2", "E"]
    # --- Fill arrays from in_ to n ---
    for i in range(in_ - 1, n):  # MATLAB in:n -> Python in_-1:n-1
        Fwe[i] = -1 * cd * we[i] * s[i]
        Fwi1[i] = cd * wi1[i] * s[i]
        Fwi2[i] = cd * wi2[i] * s[i]
    # --- Special cases depending on sd2 ---
    if sd2 == 0:
        Fwe[0] = cd * we[0] * s[0]
        Fwi1[0] = -1 * cd * wi1[0] * s[0]
        Fwi2[0] = -1 * cd * wi2[0] * s[0]
        Fwe[1] = -1 * cd * we[1] * s[1]
        Fwi1[1] = cd * wi1[1] * s[1]
        Fwi2[1] = cd * wi2[1] * s[1]
    elif sd2 > 0:
        Fwe[0] = cd * we[0] * s[0]
        Fwi1[0] = -1 * cd * wi1[0] * s[0]
        Fwi2[0] = -1 * cd * wi2[0] * s[0]
        Fwe[1] = cd * we[1] * s[1]
        Fwi1[1] = -1 * cd * wi1[1] * s[1]
        Fwi2[1] = -1 * cd * wi2[1] * s[1]
        Fwe[2] = -1 * cd * we[2] * s[2]
        Fwi1[2] = cd * wi1[2] * s[2]
        Fwi2[2] = cd * wi2[2] * s[2]
    # --- Call xyz (to be implemented separately) ---
    x,y,z = xyz(e, b, d, bt, bt2, Lx, Ly, sd2, ba)  # placeholder
    k = len(x)
    Fwez = np.zeros(k)
    Fwi1z = np.zeros(k)
    Fwi2z = np.zeros(k)
    Fwez[ed - 1:k] = Fwe[in_ - 1:n]
    Fwi1z[ed - 1:k] = Fwi1[in_ - 1:n]
    Fwi2z[ed - 1:k] = Fwi2[in_ - 1:n]
    Fwehor = np.zeros(k)
    Fwi1hor = np.zeros(k)
    Fwi2hor = np.zeros(k)
    for i in range(ed - 1):
        Fwehor[i] = Fwe[i]
        Fwi1hor[i] = Fwi1[i]
        Fwi2hor[i] = Fwi2[i]
    # --- Combine c labels ---
    c = cw + list(crt2)
    # --- Build T4 ---
    if bt == "flat roof":
        T4 = pd.DataFrame({"c": c,"Fwehor": Fwehor,"Fwi1hor": Fwi1hor,
            "Fwi2hor": Fwi2hor,"Fwez": Fwez,"Fwi1z": Fwi1z,"Fwi2z": Fwi2z,"x": x,"y": y,"z": z})
    else:
        T4 = pd.DataFrame({"c": c,
            "Fwehor": Fwehor,"Fwi1hor": Fwi1hor,"Fwez": Fwez,"Fwi1z": Fwi1z,"x": x,"y": y,"z": z
        })
    # --- Call react (to be implemented separately) ---
    T5 = reactions(k, bt, bt2, ba, ed,
        Fwehor, Fwi1hor, Fwi2hor,Fwez, Fwi1z, Fwi2z,b, srt, x, y, z, Lx, Ly, d)
    return T4, T5
