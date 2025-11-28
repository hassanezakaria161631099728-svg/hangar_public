import numpy as np
import pandas as pd

def openings(cas: str) -> pd.DataFrame:
    n = [2, 4]
    if cas == "front":
        w = [3, 2]
        h = [4, 0.6]
    elif cas == "side":
        w = [2, 3]
        h = [3, 0.6]
    else:
        raise ValueError("cas must be 'front' or 'side'")
    s = [n[i] * w[i] * h[i] for i in range(2)]
    r = ["doors", "windows"]
    columns = ["rows", "number", "width(m)", "height(m)", "surface(m^2)"]
    T = pd.DataFrame({"rows": r,"number": n,"width(m)": w,"height(m)": h,"surface(m^2)": s})
    return T

def cpi(bt,bt2,ba,b,d,n,h):
    cpi1 = np.zeros(n)
    cpi2 = np.zeros(n)
    if bt == "flat roof":
        print("on a flat roof tall building eurocode ipf1=-0.3 ipf2=+0.2")
        cpi1[:] = -0.3
        cpi2[:] = 0.2
    elif bt == "hangar":
        L = ba["Lx_m"].iloc[0]
        T = ba["Ly_m"].iloc[0]
        if bt2 == "gable":
            # openings
            cas1,cas2 = "front","side"
            T1 = openings(cas1)
            T2 = openings(cas2)
            if b == T:
              p = T2["surface(m^2)"].sum()
              k = T1["surface(m^2)"].sum()
            elif b == L:
              p = T1["surface(m^2)"].sum()
              k = T2["surface(m^2)"].sum()
            else: raise ValueError("b has to be either Lx or Ly")
            up = (p + 2 * k) / (2 * (p + k))
            ifpa = 0.726 - 1.14 * up   # h/d = 0.25
            ipfb = 0.802 - 1.371 * up  # h/d = 1
            # linear interpolation for cpi1
            cpi1scalar = (ipfb - ifpa) / 0.75 * (h / d - 0.25) + ifpa
            cpi1[:] = cpi1scalar
            cpi2[:] = 0
        else: raise ValueError("unexpected bt2")
    else: raise ValueError("unexpected bt")
    return cpi1, cpi2
