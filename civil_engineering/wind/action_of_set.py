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

def xyz_wall(b,d,sd2,Lx,Ly,h,h_roof,bt,bt2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# initial definitions
 par = b / 2 # D E
 per = [0, d] # D E
 z_flat_roof = [b / 2, (h - b) / 2 + b / 2, h/2] #D1 D2 E
 if sd2 == 0:#cw=[D,E]
        par = [par, par]
        per = [per[0], per[1]]
 elif sd2 > 0:#cw=[D1,D2,E]
        par = [par, par, par]
        per = [per[0], per[0], per[1]]
 else:
  raise ValueError("sd2 must be >= 0")
# now for z (vertical) coordinates of pressure points
 if bt == "flat roof":
        if sd2 == 0:#cw=[D,E]
         z=[z_flat_roof[2],z_flat_roof[2]]         
        elif sd2 > 0:#cw=[D1,D2,E] 
         z=z_flat_roof
        else: raise ValueError("sd2 has to be positive") 
 elif bt == "hangar":
        if bt2 == "gable":
         s1=d*h
         s2=d/2*h_roof
         z1=h/2
         z2=h_roof/3+h
         z_slope_roof=(s1*z1+s2*z2)/(s1+s2)
         z=[z_slope_roof, z_slope_roof]          
        else: raise ValueError("unknown roof type bt2")
 else:
  raise ValueError(f"Unknown roof type: {bt}")

    # orientation depending on wind direction
 if b == Lx:
        x, y = per, par
 elif b == Ly:
        y, x = per, par
 else:
  raise ValueError("b must equal either Lx or Ly")
 return np.array(x), np.array(y), np.array(z)

def xyz_roof(e,b,d,bt,bt2,Lx,Ly,h) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# perpendicular direction
 per=[e/8,b/2,b-e/8]  # f1, g, f2
# parallel direction
 par=[e/20,(e/2-e/10)/2+e/10,(d-e/2)/2+e/2]  # f1, g, f2/h/I
 if bt == "flat roof":
        if d < e / 2:
         per = [per[0], per[1], per[2], per[1]]  # f1 g f2 h
         par[1] = d - e / 10
         par = [par[0], par[0], par[0], par[1]]
        else:
         per = [per[0], per[1], per[2], per[1], per[1], per[1]]
         par = [par[0], par[0], par[0], par[1], par[1], par[1]]   
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
        else: raise ValueError("unknown roof type bt2")
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

def reactions(k,bt,bt2,ba,ed,Fwehor,Fwi1hor,Fwi2hor,Fwez,Fwi1z,Fwi2z,b,srt,x,y,z,Lx,Ly,d):
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
    Rhor1 = np.sum(Rhor1i[:ed - 1])#wi1-
    ZRhor1 = np.sum(Rhorzi1[:ed - 1]) / Rhor1 
    Rhor2 = np.sum(Rhor2i[:ed - 1])#wi2+
    ZRhor2 = np.sum(Rhorzi2[:ed - 1]) / Rhor2 
    Rz1 = np.sum(Rz1i[ed - 1:])
    horRz1 = np.sum(Rzhori1[ed - 1:]) / Rz1
    Rz2 = np.sum(Rz2i[ed - 1:])
    horRz2 = np.sum(Rzhori2[ed - 1:]) / Rz2 
    # --- Flat roof case ---
    if bt == "flat roof": # we work according to eurocode -0.3 +0.2
        sI = srt[3]  # MATLAB srt(4)
        if sI > 0:
            Rz3 = np.sum(Rz1i[ed - 1:k - 2]) + Rz1i[k - 1]
            horRz3 = (np.sum(Rzhori1[ed - 1:k - 2]) + Rzhori1[k - 1]) / Rz1 if Rz1 != 0 else 0
            Rz4 = np.sum(Rz2i[ed - 1:k - 2]) + Rz2i[k - 1]
            horRz4 = (np.sum(Rzhori2[ed - 1:k - 2]) + Rzhori2[k - 1]) / Rz2 if Rz2 != 0 else 0
            Rhor = [Rhor1, Rhor2, Rhor1, Rhor2]
            ZRhor = [ZRhor1, ZRhor2, ZRhor1, ZRhor2]
            Rz = [Rz1,Rz2,Rz3,Rz4]
            horRz = [horRz1, horRz2, horRz3, horRz4]
            row = ["wi- I-", "wi+ I-", "wi- I+", "wi+ I+"]
        elif sI==0:
            Rhor = [Rhor1,Rhor2]
            ZRhor = [ZRhor1,ZRhor2]
            Rz = [Rz1,Rz2]
            horRz = [horRz1,horRz2]
            row = ["wi1-","wi2+"]
        else: raise ValueError("sI has to be >=0")
    # --- Hangar case ---
    elif bt == "hangar":
        T=ba["Ly_m"].iloc[0]
        L=ba["Lx_m"].iloc[0]         
        if bt2 == "gable":
           if b==L: #wind2
            Rhor = [Rhor1]
            ZRhor = [ZRhor1]
            Rz = [Rz1]
            horRz = [horRz1]
            row = ["row"]
           elif b==T: #wind1
            Rz1 = np.sum(Rz1i[ed - 1:k - 6])
            horRz1 = np.sum(Rzhori1[ed - 1:k - 6]) / Rz1 if Rz1 != 0 else 0
            Rz2 = np.sum(Rz1i[ed - 1:k - 7]) + Rz1i[k - 1]
            horRz2 = (np.sum(Rzhori1[ed - 1:k - 7]) + Rzhori1[k - 1]) / Rz1 if Rz1 != 0 else 0
            Rz3 = np.sum(Rz1i[ed + 3:k - 1])
            horRz3 = np.sum(Rzhori1[ed + 3:k - 1]) / Rz1 if Rz1 != 0 else 0
            Rz4 = np.sum(Rz1i[k - 6:k])
            horRz4 = np.sum(Rzhori1[k - 6:k]) / Rz1 if Rz1 != 0 else 0
            Rhor = [Rhor1] * 4
            ZRhor = [ZRhor1] * 4
            Rz = [Rz1, Rz2, Rz3, Rz4]
            horRz = [horRz1, horRz2, horRz3, horRz4]
            row = ["--", "-+", "+-", "++"]
           else: raise ValueError("b has to be either T or L ")
        else: raise ValueError("unexpected bt2")     
    else: raise ValueError("unexpected bt")
    # --- Final moment calc ---
    o = len(Rhor)
    Mrv = np.zeros(o)
    d1 = np.full(o, d)
    for i in range(o):
        Mrv[i] =moment_renversant(Rhor[i], ZRhor[i], Rz[i], d1[i], horRz[i])
    # Build DataFrame
    T5 = pd.DataFrame({"row": row,"Rhor": Rhor,"ZRhor": ZRhor,"Rz": Rz,"d": d1,"horRz": horRz,"Mrv": Mrv})
    return T5

def xyz(e,b,d,bt,bt2,Lx,Ly,sd2,h,h_roof) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# --- Determine building height h ---
# --- Call xyz1 for wall coordinates ---
 xw,yw,zw =xyz_wall(b,d,sd2,Lx,Ly,h,h_roof,bt,bt2)
# --- Call xyz2 for roof coordinates ---
 xrt,yrt,zrt =xyz_roof(e,b,d,bt,bt2,Lx,Ly,h)
# --- Concatenate results ---
 x = np.concatenate([np.atleast_1d(xw), np.atleast_1d(xrt)])
 y = np.concatenate([np.atleast_1d(yw), np.atleast_1d(yrt)])
 z = np.concatenate([np.atleast_1d(zw), np.atleast_1d(zrt)])
 return x, y, z
    
def action_of_set(crt2,sw,we,wi1,wi2,cd,s,e,b,d,ba,bt,bt2,Lx,Ly,srt,h,h_roof):
 ncpe=len(we)
 Fwe = np.zeros(ncpe)
 Fwi1 = np.zeros(ncpe)
 Fwi2 = np.zeros(ncpe)
 sd2=sw[1]
 sc=sw[5]
# --- Decide ed, in, cw ---
 if sd2 == 0 and sc == 0:
        ed, in_ = 3, 5
        cw3 = ["D", "E"]
 elif sd2 == 0 and sc > 0:
        ed, in_ = 3, 6
        cw3 = ["D", "E"]
 elif sd2 > 0 and sc == 0:
        ed, in_ = 4, 6
        cw3 = ["D1", "D2", "E"]
 elif sd2 > 0 and sc > 0:
        ed, in_ = 4, 7
        cw3 = ["D1", "D2", "E"]
 else: raise ValueError("sd2 and sc need to be both positive")
    # --- Fill arrays from in_ to n ---
 for i in range(in_ - 1, ncpe):  # MATLAB in:n -> Python in_-1:n-1
        Fwe[i] = -1 * cd * we[i] * s[i] / 1000 
        Fwi1[i] = cd * wi1[i] * s[i]/ 1000
        Fwi2[i] = cd * wi2[i] * s[i]/ 1000
    # --- Special cases depending on sd2 ---
 if sd2 == 0:
        Fwe[0] = cd * we[0] * s[0]/ 1000
        Fwi1[0] = -1 * cd * wi1[0] * s[0]/ 1000
        Fwi2[0] = -1 * cd * wi2[0] * s[0]/ 1000
        Fwe[1] = -1 * cd * we[1] * s[1]/ 1000
        Fwi1[1] = cd * wi1[1] * s[1]/ 1000
        Fwi2[1] = cd * wi2[1] * s[1]/ 1000
 elif sd2 > 0:
        Fwe[0] = cd * we[0] * s[0]/ 1000
        Fwi1[0] = -1 * cd * wi1[0] * s[0]/ 1000
        Fwi2[0] = -1 * cd * wi2[0] * s[0]/ 1000
        Fwe[1] = cd * we[1] * s[1]/ 1000
        Fwi1[1] = -1 * cd * wi1[1] * s[1]/ 1000
        Fwi2[1] = -1 * cd * wi2[1] * s[1]/ 1000
        Fwe[2] = -1 * cd * we[2] * s[2]/ 1000
        Fwi1[2] = cd * wi1[2] * s[2]/ 1000
        Fwi2[2] = cd * wi2[2] * s[2]/ 1000
 else: raise ValueError("sd2 has to be positive") 
#x,y,z coordonees centre de pression
 x,y,z=xyz(e,b,d,bt,bt2,Lx,Ly,sd2,h,h_roof)
 k = len(x)
 Fwez = np.zeros(k)
 Fwi1z = np.zeros(k)
 Fwi2z = np.zeros(k)
 Fwez[ed - 1:k] = Fwe[in_ - 1:ncpe]
 Fwi1z[ed - 1:k] = Fwi1[in_ - 1:ncpe]
 Fwi2z[ed - 1:k] = Fwi2[in_ - 1:ncpe]
 Fwehor = np.zeros(k)
 Fwi1hor = np.zeros(k)
 Fwi2hor = np.zeros(k)
 for i in range(ed - 1):
  Fwehor[i] = Fwe[i]
 Fwi1hor[i] = Fwi1[i]
 Fwi2hor[i] = Fwi2[i]
# --- Combine c labels ---
 c = cw3 + list(crt2)
# --- Build T4 ---
 if sd2>0:
  T4 = pd.DataFrame({"c": c,"Fwehor": Fwehor,"Fwi1hor": Fwi1hor,
 "Fwi2hor": Fwi2hor,"Fwez": Fwez,"Fwi1z": Fwi1z,"Fwi2z": Fwi2z,"x": x,"y": y,"z": z})
 elif sd2==0:
  T4 = pd.DataFrame({"c": c,
 "Fwehor": Fwehor,"Fwi1hor": Fwi1hor,"Fwez": Fwez,"Fwi1z": Fwi1z,"x": x,"y": y,"z": z
  })
 else: raise ValueError("sd2 has to be positive")
 # --- Call react (to be implemented separately) ---
 T5=reactions(k,bt,bt2,ba,ed,Fwehor,Fwi1hor,Fwi2hor,Fwez,Fwi1z,Fwi2z,b,srt,x,y,z,Lx,Ly,d)
 return T4,T5
    