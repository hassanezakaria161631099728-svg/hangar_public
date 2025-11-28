import numpy as np
import pandas as pd
import math
from typing import Any, Tuple
# Pull helpers directly from vent/__init__.py
from .cpe import (wall,roof,s_cpe_wall_array,s_cpe_roof_array,cpe_from_s)
from .cpi import cpi               # if cpi is its own file/module
from .action_of_set import action_of_set
from .surfaces import (wall_perpendicular,wall_parallel,roof_list,roof_array)

def wind(ba,Lx,Ly,direction,geo,wzs,gcs) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
 bt=ba["bt"].iloc[0]
 bt2 = ba["bt2"].iloc[0]  # column 7 -> index 6
# --- call dw (direction of wind) ---
# This function is expected to have side effects like setting global variables or writing files in the MATLAB code.
# Keep the same call signature; implement dw separately in Python.
 b,d=dimensions(Lx,Ly,direction)
 bt=ba["bt"].iloc[0]
 bt2=ba["bt2"].iloc[0]
 T1,T2,qpze1,qpze2=pression_dynamique_de_pointe(b,d,geo,ba,wzs,gcs)
# --- surfaces in m^2: e = T1{1,4}  (MATLAB 1-based) ---
 e=T1["e_m"].iloc[0]   # zero-based: row 0, col 3
# --- walls surfaces
 h1,h2,sd1,sd2,se=wall_perpendicular(ba,b)
 cd,sa,sb,sc=wall_parallel(h1,h2,b,d,bt,e)
# --- sw (stacked vertically)
 sw=np.array([sd1,sd2,se,sa,sb,sc], dtype=float)
#cpe
 Twall=wall()
 nw=Twall.shape[1]
 cpew=cpe_from_s(nw,Twall,sw)
 cw,sw2,cpew2=s_cpe_wall_array(sw,cpew)
#roof
# fshroof returns (something, srt) in MATLAB where first output is unused (~)
 sf,sg,sh,sJ,sI=roof_list(b,d,ba,bt2,e)
 srt,crt=roof_array(sf,sg,sh,sJ,sI,bt,bt2,b,ba,h1,h2) 
# --- epf roof
 Troof=roof(ba,bt,bt2,b,h1,h2)
 crt2,srt2,cpert=s_cpe_roof_array(Troof,b,ba,srt,bt,bt2)
  # Concatenate epf arrays similar to MATLAB epf=[epfw;epfrt]
    # In Python we will just create a list or numpy array depending on inputs
 try:
    cpe = np.concatenate([np.atleast_1d(cpew2), np.atleast_1d(cpert)])
    c= np.concatenate([np.atleast_1d(cw), np.atleast_1d(crt2)])
    s= np.concatenate([np.atleast_1d(sw2), np.atleast_1d(srt2)])
    ncpe=len(cpe)
 except Exception:
 # fallback to lists
  cpe = list(cpew) + list(cpert)
 c=list(cw) + list(crt2)
 s=list(sw2) + list(srt2)
 ncpe=len(cpe)
# --- ipf internal pressure factors ---
 cpi1,cpi2=cpi(bt,bt2,ba,b,d,ncpe,h1)
# --- aerodynamic pressure exerted on surfaces ---
 we=wewi(cpe,sw,qpze1,qpze2)
 wi1=wewi(cpi1,sw,qpze1,qpze2)
 wi2=wewi(cpi2,sw,qpze1,qpze2)
# --- build T3 table depending on roof type ---
 if bt == "flat roof":
  T3 = pd.DataFrame({"c": c,"epf": cpe,"ipf1": cpi1,"ipf2": cpi2,"we": we,"wi1": wi1,"wi2": wi2})
 elif bt == "hangar":
  T3 = pd.DataFrame({"c": c,"epf": cpe,"ipf1": cpi1,"we": we,"wi1": wi1})
 else: raise ValueError("Unexpected building type")
# --- action of set ---
 T4,T5=action_of_set(crt2,sw,we,wi1,wi2,cd,s,e,b,d,ba,bt,bt2,Lx,Ly,srt,h2,2)
 return T1,T2,T3,T4,T5,Troof,Twall

def dimensions(Lx,Ly,direction):
    if direction == 'wind1':
        b = Ly
        d = Lx
    elif direction == 'wind2':
        b = Lx
        d = Ly
    else: b,d=0,0 
    return b, d

def pdp(KT, ze, z0, zmin, qref):
    CT= 1    
    if zmin < ze:
        crze = KT * np.log(ze / z0)
        Ivze = 1.0 / (CT * np.log(ze / z0))
        ceze = CT**2*crze**2*(1+7*Ivze)
        qpze = qref * ceze
    else: crze, Ivze, ceze, qpze=0,0,0,0
    return crze, Ivze, ceze, qpze

def pression_dynamique_de_pointe(b,d,geo,ba,wzs,gcs):
    # extract geo
    wz = geo.iloc[0,0]
    gc = geo.iloc[0,1]
    # look up ground category row
    Tgc=gcs[gcs.iloc[:,0]==gc].iloc[0]
    KT=Tgc.iloc[1]
    z0=Tgc.iloc[2]
    zmin=Tgc.iloc[3]
    # look up wind zone row
    Twz = wzs[wzs.iloc[:, 0] == wz].iloc[0]
    qref = Twz.iloc[1]
    # building height depending on type
    floor_height = ba["floor_height_m"].iloc[0]
    number_floors = ba["n_floors"].iloc[0]
    hp=ba["hp_m"].iloc[0] 
    h=floor_height*number_floors+hp
    # effective dimension
    e=min(2*h,b)
    # determine ze
    if h<=b:
        ze1, ze2 = h, 0
    elif b<h:
        ze1, ze2 = b, h
    # call pdp1 at ze1
    crze1, Ivze1, ceze1, qpze1 = pdp(KT, ze1, z0, zmin, qref)
    qpze2=None 
    if h<=b:
        var1 = ["b_m", "d_m", "h_m", "e_m", "ze_m", "crze"]
        var2 = ["Ivze", "Ceze", "qpze_N_m_2"]
        T1 = pd.DataFrame([[b,d,h,e,ze1,crze1]], columns=var1)
        T2 = pd.DataFrame([[Ivze1,ceze1,qpze1]], columns=var2)
    elif b<h:
        crze2,Ivze2,ceze2,qpze2=pdp(KT,ze2,z0,zmin,qref)
        var1 = ["b_m", "d_m", "h_m", "e_m", "ze1_m", "ze2_m", "crze1", "crze2"]
        var2 = ["Ivze1", "Ivze2", "Ceze1", "Ceze2", "qpze1_N_m_2", "qpze2_N_m_2"]
        T1 = pd.DataFrame([[b, d, h, e, ze1, ze2, crze1, crze2]], columns=var1)
        T2 = pd.DataFrame([[Ivze1,Ivze2,ceze1,ceze2,qpze1,qpze2]],columns=var2)
    return T1,T2,qpze1,qpze2

def wewi(pf,sw,qpze1,qpze2=None): #aerodynamic pressure on surfaces
    sd2 = sw[1]
    w=np.zeros_like(pf,dtype=float)
    if sd2 == 0:
        w[:] = pf * qpze1
    elif sd2>0:
        w[0]=pf[0]*qpze1
        w[1:]=pf[1:]*qpze2
    else: raise ValueError("sd2 is negative")
    return w

