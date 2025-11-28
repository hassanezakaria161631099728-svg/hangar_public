import pandas as pd
import numpy as np
import math
def wall():
    # Define pressure coefficients (matching MATLAB arrays)
    cpeD = [1, 0.8]
    cpeE = [-0.3, -0.3]
    cpeA = [-1.3, -1]
    cpeB = [-1, -0.8]
    cpeC = [-0.5, -0.5]    
    # Row labels
    r = ["epf1", "epf10"]    
    # Column names
    columns = ["factors", "D1", "D2", "E", "A", "B", "C"]    
    # Construct dataframe
    Twall = pd.DataFrame({"factors": r,"D1": cpeD,"D2": cpeD,"E": cpeE,"A": cpeA,"B": cpeB,"C": cpeC}, columns=columns)    
    return Twall

def s_cpe_wall_array(sw,cpew):
    sw = np.asarray(sw)
    cpew = np.asarray(cpew)
    sd2 = sw[1]   # D2
    sc = sw[5]    # C
    if sd2 == 0 and sc > 0:   # Case 5
        cpew2 = np.array([cpew[0], -0.3, cpew[2], cpew[3], -0.5])
        sw2 = np.array([sw[0], sw[2], sw[3], sw[4], sw[5]])
        cw = ["D", "E", "A", "B", "C"]
    elif sd2 == 0 and sc == 0:  # Case 4
        cpew2 = np.array([cpew[0], -0.3, cpew[2], cpew[3]])
        sw2 = np.array([sw[0], sw[2], sw[3], sw[4]])
        cw = ["D", "E", "A", "B"]
    elif sd2 > 0 and sc > 0:  # Case 6
        cpew2 = np.array([cpew[0], cpew[1], -0.3, cpew[2], cpew[3], -0.5])
        sw2 = sw
        cw = ["D1", "D2", "E", "A", "B", "C"]
    else:
        raise ValueError("Unexpected surface configuration in vent_surfaces_cpe_mur")
    return cw,sw2,cpew2

def interpolation(cpe1,cpe2,bt,bt2,h1,h2,Lx):
    if bt == "flat roof":
        h = h1
        hacr = 0.6
        cpe = (cpe2 - cpe1) / 0.05 * (hacr / h) + cpe1    
    elif bt == "hangar":
        L = Lx        
        if bt2 == "gable roof":
            alpha = math.degrees(math.atan((h1 - h2) / (L / 2)))
            cpe = (cpe2 - cpe1) / 10 * (alpha - 5) + cpe1    
        else:
         raise ValueError("Unexpected L")
    else:
     raise ValueError("Unexpected bt") 
    return cpe

def roof(ba,bt,bt2,b,h1,h2):
    Lx = ba["Lx_m"].iloc[0]  
    Ly = ba["Ly_m"].iloc[0]  
    if bt=="flat roof":
     Troof=flat_roof(Lx,bt,bt2,h1,h2)
    elif bt=="hangar":
        L = Lx  
        T = Ly  
        if bt2=="gable":
            if b==T:
                Troof=roof_0deg(h1,h2,L)
            elif b==L:
                Troof=roof_90deg(h1,h2,L)
            else:
              raise ValueError("Unexpected b")  
        else: 
         raise ValueError("Unexpected bt2")
    else: 
      raise ValueError("Unexpected bt")
    return Troof

def flat_roof(Lx,bt,bt2,h1,h2):
    # Equivalent of bt and bt2
    #bt = ["flat roof"]
    #bt2 = ["flat roof"]    
    # Constants
    K = [-1.8, -2.5, -1.2, -2.0]  # cpe 0
    V = [-1.4, -2.0, -0.9, -1.6]  # cpe hp/h=0.05    
    # Placeholder for cpe values
    cpe = np.zeros(4)    
    for i in range(4):
        cpe[i] =interpolation(K[i],V[i],bt,bt2,h1,h2,Lx)    
    # Construct arrays
    cpeF = [cpe[1], cpe[0]]
    cpeG = [cpe[3], cpe[2]]
    cpeH = [-1.2, -0.7]
    cpeIn = [-0.2, -0.2]
    cpeIp = [0.2, 0.2]    
    # Row names
    r = ["epf1", "epf10"]    
    # Build dataframe (like MATLAB table)
    data={"factors": r,"F-": cpeF,"G-": cpeG,"H-": cpeH,"I-": cpeIn,"I+": cpeIp,}    
    Troof = pd.DataFrame(data)
    return Troof

def roof_0deg(h1,h2,Lx):
    # Labels (for debugging, not used in final dataframe)
    c = ['F10-', 'F1-', 'G10-', 'G1-', 'H10-', 'H1-', 'I10-', 'I1-', 'J10-', 'J1-',
         'F+', 'G+', 'H+', 'I+', 'J+']    
    # Constants
    K = [-1.7, -2.5, -1.2, -2.0, -0.6, -1.2, -0.6, -0.6, -0.6, -0.6,
         0.0, 0.0, 0.0, -0.6, 0.2]
    V = [-0.9, -2.0, -0.8, -1.5, -0.3, -0.3, -0.4, -0.4, -1.0, -1.5,
         0.2, 0.2, 0.2, 0.0, 0.0]    
    n = len(K)
    cpe = np.zeros(n)    
    # Roof types
    bt = "hangar"
    bt2 = "gable roof"    
    # Compute cpe values using vent_interpolation
    for i in range(n):
        cpe[i] =interpolation(K[i], V[i],bt,bt2,h1,h2,Lx)    
    # Grouping values like MATLAB
    cpeF = [cpe[1], cpe[0], cpe[10]]
    cpeG = [cpe[3], cpe[2], cpe[11]]
    cpeH = [cpe[5], cpe[4], cpe[12]]
    cpeI = [cpe[7], cpe[6], cpe[13]]
    cpeJ = [cpe[9], cpe[8], cpe[14]]    
    # Row labels
    r = ["epf1-", "epf10-", "epf+"]    
    # Build dataframe
    data = {"factors": r,"F": cpeF,"G": cpeG,"H": cpeH,"I": cpeI,"J": cpeJ,}    
    Troof = pd.DataFrame(data)
    return Troof

def roof_90deg(h1,h2,Lx):
    # Constants
    K = [-1.6, -2.2, -0.7, -0.6]
    V = [-1.3, -2.0, -0.6, -0.5]    
    n = len(K)
    cpe = np.zeros(n)    
    # Roof types
    bt = "hangar"
    bt2 = "gable roof"    
    # Compute interpolated cpe values
    for i in range(n):
        cpe[i] =interpolation(K[i], V[i],bt,bt2,h1,h2,Lx)    
    # Grouping like in MATLAB
    cpeF = [cpe[1], cpe[0]]
    cpeG = [-2.0, -1.3]   # fixed values, not interpolated
    cpeH = [-1.2, cpe[2]]
    cpeI = [cpe[3], cpe[3]]    
    # Row labels
    r = ["epf1-", "epf10-"]    
    # Build dataframe
    data = {"factors": r,"F": cpeF,"G": cpeG,"H": cpeH,"I": cpeI,}    
    Troof = pd.DataFrame(data)
    return Troof

def cpe_from_s(n,T,s):
    epf10 = T.iloc[1, 1:n].to_numpy()  # row 2 (index 1), columns 2:n
    epf1 = T.iloc[0, 1:n].to_numpy()   # row 1 (index 0), columns 2:n    
    epf = np.zeros(n - 1)    
    for i in range(n - 1):
        if s[i] == 0:
            epf[i] = 0
        elif s[i] <= 1:
            epf[i] = epf1[i]
        elif 1 < s[i] < 10:
            epf[i] = epf1[i] + (epf10[i] - epf1[i]) * np.log10(s[i])
        elif s[i] >= 10:
            epf[i] = epf10[i]    
    return epf

def s_cpe_roof_array(Troof,b,ba,srt,bt,bt2):
    n = Troof.shape[1]    
    # Placeholder for depf (must be defined separately)
    cpen=cpe_from_s(n,Troof,srt)      
    if bt == "flat roof":
        sI = srt[3]  # MATLAB index (4th element)
        if sI == 0:
            cpe = [cpen[0], cpen[1], cpen[0], cpen[2]]
            s = [srt[0], srt[0], srt[1], srt[2]]
            c = ["F1-", "G1-", "F2-", "H-"]
        elif sI > 0:
            cpe = [cpen[0], cpen[1], cpen[0], cpen[2], -0.2, 0.2]
            s = [srt[0], srt[0], srt[1], srt[2], srt[3], srt[3]]
            c = ["F1-", "G1-", "F2-", "H-", "I-", "I+"]    
        else: raise ValueError("sI value is negative") 
    elif bt == "hangar" and bt2 == "gable":
        L = ba["Lx_m"].iloc[0]
        T = ba["Ly_m"].iloc[0]        
        if b == T:  # wind1
            cpep = Troof.iloc[2, 1:].tolist()  # row 3, cols 2:n
            cpe = [cpen[0], cpen[1], cpen[0], cpen[2], cpen[3], cpen[4],
                   cpep[0], cpep[1], cpep[0], cpep[2], cpep[3], cpep[4]]
            s = [srt[0], srt[1], srt[0], srt[2], srt[3], srt[4],
                 srt[0], srt[1], srt[0], srt[2], srt[3], srt[4]]
            c = ["F1-", "G-", "F2-", "H-", "J-", "I-",
                 "F1+", "G+", "F2+", "H+", "J+", "I+"]        
        elif b == L:  # wind2
            cpe = [cpen[0], cpen[1], cpen[1], cpen[0],
                   cpen[2], cpen[2], cpen[3], cpen[3]]
            s = [srt[0], srt[1], srt[1], srt[0],
                 srt[2], srt[2], srt[3], srt[3]]
            c = ["F1", "G1", "G2", "F2", "H1", "H2", "I1", "I2"]    
        else: raise ValueError("unexpected perpendicular distance b value")
    else:
        raise ValueError("unexpected bt or bt2")    
    # Return in same order: ct2, st2, cpet
    return c,s,cpe
