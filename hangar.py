#%%
import pandas as pd
import os
from shared_functions.wind.wind import wind,dimensions
from shared_functions.expxlsx import expxlsx
#from shared_functions.vent.surfaces import wall_parallel,wall_perpendicular,roof_array,roof_list
#from shared_functions.vent.cpe import wall,s_cpe_wall_array,cpe_from_s,roof,s_cpe_roof_array
#from shared_functions.vent.cpi import cpi
#from shared_functions.vent.action_of_set import xyz,reactions,action_of_set
#from shared_functions.expxlsx import expxlsx
excel_path = os.path.join(os.path.dirname(__file__),"excel","eurocode.xlsx")
eurocode = pd.ExcelFile(excel_path)
# --- Excel file path (relative to project root) ---
excel_path = os.path.join(os.path.dirname(__file__), "excel", "hangar.xlsx")
# Load Excel
hangarf = pd.ExcelFile(excel_path)# Read Excel input
geo = pd.read_excel(hangarf,sheet_name="geography attributes")
wzs = pd.read_excel(eurocode,sheet_name="wind zones")
gcs = pd.read_excel(eurocode,sheet_name="ground categories")
ba = pd.read_excel(hangarf,sheet_name="building attributes")
Lx = ba["Lx_m"].iloc[0]
Ly = ba["Ly_m"].iloc[0]
direction1='wind1'
direction2='wind2'
b,d=dimensions(Lx,Ly,direction1)
bt=ba["bt"].iloc[0]
bt2=ba["bt2"].iloc[0]
T1,T2,T3,T4,T5,Troof1,Twall=wind(ba,Lx,Ly,direction1,geo,wzs,gcs)
T6,T7,T8,T9,T10,Troof2,_=wind(ba,Lx,Ly,direction2,geo,wzs,gcs)
#s = snow(geo, ba, Ly, Lx, Lx, Ly)  # daN/m²
# Save Chapter I tables
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
Tables = [T1,T2,T3,T4,T5,Troof1,Twall,T6,T7,T8,T9,T10,Troof2]
sheetNames = ["T1", "T2", "T3", "T4", "T5", "Troof1", "Twall",
              "T6", "T7", "T8", "T9", "T10", "Troof2"]
expxlsx(Tables, os.path.join(excelDir, "chapterI.xlsx"), sheetNames)
#%% ---- Chapter II-1 ----
import pandas as pd
import os
# Directory where hangar.py lives
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
beamT = os.path.join(excelDir, "tableaudesprofiles.xlsx")
chI = os.path.join(excelDir, "chapterI.xlsx")
hangarf = os.path.join(excelDir, "hangar.xlsx")
from shared_functions.purlin import purlin
from shared_functions.expxlsx import expxlsx
ba = pd.read_excel(hangarf,sheet_name="building attributes")
row = ba.squeeze()
bt2 = row["bt2"]   
Lx = row["Lx_m"]   
Ly = row["Ly_m"]   
b1=Ly
b2=Lx
Tpurlin,T2,loads,acp,combdel,combV,combM,T8,T9,T10=purlin(b1,b2,hangarf,chI,beamT)
Tables = [Tpurlin,T2,loads,acp,combdel,combV,combM,T8,T9,T10]
sheetNames = ["Tpurlin","T2","loads","acp","combdel","combV","combM","T8","T9","T10"]
expxlsx(Tables, os.path.join(excelDir, "chapterII-1.xlsx"), sheetNames)

#%% ---- Chapter II-2 ----
import pandas as pd
import os
from shared_functions.girt_internal_column import girt,internal_column
from shared_functions.expxlsx import expxlsx 
# Directory where hangar.py lives
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
beamT = os.path.join(excelDir, "tableaudesprofiles.xlsx")
chI = os.path.join(excelDir, "chapterI.xlsx")
hangarf = os.path.join(excelDir, "hangar.xlsx")
Tgirt, T2, T3, T4=girt(hangarf, chI, beamT)
T5, Tinter_column, T6, T7, T8, T9 = internal_column(Tgirt, hangarf, chI, beamT)
Tables = [Tgirt, T2, T3, T4, T5, Tinter_column, T6, T7, T8, T9]
sheetNames = ["Tgirt", "T2", "T3", "T4", "T5", "Tinter_column", "T6", "T7", "T8", "T9"]
expxlsx(Tables, os.path.join(excelDir, "chapterII-2.xlsx"), sheetNames)

#%% ---- Chapter III-1 ----
import os
from shared_functions.truss import roof_bracing
from shared_functions.expxlsx import expxlsx
# Directory where hangar.py lives
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
beamT = os.path.join(excelDir, "tableaudesprofiles.xlsx")
chI = os.path.join(excelDir, "chapterI.xlsx")
hangarf = os.path.join(excelDir, "hangar.xlsx")
chII_1 = os.path.join(excelDir, "chapterII-1.xlsx")
T1, T2, Trafter, Tcolumn, Tdiagonal, T6, T7, T8, T9, T10, T11, T12, T13,FV1,FV2=\
roof_bracing(hangarf, beamT, chI, chII_1)
# If you want to export:
#Tables = [T1, T2, Trafter, Tcolumn, Tdiagonal, T6, T7, T8, T9, T10, T11, T12, T13]
#sheetNames = ["T1","T2","Ttrafter","Tcolumn","Tdiagonal","T6","T7","T8","T9","T10","T11","T12","T13"]
#expxlsx(Tables, os.path.join(excelDir,"chapterIII-1.xlsx"), sheetNames)
#%% ---- Chapter III-2 ----
import os
from shared_functions.truss import wall_bracing
from shared_functions.expxlsx import expxlsx
# Directory where hangar.py lives
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
beamT = os.path.join(excelDir, "tableaudesprofiles.xlsx")
chIII_1 = os.path.join(excelDir, "chapterIII-1.xlsx")
hangarf = os.path.join(excelDir, "hangar.xlsx")
T1, T2, TEave_purlin, Tdiagonal, T3, T4, T5 = wall_bracing(beamT, chIII_1, hangarf)
#Tables = [T1, T2, TEave_purlin, Tdiagonal, T3, T4, T5]
#sheetNames = ["T1", "T2", "TEave_purlin", "Tdiagonal", "T3", "T4", "T5"]
#expxlsx(Tables, os.path.join(excelDir, "chapterIII-2.xlsx"), sheetNames)


# %% chapter IV frame 
import pandas as pd
import numpy as np
from shared_functions.frame import frame 
from shared_functions.expxlsx import expxlsx
# Directory where hangar.py lives
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
hangarf = os.path.join(excelDir, "hangar.xlsx")
chII_1 = os.path.join(excelDir, "chapterII-1.xlsx")
chII_2 = os.path.join(excelDir, "chapterII-2.xlsx")
chIII_1 = os.path.join(excelDir, "chapterIII-1.xlsx")
chIII_2 = os.path.join(excelDir, "chapterIII-2.xlsx")
E=210e6 #Kpa
(Tdistributed_loadG,Tnodal_loadG,T_axial_forces,T_shear_forces,
T_bending_moments,T_displacements_reactions)=frame(hangarf,chII_1,chIII_1,chIII_2,chII_2)
Tables = [Tdistributed_loadG,Tnodal_loadG,T_axial_forces,T_shear_forces,T_bending_moments,T_displacements_reactions]
sheetNames = ["Tdistributed_loadG","Tnodal_loadG","T_axial_forces","T_shear_forces","T_bending_moments","T_displacements_reactions"]
expxlsx(Tables, os.path.join(excelDir, "chapterIV-1.xlsx"), sheetNames)
#%%
import pandas as pd
import numpy as np
import os
from shared_functions.FEM import plot_frame2,FEM2D_frame2
from shared_functions.FEM2D import FEM2D_frame_axial_ends
from shared_functions.expxlsx import expxlsx
nodes=[[0,0],[0,7],[9,9],[18,7],[18,0]]
elements=[[0,1],[1,2],[2,3],[3,4]]
constraints = [0,1,2,  3*4, 3*4+1, 3*4+2]   # all DOFs at node 0 and node 4 are fixed on the gound
#frame properties 
A_rafter = 84.5*10e-4
I_rafter = 23130*10e-8
A_column = 131.4*10e-4
I_column = 19270*10e-8
E=210e6
elem_props = [
# element 0: left column
    {"E": E, "A": A_column, "I": I_column, "q": 0, "load_type": "global"},
#element 1: left rafter
    {"E": E, "A": A_rafter, "I": I_rafter, "q": -195, "load_type": "global"},
#element 2: right rafter
    {"E": E, "A": A_rafter, "I": I_rafter, "q": -195, "load_type": "global"},
#element 3: right column
    {"E": E, "A": A_column, "I": I_column, "q": 0, "load_type": "global"},
]
#loads
loads = [[4,-1387.1],[10,-1387.1]] 
plot_frame2(nodes, elements, elem_props, loads ,constraints, load_scale=0.001, q_scale=0.001, show_labels=True)
u, R, N, V, M=\
FEM2D_frame2(nodes, elements, elem_props, loads, constraints, default_E=210e6)
# Directory where hangar.py lives
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
Tables = [pd.DataFrame(u),pd.DataFrame(R),pd.DataFrame(N),pd.DataFrame(V),pd.DataFrame(M)]
sheetNames = ["V","M","RD","u"]
expxlsx(Tables, os.path.join(excelDir, "frame.xlsx"), sheetNames)

# %%
import numpy as np
import pandas as pd
import os
from shared_functions.FEM import create_X_horizontal_truss,plot_truss,FEM2D_frame2,plot_frame2,FEM2D
from shared_functions.expxlsx import expxlsx
#truss X bracing roof
#nodes, elements = create_X_horizontal_truss(4, 4.5, 4)
#nodes=[[0,0],[4.5,0],[9,0],[13.5,0],[18,0],[0,4],[4.5,4],[9,4],[13.5,4],[18,4]]
#elements=[[0,5],[1,6],[2,7],[3,8],[4,9], #purlins
#          [0,1],[1,2],[2,3],[3,4],[5,6],[6,7],[7,8],[8,9],#rafter
#          [0,6],[5,1],[1,7],[6,2],[2,8],[7,3],[3,9],[8,4],#diagonals  
#       ]                                   
#truss A wall bracing 
nodes=[[0,0],[0,7],[2,7],[4,7],[4,0]]
elements=[[1,2],[2,3],[0,1],[4,3],[0,2],[2,4]]
#Loads for each truss case
#loads = np.array([[1,1282.6], [3,2815.6], [5,3066.1], [7,2815.6], [9,1282.6]]) 
#loads = np.array([[1,-1157.03876], [3,-2200.230815], [5,-2246.597696], [7,-2087.170172], [9,-721.0281682]]) 
loads1=[[0,3334.130615],[2,8965.394952]]
loads2=[[0,-3334.130615],[2,-7786.433877]]
constraints = np.array([0, 1, 8, 9])
plot_truss(nodes, elements, loads1, constraints,load_scale=10e-5)
plot_truss(nodes, elements, loads2, constraints,load_scale=10e-5)
E=210e6
# Areas *1e-4 convert cm2 to m2
A_h = 21.236 * 1e-4   # horizontal eave purlin 
A_v = 131.364 * 1e-4    # vertical column
A_d = 12.267  * 1e-4   # diagonal 
u1, reactions1, axial_forces1, elem_types=\
FEM2D(A_h, A_v, A_d, nodes, elements, loads1, constraints, E=210e6)
u2, reactions2, axial_forces2, _ =\
FEM2D(A_h, A_v, A_d, nodes, elements, loads2, constraints, E=210e6)
# Directory where hangar.py lives
baseDir = os.path.dirname(__file__)
# Excel folder is inside this directory
excelDir = os.path.join(baseDir, "excel")
Tables = [pd.DataFrame(u1),pd.DataFrame(reactions1),pd.DataFrame(axial_forces1),pd.DataFrame(elem_types),
pd.DataFrame(u2),pd.DataFrame(reactions2),pd.DataFrame(axial_forces2)]
sheetNames = ["u1","reactions1","axial_forces1","elem_types","u2","reactions2","axial_forces2"]
expxlsx(Tables, os.path.join(excelDir, "roofbracing.xlsx"), sheetNames)
