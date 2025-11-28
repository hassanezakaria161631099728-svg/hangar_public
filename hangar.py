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
from shared_functions.FEM2D import plot_frame
nodes=[[0,0],[0,7],[9,9],[18,7],[18,0]]
elements=[[0,1],[1,2],[2,3],[3,4]]

plot_frame(nodes, elements,save_path="steel_frame.png")

# %%
