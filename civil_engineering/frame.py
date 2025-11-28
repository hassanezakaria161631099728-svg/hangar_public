import numpy as np
import pandas as pd
from shared_functions.FEM2D import FEM2D_frame_axial_ends
from shared_functions.utils import snow
def frame1(chIII_1,q0,q1,q2,q3,loads,distributed_load_reference):
 nodes=[[0,0],[0,7],[9,9],[18,7],[18,0]]
 elements=[[0,1],[1,2],[2,3],[3,4]]
 constraints = [0,1,2,  3*4, 3*4+1, 3*4+2]   # all DOFs at node 0 and node 4 are fixed on the gound
#frame properties 
 Trafter = pd.read_excel(chIII_1,sheet_name="Ttrafter")
 A_rafter = Trafter["A"].iloc[0]*10e-4
 I_rafter = Trafter["Iy"].iloc[0]*10e-8
 Tcolumn = pd.read_excel(chIII_1,sheet_name="Tcolumn")
 A_column = Tcolumn["A"].iloc[0]*10e-4
 I_column = Tcolumn["Iy"].iloc[0]*10e-8
 E=210e6
 if distributed_load_reference=="global": 
  elem_props=[
# element 0: left column
  {'type':'beam','A':A_column,'I':I_column,'E':E,'w':q0},  #KN/m   
#element 1: left rafter
  {'type':'beam','A':A_rafter,'I':I_rafter,'E':E,'w':q1},   #KN/m
#element 2: right rafter
  {'type':'beam','A':A_rafter,'I':I_rafter,'E':E,'w':q2},  #KN/m
#element 3: right column
  {'type':'beam','A':A_column,'I':I_column,'E':E,'w':q3}, #KN/m
 ]
 elif distributed_load_reference=="local": 
  elem_props=[
# element 0: left column
  {'type':'beam','A':A_column,'I':I_column,'E':E,'q':q0},  #KN/m   
#element 1: left rafter
  {'type':'beam','A':A_rafter,'I':I_rafter,'E':E,'q':q1},   #KN/m
#element 2: right rafter
  {'type':'beam','A':A_rafter,'I':I_rafter,'E':E,'q':q2},  #KN/m
#element 3: right column
  {'type':'beam','A':A_column,'I':I_column,'E':E,'q':q3}, #KN/m
 ]
 else: raise ValueError("unidentified distributed_load_reference")
 u, reactions, axial_avg, axial_ends, shear_forces, end_moments=\
 FEM2D_frame_axial_ends(nodes, elements, elem_props, loads, constraints, default_E=210e6)
 n_elems=axial_ends.shape[0]
 N=np.zeros((2*n_elems,1)) 
 V=np.zeros((2*n_elems,1)) 
 M=np.zeros((2*n_elems,1)) 
 for e in range(n_elems):
  N[2*e,0]=axial_ends[e,0]
  N[2*e+1,0]=axial_ends[e,1]
  V[2*e,0]=shear_forces[e,0]
  V[2*e+1,0]=shear_forces[e,1]
  M[2*e,0]=end_moments[e,0]
  M[2*e+1,0]=end_moments[e,1]
 ux1=u[3]
 ux3=u[9]
 uy2=u[7]
 Vert=reactions[1]+reactions[4]
 Horiz=reactions[0]+reactions[3]
 RD=np.array([Vert,Horiz,ux1,ux3,uy2])
 return N,V,M,RD   

def dead_load(chIII_1,chIII_2,chII_1,chII_2,hangarf):
 ba = pd.read_excel(hangarf,sheet_name="building attributes")
 Lx=ba["Lx_m"].iloc[0]
 Ly=ba["Ly_m"].iloc[0]
 l=Lx/4
 t=Ly/4
#distributed load daN/m
#rafter 
 Trafter = pd.read_excel(chIII_1,sheet_name="Ttrafter")
 q_rafter = Trafter["P"].iloc[0]#rafter daN/m
#purlin
 Tpurlin = pd.read_excel(chII_1,sheet_name="Tpanne")
 if Lx==16 or Lx==18:
  n_purlin=14
 elif Lx==20 or Lx==24:
  n_purlin=18
 else: raise ValueError("this hangar front doesn't exist in our current variantes")
 lm_purlin = Tpurlin["P"].iloc[0]#purlin daN/m
 q_purlin=lm_purlin*n_purlin*t/Lx
#diagonals
 Tdiagonal = pd.read_excel(chIII_1,sheet_name="Tdiagonal")
 ml_diagonal = Tdiagonal["P"].iloc[0] * 2 #diagonal daN/m do not forget there are two corners forming one diagonal
 n_diagonal=8
 lenght_diagonal=(l**2+t**2)**0.5
 q_diagonal=ml_diagonal*n_diagonal*(lenght_diagonal/2)/Lx
#covering
 qgc=15.21 #daN/m2
 q_covering= qgc*t
#sum
 distributed_load=(q_rafter+q_purlin+q_diagonal+q_covering)*1.1
#nodal load on the two columns daN
#colomn
 Tcolumn = pd.read_excel(chIII_1,sheet_name="Tcolumn")
 lm_column = Tcolumn["P"].iloc[0]#column daN/m linear mass
 hc=ba["floor_height_m"].iloc[0]
 q_column=lm_column*hc #daN
#girt
 Tgirt = pd.read_excel(chII_2,sheet_name="Tgirt")
 lm_girt = Tgirt["P"].iloc[0]#girt daN/m linear mass
 n_girt=6
 q_girt=lm_girt*n_girt*t 
#eave purlin
 TEave_purlin = pd.read_excel(chIII_2,sheet_name="TEave_purlin")
 lm_Eave_purlin = TEave_purlin["P"].iloc[0]#Eave_purlin daN/m linear mass
 q_Eave_purlin=lm_Eave_purlin*t  
#cladding
 qgb=11.89
 q_cladding=qgb*(hc-2)*t
#sum
 nodal_load=(q_column+q_girt+q_Eave_purlin+q_cladding)*1.1              
 Tdistributed_load=pd.DataFrame({
"frame_gear":["rafter","purlin","diagonal","covering","sum"],
"distributed_load_daN_m":[q_rafter,q_purlin,q_diagonal,q_covering,distributed_load],
 })
 Tnodal_load=pd.DataFrame({
"frame_gear":["column","girt","Eave_purlin","cladding","sum"],
"nodal_load_daN":[q_column,q_girt,q_Eave_purlin,q_cladding,nodal_load],
 })
 return distributed_load,nodal_load,Tdistributed_load,Tnodal_load

def snow_load(hangarf):
 ba = pd.read_excel(hangarf,sheet_name="building attributes")
 geo = pd.read_excel(hangarf,sheet_name="geography attributes")
 Lx=ba["Lx_m"].iloc[0]
 Ly=ba["Ly_m"].iloc[0]
 qs=snow(geo, ba, Lx, Lx, Ly) #daN/m2 here we take direction 2 to the front b=Lx to get the total load
 t=Ly/4
 snow_load=qs*t #daN/m
 return snow_load

def frame(hangarf,chII_1,chIII_1,chIII_2,chII_2):
#dead load G
 distributed_loadG,loadG,Tdistributed_loadG,Tnodal_loadG=\
 dead_load(chIII_1,chIII_2,chII_1,chII_2,hangarf)
 nodal_loadsG = np.array([[4,loadG*-1], [10,loadG*-1]]) 
 NG,VG,MG,RDG=frame1(chIII_1,0,distributed_loadG,distributed_loadG,0,nodal_loadsG,"global")
#snow1
 qs=snow_load(hangarf)
 nodal_loadss1=[]
 Ns1,Vs1,Ms1,RDs1=frame1(chIII_1,0,qs*0.5,qs,0,nodal_loadss1,"global")
#snow2
 Ns2,Vs2,Ms2,RDs2=frame1(chIII_1,0,qs,qs,0,nodal_loadss1,"global")
#final tables

 T_axial_forces=pd.DataFrame({
 "element":["column1","column1","rafter1","rafter1","rafter2","rafter2","column2","column2"],    
 "node":[0,1,1,2,2,3,3,4],
 "G":NG.reshape(-1),
 "S1":Ns1.reshape(-1),
 "S2":Ns2.reshape(-1),
 })

 T_shear_forces=pd.DataFrame({
 "element":["column1","column1","rafter1","rafter1","rafter2","rafter2","column2","column2"],    
 "node":[0,1,1,2,2,3,3,4],
 "G":VG.reshape(-1),
 "S1":Vs1.reshape(-1),
 "S2":Vs2.reshape(-1),
 })

 T_bending_moments=pd.DataFrame({
 "element":["column1","column1","rafter1","rafter1","rafter2","rafter2","column2","column2"],    
 "node":[0,1,1,2,2,3,3,4],
 "G":MG.reshape(-1),
 "S1":Ms1.reshape(-1),
 "S2":Ms2.reshape(-1),
 })

 T_displacements_reactions=pd.DataFrame({
 "parameter":["vertical reaction V [daN]","horizontal reaction H [daN]","ux1","ux3","uy2"],    
 "G":RDG.reshape(-1),
 "S1":RDs1.reshape(-1),
 "S2":RDs2.reshape(-1),
 })

 return (Tdistributed_loadG,Tnodal_loadG,T_axial_forces,T_shear_forces,
T_bending_moments,T_displacements_reactions)

