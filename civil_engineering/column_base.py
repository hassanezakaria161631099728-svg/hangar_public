# column base 
# poteau: h,b,tf,tw
# concrete foundation: fc28,Ecm,taus
# anchor bolts: d,d0,A,fu,dm,Asp,H,D,l2
# plate:tp,fu,fy
# case 1 compression Ncsd Vsd unit KN
import pandas as pd
import os
def compression(input,steel):
#loading input tables
 general = pd.read_excel(input,sheet_name="general")
 plate = pd.read_excel(input,sheet_name="plate")
#loading input data
#column 
 column_type = general["column_type"].iloc[0]
 if column_type=="tube":
  b=general["tube_b_mm"].iloc[0]
  h=general["tube_h_mm"].iloc[0]
 else:
  column_data = pd.read_excel(steel,sheet_name=column_type)
  column_class = general["column_class"].iloc[0]
  selected_column = column_data[column_data[column_type] == column_class]
  b=selected_column["b"].iloc[0]*10
  h=selected_column["h"].iloc[0]*10
  tf=selected_column["tf"].iloc[0]*10
  tw=selected_column["tw"].iloc[0]*10
#foundation concrete
 fck = general["fck_N_mm2"].iloc[0] 
#plate
 tp=plate["tp_mm"].iloc[0]
 steel_grade=plate["steel_grade"].iloc[0]
 grade_data = pd.read_excel(steel,sheet_name="grade")
 selected_grade = grade_data[grade_data["steel_grade"] == steel_grade]
 fy=selected_grade["fy_N_mm2"].iloc[0]  
 bp=plate["bp"].iloc[0]
 hp=plate["hp"].iloc[0]
 Ncsd=general["Ncsd_KN"].iloc[0] #KN applicated compression force
 Msd=general["Msd"].iloc[0] #KN applicated compression force
#start
#foundation concrete
 kj=1
 gamac=1.5 # concrete factor
 fj=2/3*kj*fck/gamac #N/mm2
 c=tp*(fy/(3*fj*1.1))**0.5 #mm
 x=(hp-h)/2
 y=(bp-b)/2
#1-effective range of compression
#compressed section 
 if column_type=="tube":
  if c<y:
   tube_width=b+2*c
  else:
   tube_width=bp
  if c<x:
   tube_lenght=h+2*c
  else:
   tube_lenght=hp 
   effectif_surface=tube_width*tube_lenght
 else:
  if c<y:
   flange_lenght=b+2*c    
  else:
   flange_lenght=bp
  if c<x:
   flange_width=tf+2*c    
  else:
   flange_width=tf+c+x
   flange_surface=flange_width*flange_lenght
   web_surface=(h-2*c-2*tf)*(tw+2*c)
   effectif_surface=flange_surface*2+web_surface

 if Msd==0:
  segma=Ncsd*1000/effectif_surface # N/mm2
 elif Msd>0:
  segma=0 
 else: raise ValueError("Msd has to be positive") 
 Ncrd=fj*effectif_surface/1000 #KN
#steel plate
 vsd=segma*c # N
 Vplrd=tp*fy/(3**0.5*1.1) #N
 if vsd<=0.5*Vplrd:
  print("shear force resistance for plate has been aquired")
 else:
  print("shear force will have influence over momentum for plate")
 if column_type=="tube":
  T1=pd.DataFrame({"fj":fj,"c":c,"x":x,"y":y,"bp":bp,"hp":hp,"tube_width":tube_width,
  "tube_lenght":tube_lenght,"effectif_surface":effectif_surface,"Ncsd":Ncsd,"Ncrd":Ncrd,
  "segma":segma,"vsd":vsd,"Vplrd":Vplrd,"Msd":Msd},index=[0])
 else:
  T1=pd.DataFrame({"fj":fj,"c":c,"x":x,"y":y,"flange_width":flange_width,"flange_lenght":flange_lenght,
  "flange_surface":flange_surface,"web_surface":web_surface,"effectif_surface":effectif_surface,
  "Ncsd":Ncsd,"Ncrd":Ncrd,"segma":segma,"vsd":vsd,"Vplrd":Vplrd,"Msd":Msd},index=[0])   
 return T1

def combined_action(input,steel):
#loading input tables
 general = pd.read_excel(input,sheet_name="general")
 plate = pd.read_excel(input,sheet_name="plate")
 bolt_caliber_data = pd.read_excel(steel,sheet_name="bolt_caliber")
#loading input data 
#actions
 Vsd=general["Vsd_KN"].iloc[0]
 Ntsd=general["Ntsd_KN"].iloc[0]
#foundation concrete
 fck=general["fck_N_mm2"].iloc[0]
 if fck==20: Ecm=29 
 elif fck==25: Ecm=30.5 
 elif fck==30: Ecm=32 
 else: raise ValueError("no corresponding Ecm value for fck outside 20 25 30 ") 
#bolt
 class_x=general["class_x"].iloc[0]
 fub=class_x*100 #N/mm2 
 d=general["bolt_d_mm"].iloc[0]
 bolt_attributes = bolt_caliber_data[bolt_caliber_data["d_mm"] == d]
 A=bolt_attributes["A_mm2"].iloc[0]
 d0=bolt_attributes["d0_mm"].iloc[0]
 As=bolt_attributes["As_mm2"].iloc[0]
#plate
 steel_grade_data = pd.read_excel(steel,sheet_name="grade")
 n_total=plate["n_total"].iloc[0] # number of bolts
 steel_grade=plate["steel_grade"].iloc[0] 
 steel_grade_attributes = steel_grade_data[steel_grade_data["steel_grade"] == steel_grade]
 fu=steel_grade_attributes["fu_N_mm2"].iloc[0]
 p=plate["p_mm"].iloc[0]
 tp=plate["tp_mm"].iloc[0]
 l=plate["l_mm"].iloc[0]
 hp=plate["hp_mm"].iloc[0]
 e=(hp-l)/2 
#start
 alpha=min(e/(3*d0),p/(3*d0)-1/4,fub/fu,1)
#verification of shear forces Fvsd<Frd
 Fvsd=Vsd/n_total
#three to choose the minimum from
 Fvrd=0.6*fub*A/1.25 #N shear resistance 
 Fbrd=2.5*alpha*fu*d*tp/1.25 #N diametral pressure resistance
 Fbcrd=0.29*d**2/1.25*(fck*Ecm*1000)**0.5 #N concrete bursting resistance 
#
 Frd=min(Fvrd,Fbrd,Fbcrd)/1000 #KN
 Ftsd=Ntsd/n_total
 Ftrd=0.9*As*fub/1.25/1000
 result=Ftsd/(1.4*Ftrd)+Fvsd/Frd
 if Ntsd==0:
  if result<=1:
   print("shear force resistance on column base has been aquired")
  else:
   print("shear force resistance on column base has failed")
 elif Ntsd>=0:
  if result<=1:
   print("shear force and tension resistance on column base has been aquired")
  else: 
   print("shear force and tension resistance on column base has failed")
 T2=pd.DataFrame({"n_total":n_total,"Fvsd":Fvsd,"alpha":alpha,"Fvrd":Fvrd,"Fbrd":Fbrd,"Fbcrd":Fbcrd,
 "Frd":Frd,"Ntsd":Ntsd,"Ftsd":Ftsd,"Ftrd":Ftrd,"result":result},index=[0]) 
#      (n_total,Fvsd,alpha,Fvrd,Fbrd,Fbcrd,Frd,Ntsd,Ftsd,Ftrd,result)
 return T2 

def tension(input,steel):
#loading input tables
 general = pd.read_excel(input,sheet_name="general")
 plate = pd.read_excel(input,sheet_name="plate")
#loading input data
#column used 
 column_type = general["column_type"].iloc[0]
 column_data = pd.read_excel(steel,sheet_name=column_type)
 column_class = general["column_class"].iloc[0]
 selected_column = column_data[column_data[column_type] == column_class]
 tf=selected_column["tf"].iloc[0]*10
 tw=selected_column["tw"].iloc[0]*10
#plate
 tp=plate["tp_mm"].iloc[0]
 steel_grade=plate["steel_grade"].iloc[0]
 grade_data = pd.read_excel(steel,sheet_name="grade")
 selected_grade = grade_data[grade_data["steel_grade"] == steel_grade]
 fu=selected_grade["fu_N_mm2"].iloc[0]  
 fy=selected_grade["fy_N_mm2"].iloc[0]  
 bp=plate["bp_mm"].iloc[0]
 p=plate["p_mm"].iloc[0]
 l=plate["l_mm"].iloc[0]
 af=plate["af"].iloc[0]
 aw=plate["aw"].iloc[0]
 eex=plate["eex"].iloc[0]
 kex=plate["kex"].iloc[0]
 set1=plate["set1"].iloc[0]
 set2=plate["set2"].iloc[0]
 nex=plate["nex"].iloc[0]
 nin=plate["nin"].iloc[0]
#bolt
 bolt_caliber_data = pd.read_excel(steel,sheet_name="bolt_caliber")
 d = general["bolt_d_mm"].iloc[0]
 selected_bolt = bolt_caliber_data[bolt_caliber_data["d_mm"] == d]
 As = selected_bolt["As_mm2"].iloc[0]
 dm = selected_bolt["dm_mm"].iloc[0]
 class_x = general["class_x"].iloc[0]
 fub=class_x*100 #N/mm2
#foundation concrete taus [N/mm2]
 fck=general["fck_N_mm2"].iloc[0]
 if fck==20: taus=1.1 
 elif fck==25: taus=1.2 
 elif fck==30: taus=1.3 
 else: raise ValueError("no corresponding Ecm value for fck outside 20 25 30 ") 
# anchor attributes
 H = general["H"].iloc[0] #depth
 D = general["D"].iloc[0] #diameter
 l2 = general["l2"].iloc[0] #hook length
# actions
 Ntsd=general["Ntsd_KN"].iloc[0] #KN applicated compression force
#start
#distances
 kin1=(l-tw)/2
 ein=(bp-l)/2
 m_exter=kex-0.8*af*2**0.5 
 m_inter1=kin1-0.8*aw*2**0.5 
#
 Ftrd=0.9*As*fub/1.5
 Bplrd=0.6*3.14*dm*tp*fu/1.5
 Btrd=min(Ftrd,Bplrd)/1000
#case1 external case2 internal
 Fsrd=3.14*d*taus*(H+2.7*D+4*d/1000+3.5*l2) #KN force on a single anchor bolt
 leff1,Mplrd1,Ftrd_3modes1,Ftrdc1,Ftrd_bolts1,Ftrd_set1=bolt_set(set1,eex,m_exter,nex,bp,p,tp,
fy,Btrd,Fsrd,tf)
 leff2,Mplrd2,Ftrd_3modes2,Ftrdc2,Ftrd_bolts2,Ftrd_set2=bolt_set(set2,ein,m_inter1,nin,bp,p,tp,
fy,Btrd,Fsrd,tw)
 if set1=="external" and set2=="none":
  Ftrd_total=Ftrd_set1
 elif set1=="none" and set2=="internal":
  Ftrd_total=Ftrd_set2
 elif set1=="external" and set2=="internal":
  Ftrd_total=Ftrd_set1+Ftrd_set2
 else: raise ValueError("there has to be at least one set of bolts")
 if Ntsd==0:
  Ntrd=Ftrd_total
  print("one part of bolts is receiving tension therefor the other part is receiving compression") 
 elif Ntsd>=0:
  Ntrd=2*Ftrd_total
  print("all bolts are receiving tension")
  if Ntsd<=Ntrd:
   print("Ntsd<=Ntrd bolts lifting resistance has been aquired")
  else:
   print("Ntsd>Ntrd bolts lifting resistance has failed")
 else: raise ValueError("Ntsd has to be positive")
 T=pd.DataFrame({"leff1":leff1,"Mplrd1":Mplrd1,"Ftrd_3modes1":Ftrd_3modes1,"Ftrdc1":Ftrdc1,
 "Ftrd_bolts1":Ftrd_bolts1,"Ftrd_set1":Ftrd_set1,"leff2":leff2,"Mplrd2":Mplrd2,
 "Ftrd_3modes2":Ftrd_3modes2,"Ftrdc2":Ftrdc2,"Ftrd_bolts2":Ftrd_bolts2,
 "Ftrd_set2":Ftrd_set2,"Ftrd_total":Ftrd_total,"Ntsd":Ntsd,"Ntrd":Ntrd},index=[0])
 return T

def bolt_set(set,e,m,ns,bp,p,tp,fy,Btrd,Fsrd,t):
 if set=="external":
  leff=min(bp,4*m+1.25*e+(ns-1)*p,ns*(4*m+1.25*e),ns*2*3.14*m)*(1/2) #mm
 elif set=="internal":
  leff=min(4*m+1.25*e,2*3.14*m) #mm
 elif set=="none":
  leff=0 
 else: raise ValueError("unidentified bolts set type")
 Mplrd=0.25*leff*tp**2*fy/1.1/1000 #KN.mm 
 if set=="none":
  n,Ftrd1,Ftrd2,Ftrd3,Ftrd_3modes,Ftrdc,Ftrd_bolts,Ftrd=0,0,0,0,0,0,0,0
 else:
  n=min(e,1.25*m)
  Ftrd1=4*Mplrd/m
  Ftrd3=ns*Btrd
  Ftrd2=(2*Mplrd+n*Ftrd3)/(n+m)
  Ftrd_3modes=min(Ftrd1,Ftrd2,Ftrd3)
  Ftrdc=leff*t*fy/1.1/1000 # t for flange in external and web for internal  
  Ftrd_bolts=Fsrd*ns
  Ftrd=min(Ftrd_3modes,Ftrdc,Ftrd_bolts)
 return leff,Mplrd,Ftrd_3modes,Ftrdc,Ftrd_bolts,Ftrd 

def shear_force_tension(Ntsd,n,Vsd,As,fub,A):
 Ftsd=Ntsd/n
 Fvsd=Vsd/n
 Ftrd=0.9*As*fub/1.25
 Fvrd=0.6*fub*A/1.25
 result=Ftsd/(1.4*Ftrd)+Fvsd/Fvrd
 if result<=1:
  print("column base can resist")
 else:print("column base cannot resist")
 return result

def compressed_zone(Ftrd,Ftrd1,Ftrd2,input,steel):
#loading input tables
 general = pd.read_excel(input,sheet_name="general")
 plate = pd.read_excel(input,sheet_name="plate")
#loading input data
#actions
 Msd=general["Msd"].iloc[0] #KN.m
#column used 
 column_type = general["column_type"].iloc[0]
 column_data = pd.read_excel(steel,sheet_name=column_type)
 column_class = general["column_class"].iloc[0]
 selected_column = column_data[column_data[column_type] == column_class]
 h=selected_column["h"].iloc[0]*10
 b=selected_column["b"].iloc[0]*10
 tf=selected_column["tf"].iloc[0]*10
 tw=selected_column["tw"].iloc[0]*10
#foundation concrete
 fck=general["fck_N_mm2"].iloc[0]
#steel plate
 tp=plate["tp_mm"].iloc[0]
 steel_grade=plate["steel_grade"].iloc[0] 
 steel_grade_data = pd.read_excel(steel,sheet_name="grade")
 steel_grade_attributes = steel_grade_data[steel_grade_data["steel_grade"] == steel_grade]
 fy=steel_grade_attributes["fy_N_mm2"].iloc[0]
 bp=plate["bp_mm"].iloc[0]
 kex=plate["kex"].iloc[0] 
 kin2=plate["kin2"].iloc[0]  
#start
 kj=1
 fj=2/3*kj*fck/1.5
 c=tp*(fy/(3*fj*1.1))**0.5
 if b+2*c<bp:
  bc=b+2*c
 elif b+2*c>=bp:
  bc=bp
 else: raise ValueError("b+2*c has to be positive")
 hc=Ftrd*1000/(bc*fj)
 if hc<= tf+2*c:
  yg=hc/2
  a1,b1,a2,b2,s1,s2,y1,y2=0,0,0,0,0,0,0,0 
 elif hc> tf+2*c:
  a1=tf+2*c
  b1=bc
  a2=(Ftrd*1000/fj-(2*c+tf)*bc)*(1/(2*c+tw))
  b2=tw+2*c
  s1=a1*b1
  s2=a2*b2
  y1=a1/2
  y2=a2/2+a1
  yg=(s1*y1+s2*y2)/(s1+s2)
 else: raise ValueError("hc has to be positive")
 lever_set1=kex+h+c-yg
 lever_set2=h-kin2+c-yg
 if Ftrd1>0 and Ftrd2==0:
  h1=lever_set1
  h2=0  
 elif Ftrd1==0 and Ftrd2>0:
  h1=0
  h2=lever_set2
 elif Ftrd1>0 and Ftrd2>0:
  h1=lever_set1
  h2=lever_set2  
 else: raise ValueError("")
 Mrd=(Ftrd1*h1+Ftrd2*h2)/1000
 return fj,c,bc,hc,yg,h1,h2,a1,b1,a2,b2,s1,s2,y1,y2,Msd,Mrd

def column_base(input,steel):
 general = pd.read_excel(input,sheet_name="general")
 Ncsd=general["Ncsd_KN"].iloc[0]
 Ntsd=general["Ntsd_KN"].iloc[0]
 Vsd=general["Vsd_KN"].iloc[0]
 Msd=general["Msd"].iloc[0]
 if Ncsd>0 and Vsd>0:
  print("case1 compression Ncsd and shear force Vsd ")
  Tables,sheetNames,xlsx_name=case1(input)
 elif Ntsd>0 and Vsd>0:
  print("case2 tension Ntsd and shear force Vsd ")
  Tables,sheetNames,xlsx_name=case2(input)
 elif Ncsd>0 and Msd>0:
  print("case3 compression Ncsd and bending moment Msd we verify that (Ncsd/Ncrd+Msd/Mrd)<=1")
  Tables,sheetNames,xlsx_name=case3(input)
 elif Ntsd>0 and Msd>0:
  print("case4 tension Ntsd and bending moment Msd we verify that (Ntsd/Ntrd+Msd/Mrd)<=1")
  Tables,sheetNames,xlsx_name=case4(input)
 else : raise ValueError("unidentified case due to lack of primary actions inputs")
 return Tables,sheetNames,xlsx_name

def case1(input,steel):
# Ncsd Vsd KN
#1compression stage Ncsd effect
 T1=compression(input,steel)
#2shear force stage Vsd
 T2=combined_action(input,steel)
 Tables = [T1,T2]
 sheetNames = ["compression stage Ncsd effect", "shear force stage Vsd"]
 xlsx_name = "column_base_case1.xlsx"
 return Tables,sheetNames,xlsx_name

def case2(input):
# Ntsd Vsd KN
#1-lifting
 (leff1,Mplrd1,Ftrd_3modes1,Ftrdc1,Ftrd_bolts1,Ftrd_set1,leff2,Mplrd2,Ftrd_3modes2,Ftrdc2,
 Ftrd_bolts2,Ftrd_set2,Ftrd_total,Ntrd)=tension(input)
 T1=pd.DataFrame({"leff1":leff1,"Mplrd1":Mplrd1,"Ftrd_3modes1":Ftrd_3modes1,"Ftrdc1":Ftrdc1,
 "Ftrd_bolts1":Ftrd_bolts1,"Ftrd_set1":Ftrd_set1,"leff2":leff2,"Mplrd2":Mplrd2,
 "Ftrd_3modes2":Ftrd_3modes2,"Ftrdc2":Ftrdc2,"Ftrd_bolts2":Ftrd_bolts2,
 "Ftrd_set2":Ftrd_set2,"Ftrd_total":Ftrd_total,"Ntrd":Ntrd},index=[0])
#2-verifying combined actions
 (n,Fvsd,alpha,Fvrd,Fbrd,Fbcrd,Frd,Ntsd,Ftsd,Ftrd,result)=combined_action(input)
 T2=pd.DataFrame({"n":n,"Fvsd":Fvsd,"alpha":alpha,"Fvrd":Fvrd,"Fbrd":Fbrd,"Fbcrd":Fbcrd,
 "Frd":Frd,"Ntsd":Ntsd,"Ftsd":Ftsd,"Ftrd":Ftrd,"result":result},index=[0])
 Tables = [T1,T2]
 sheetNames = ["lifting", "combined actions"]
 xlsx_name = "column_base_case2.xlsx"
 return Tables,sheetNames,xlsx_name

def case3(input):
#Ncsd KN Msd KN.m
#1-compression stage Ncsd effect we look for Ncrd
 T1=compression(input)
#2-bending moment effect Msd we look for Mrd    
 #tension zone 
 T2=tension(input)
 Ncsd=T2["Ncsd"].iloc[0]
 Ncrd=T2["Ncrd"].iloc[0]
 Ftrd_total=T2["Ftrd_total"].iloc[0]
 Ftrd_set1=T2["Ftrd_set1"].iloc[0]
 Ftrd_set2=T2["Ftrd_set2"].iloc[0]
 #compression zone
 fj,c,bc,hc,yg,h1,h2,a1,b1,a2,b2,s1,s2,y1,y2,Msd,Mrd=compressed_zone(Ftrd_total,Ftrd_set1,Ftrd_set2,input) 
 result=Ncsd/Ncrd+Msd/Mrd
 T3=pd.DataFrame({"fj":fj,"c":c,"bc":bc,"hc":hc,"yg":yg,"h1":h1,"h2":h2,"a1":a1,"b1":b1,
  "a2":a2,"b2":b2,"s1":s1,"s2":s2,"y1":y1,"y2":y2,"Msd":Msd,"Mrd":Mrd,"result":result},index=[0]) 
 if result<=1:print("compression Ncsd and bending moment Msd resistance has been aquired")
 else:print("compression Ncsd and bending moment Msd resistance has failed")
 Tables = [T1,T2,T3]
 sheetNames = ["compression stage Ncsd effect", "Msd effect tension zone","Msd effect compression zone"]
 xlsx_name = "column_base_case3.xlsx"
 return Tables,sheetNames,xlsx_name

def case4(input):
#Ntsd KN Msd KN.m
#1-bending moment effect Msd we look for Mrd    
 #tension zone 
 (leff1,Mplrd1,Ftrd_3modes1,Ftrdc1,Ftrd_bolts1,Ftrd_set1,leff2,Mplrd2,Ftrd_3modes2,Ftrdc2,
 Ftrd_bolts2,Ftrd_set2,Ftrd_total,Ntsd,Ntrd)=tension(input)
 T1=pd.DataFrame({"leff1":leff1,"Mplrd1":Mplrd1,"Ftrd_3modes1":Ftrd_3modes1,"Ftrdc1":Ftrdc1,
 "Ftrd_bolts1":Ftrd_bolts1,"Ftrd_set1":Ftrd_set1,"leff2":leff2,"Mplrd2":Mplrd2,
 "Ftrd_3modes2":Ftrd_3modes2,"Ftrdc2":Ftrdc2,"Ftrd_bolts2":Ftrd_bolts2,"Ftrd_set2":Ftrd_set2,
 "Ntrd":Ntrd,},index=[0])
 #compression zone
 fj,c,bc,hc,yg,h1,h2,a1,b1,a2,b2,s1,s2,y1,y2,Msd,Mrd=compressed_zone(Ftrd_total,Ftrd_set1,Ftrd_set2,input) 
 result=Ntsd/Ntrd+Msd/Mrd
 T2=pd.DataFrame({"fj":fj,"c":c,"bc":bc,"hc":hc,"yg":yg,"h1":h1,"h2":h2,"a1":a1,"b1":b1,
  "a2":a2,"b2":b2,"s1":s1,"s2":s2,"y1":y1,"y2":y2,"Msd":Msd,"Mrd":Mrd,"result":result},index=[0]) 
 if result<=1:print("tension Ntsd and bending moment Msd resistance has been aquired")
 else:print("tension Ntsd and bending moment Msd resistance has failed")
 Tables = [T1,T2]
 sheetNames = ["tension zone", "compression zone"]
 xlsx_name = "column_base_case4.xlsx"
 return Tables,sheetNames,xlsx_name

