import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
import os
from config.definitions import ROOT_DIR
from PIL import Image
import matplotlib.pyplot as plt

#Load models and scalers
#Circular columns
CC_GBR=joblib.load(os.path.join(ROOT_DIR,'CFST_Circ_Columns_GBR.joblib'))
CC_GBR_sc=pickle.load(open(os.path.join(ROOT_DIR,'CFST_Circ_Columns_GBR.pkl'),'rb'))

#Circular beam-columns
CBC_XB=joblib.load(os.path.join(ROOT_DIR,'CFST_Circ_Beam_Columns_XGBoost_1p5p0.joblib'))
CBC_XB_sc=pickle.load(open(os.path.join(ROOT_DIR,'CFST_Circ_Beam_Columns_XGBoost_1p5p0.pkl'),'rb'))

#Rectangular columns
RC_CB=joblib.load(os.path.join(ROOT_DIR,'CFST_Rect_Columns_CatBoost_R1.joblib'))
RC_CB_sc=pickle.load(open(os.path.join(ROOT_DIR,'CFST_Rect_Columns_CatBoost_R1.pkl'),'rb'))

#Rectangular beam-columns
RBC_CB=joblib.load(os.path.join(ROOT_DIR,'CFST_Rect_Beam_Columns_CatBoost_R1.joblib'))
RBC_CB_sc=pickle.load(open(os.path.join(ROOT_DIR,'CFST_Rect_Beam_Columns_CatBoost_R1.pkl'),'rb'))

#Resistance factors
phi_CC_GBR=0.85
phi_CBC_XB=0.75
phi_RC_CB=0.85
phi_RBC_CB=0.85

st.header('Resistance of Concrete-Filled Steel Tubular (CFST) Columns Predicted by Machine Learning Models')

st.sidebar.header('User Input Parameters')

column_type=st.sidebar.radio("Column Type",('Circular Column','Circular Beam-Column','Rectangular Column','Rectangular Beam-Column'))

if column_type=='Circular Column':
    fc_CC_min=20
    fc_CC_max=110
    fc=st.sidebar.slider("f'c (MPa)",min_value=fc_CC_min, max_value=fc_CC_max, step=10)
    fy_CC_min=250
    fy_CC_max=math.floor(min([450,0.7*200*(fc+8)**0.31])/50)*50
    fy=st.sidebar.slider("fy (MPa)",min_value=fy_CC_min, max_value=fy_CC_max,step=50)
    t_CC_min=1.5
    t_CC_max=10.0   
    t=st.sidebar.slider("t (mm)",min_value=t_CC_min, max_value=t_CC_max, step=0.5)
    d_CC_min=math.ceil(max([200000*0.03*t/fy,60.0])/10)*10.0
    d_CC_max=math.floor(min([200000*0.17*t/fy,320.0])/10)*10.0
    d=st.sidebar.slider("D (mm)",min_value=d_CC_min, max_value=d_CC_max, step=10.0)
    l_CC_min=math.ceil(max([2*d,500.0])/100)*100.0
    l_CC_max=math.floor(min([22*d,3400.0])/100)*100.0
    l=st.sidebar.slider("L (mm)",min_value=l_CC_min, max_value=l_CC_max, step=100.0)
elif column_type=='Circular Beam-Column':
    fc_CBC_min=20
    fc_CBC_max=100
    fc=st.sidebar.slider("f'c (MPa)",min_value=fc_CBC_min, max_value=fc_CBC_max, step=10)
    fy_CBC_min=250
    fy_CBC_max=math.floor(min([450,0.7*200*(fc+8)**0.31])/50)*50
    fy=st.sidebar.slider("fy (MPa)",min_value=fy_CBC_min, max_value=fy_CBC_max,step=50)
    t_CBC_min=2.0
    t_CBC_max=7.5
    t=st.sidebar.slider("t (mm)",min_value=t_CBC_min, max_value=t_CBC_max, step=0.5)
    d_CBC_min=math.ceil(max([200000*0.03*t/fy,80.0])/10)*10.0
    d_CBC_max=math.floor(min([200000*0.13*t/fy,220.0])/10)*10.0
    d=st.sidebar.slider("D (mm)",min_value=d_CBC_min, max_value=d_CBC_max, step=10.0)
    l_CBC_min=math.ceil(max([4*d,500.0])/100)*100.0
    l_CBC_max=math.floor(min([29*d,3700.0])/100)*100.0
    l=st.sidebar.slider("L (mm)",min_value=l_CBC_min, max_value=l_CBC_max, step=100.0)
    e_CBC_min=math.ceil(0.05*d/5.0)*5.0
    e_CBC_max=math.floor(0.60*d/5.0)*5.0
    e=st.sidebar.slider("e (mm)",min_value=e_CBC_min, max_value=e_CBC_max, step=5.0)
elif column_type=='Rectangular Column':
    fc_RC_min=20
    fc_RC_max=120
    fc=st.sidebar.slider("f'c (MPa)",min_value=fc_RC_min, max_value=fc_RC_max, step=10)
    fy_RC_min=250
    fy_RC_max=math.floor(min([750,0.7*200*(fc+8)**0.31])/50)*50
    fy=st.sidebar.slider("fy (MPa)",min_value=fy_RC_min, max_value=fy_RC_max,step=50)
    t_RC_min=2.0
    t_RC_max=8.0
    t=st.sidebar.slider("t (mm)",min_value=t_RC_min, max_value=t_RC_max, step=0.5)
    b_RC_min=math.ceil(max([(200000**0.5)*0.65*t/(fy**0.5),80.0])/10)*10.0
    b_RC_max=math.floor(min([(200000**0.5)*3.3*t/(fy**0.5),250.0])/10)*10.0
    b=st.sidebar.slider("B (mm)",min_value=b_RC_min, max_value=b_RC_max, step=10.0)
    h_RC_min=100
    h_RC_max=math.floor(min([1.67*b,250.0])/10)*10.0
    h=st.sidebar.slider("H (mm)",min_value=h_RC_min, max_value=h_RC_max, step=10.0)
    l_RC_min=math.ceil(max([3*b,500.0])/100)*100.0
    l_RC_max=math.floor(min([25*b,2700.0])/100)*100.0
    l=st.sidebar.slider("L (mm)",min_value=l_RC_min, max_value=l_RC_max, step=100.0)
elif column_type=='Rectangular Beam-Column':
    fc_RBC_min=20
    fc_RBC_max=100
    fc=st.sidebar.slider("f'c (MPa)",min_value=fc_RBC_min, max_value=fc_RBC_max, step=10)
    fy_RBC_min=250
    fy_RBC_max=math.floor(min([750,0.7*200*(fc+8)**0.31])/50)*50
    fy=st.sidebar.slider("fy (MPa)",min_value=fy_RBC_min, max_value=fy_RBC_max,step=50)
    t_RBC_min=3.0
    t_RBC_max=8.0
    t=st.sidebar.slider("t (mm)",min_value=t_RBC_min, max_value=t_RBC_max, step=0.5)
    b_RBC_min=math.ceil(max([(200000**0.5)*0.72*t/(fy**0.5),100.0])/10)*10.0
    b_RBC_max=math.floor(min([(200000**0.5)*2.67*t/(fy**0.5),220.0])/10)*10.0
    b=st.sidebar.slider("B (mm)",min_value=b_RBC_min, max_value=b_RBC_max, step=10.0)
    h_RBC_min=100
    h_RBC_max=math.floor(min([1.5*b,220.0])/10)*10.0
    h=st.sidebar.slider("H (mm)",min_value=h_RBC_min, max_value=h_RBC_max, step=10.0)
    l_RBC_min=math.ceil(max([3*b,500.0])/100)*100.0
    l_RBC_max=math.floor(min([31*b,3600.0])/100)*100.0
    l=st.sidebar.slider("L (mm)",min_value=l_RBC_min, max_value=l_RBC_max, step=100.0)
    e_RBC_min=math.ceil(0.05*h/5.0)*5.0
    e_RBC_max=math.floor(0.80*h/5.0)*5.0
    e=st.sidebar.slider("e (mm)",min_value=e_RBC_min, max_value=e_RBC_max, step=5.0)
    
if column_type=='Circular Column':
    data = {"Column Type": column_type, "D (mm)": "{:.1f}".format(d), "t (mm)": "{:.2f}".format(t), "L (mm)": "{:.0f}".format(l), "fy (MPa)": fy, "f'c (MPa)": fc}
elif column_type=='Circular Beam-Column':
    data = {"Column Type": column_type, "D (mm)": "{:.1f}".format(d), "t (mm)": "{:.2f}".format(t), "L (mm)": "{:.0f}".format(l), "e (mm)": "{:.1f}".format(e), "fy (MPa)": fy, "f'c (MPa)": fc}
elif column_type=='Rectangular Column':
    data = {"Column Type": column_type, "B (mm)": "{:.1f}".format(b), "H (mm)": "{:.1f}".format(h), "t (mm)": "{:.2f}".format(t), "L (mm)": "{:.0f}".format(l), "fy (MPa)": fy, "f'c (MPa)": fc}
elif column_type=='Rectangular Beam-Column':
    data = {"Column Type": column_type, "B (mm)": "{:.1f}".format(b), "H (mm)": "{:.1f}".format(h), "t (mm)": "{:.2f}".format(t), "L (mm)": "{:.0f}".format(l), "e (mm)": "{:.1f}".format(e), "fy (MPa)": fy, "f'c (MPa)": fc} 

st.subheader('Input Parameters')

input_parameters_df=pd.DataFrame(data, index=[0]) 

hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

st.dataframe(input_parameters_df)

if column_type=='Circular Column':
    X_ML_CC=np.array([[d,t,l,fy,fc]])
    X_ML_CC_GBR=CC_GBR_sc.transform(X_ML_CC)
    Nn_CC_GBR=CC_GBR.predict(X_ML_CC_GBR).item()
    Nd_CC_GBR=Nn_CC_GBR*phi_CC_GBR
    
elif column_type=='Circular Beam-Column':
    X_ML_CBC=np.array([[d,t,l,fy,fc,e,e]])
    X_ML_CBC_XB=CBC_XB_sc.transform(X_ML_CBC)
    Nn_CBC_XB=CBC_XB.predict(X_ML_CBC_XB).item()
    Nd_CBC_XB=Nn_CBC_XB*phi_CBC_XB
    
elif column_type=='Rectangular Column':
    X_ML_RC=np.array([[b,h,t,l,fy,fc]])
    X_ML_RC_CB=RC_CB_sc.transform(X_ML_RC)
    Nn_RC_CB=RC_CB.predict(X_ML_RC_CB).item()
    Nd_RC_CB=Nn_RC_CB*phi_RC_CB
     
elif column_type=='Rectangular Beam-Column':
    X_ML_RBC=np.array([[b,h,t,l,fy,fc,e,e]])
    X_ML_RBC_CB=RBC_CB_sc.transform(X_ML_RBC)
    Nn_RBC_CB=RBC_CB.predict(X_ML_RBC_CB).item()
    Nd_RBC_CB=Nn_RBC_CB*phi_RBC_CB
    
st.subheader('Nominal (Nn) and Design (Nd) Resistances (kN)')
    
if column_type=='Circular Column': 
    if Nn_CC_GBR>0: N={'GBR, Nn': "{:.2f}".format(Nn_CC_GBR), 'GBR, Nd': "{:.2f}".format(Nd_CC_GBR)}
    else: N={'GBR, Nn': 'NG', 'GBR, Nd': 'NG'}    
        
elif column_type=='Circular Beam-Column':
    if Nn_CBC_XB>0: N={'XGBoost, Nn': "{:.2f}".format(Nn_CBC_XB),'XGBoost, Nd': "{:.2f}".format(Nd_CBC_XB)}
    else: N={'XGBoost, Nn': 'NG', 'XGBoost, Nd': 'NG'}

elif column_type=='Rectangular Column':
    if Nn_RC_CB>0: N={'CatBoost, Nn': "{:.2f}".format(Nn_RC_CB), 'CatBoost, Nd': "{:.2f}".format(Nd_RC_CB)}
    else: N={'CatBoost, Nn': 'NG', 'CatBoost, Nd': 'NG'}
   
elif column_type=='Rectangular Beam-Column':
    if Nn_RBC_CB>0: N={'CatBoost, Nn': "{:.2f}".format(Nn_RBC_CB), 'CatBoost, Nd': "{:.2f}".format(Nd_RBC_CB)}
    else: N={'CatBoost, Nn': 'NG', 'CatBoost, Nd': 'NG'}
    
N_df=pd.DataFrame(N, index=[0])
st.dataframe(N_df)

st.subheader('Resistance Plots as Functions of Design Variables')

if column_type=='Circular Column':
    fy1=np.arange(fy_CC_min,fy_CC_max+1,50)
    fy1=fy1.reshape(len(fy1),1)
    fc1=np.full((len(fy1),1),fc)
    d1=np.full((len(fy1),1),d)
    t1=np.full((len(fy1),1),t)
    l1=np.full((len(fy1),1),l)
    X_ML_CC_1=np.concatenate((d1, t1, l1, fy1, fc1), axis=1)
    X_ML_CC_GBR_1=CC_GBR_sc.transform(X_ML_CC_1)
    Nn_CC_GBR_1=CC_GBR.predict(X_ML_CC_GBR_1)
    Nd_CC_GBR_1=Nn_CC_GBR_1*phi_CC_GBR
   
    fc2=np.arange(fc_CC_min,fc_CC_max+1,10)
    fc2=fc2.reshape(len(fc2),1)
    fy2=np.full((len(fc2),1),fy)
    d2=np.full((len(fc2),1),d)
    t2=np.full((len(fc2),1),t)
    l2=np.full((len(fc2),1),l)   
    X_ML_CC_2=np.concatenate((d2, t2, l2, fy2, fc2), axis=1)
    X_ML_CC_GBR_2=CC_GBR_sc.transform(X_ML_CC_2)
    Nn_CC_GBR_2=CC_GBR.predict(X_ML_CC_GBR_2)
    Nd_CC_GBR_2=Nn_CC_GBR_2*phi_CC_GBR 

    d3=np.arange(d_CC_min,d_CC_max+1,10.0)
    d3=d3.reshape(len(d3),1)
    fy3=np.full((len(d3),1),fy)    
    fc3=np.full((len(d3),1),fc)
    t3=np.full((len(d3),1),t)
    l3=np.full((len(d3),1),l)
    X_ML_CC_3=np.concatenate((d3, t3, l3, fy3, fc3), axis=1)
    X_ML_CC_GBR_3=CC_GBR_sc.transform(X_ML_CC_3)
    Nn_CC_GBR_3=CC_GBR.predict(X_ML_CC_GBR_3)
    Nd_CC_GBR_3=Nn_CC_GBR_3*phi_CC_GBR    
    
    t4=np.arange(t_CC_min,t_CC_max+0.1,0.5)
    t4=t4.reshape(len(t4),1)
    fy4=np.full((len(t4),1),fy)   
    fc4=np.full((len(t4),1),fc)
    d4=np.full((len(t4),1),d)
    l4=np.full((len(t4),1),l)
    X_ML_CC_4=np.concatenate((d4, t4, l4, fy4, fc4), axis=1)
    X_ML_CC_GBR_4=CC_GBR_sc.transform(X_ML_CC_4)
    Nn_CC_GBR_4=CC_GBR.predict(X_ML_CC_GBR_4)
    Nd_CC_GBR_4=Nn_CC_GBR_4*phi_CC_GBR    

    l5=np.arange(l_CC_min,l_CC_max+1,100.0)
    l5=l5.reshape(len(l5),1)
    fy5=np.full((len(l5),1),fy)
    fc5=np.full((len(l5),1),fc)
    d5=np.full((len(l5),1),d)
    t5=np.full((len(l5),1),t)
    X_ML_CC_5=np.concatenate((d5, t5, l5, fy5, fc5), axis=1)
    X_ML_CC_GBR_5=CC_GBR_sc.transform(X_ML_CC_5)
    Nn_CC_GBR_5=CC_GBR.predict(X_ML_CC_GBR_5)
    Nd_CC_GBR_5=Nn_CC_GBR_5*phi_CC_GBR    
    
    f1 = plt.figure(figsize=(6.75,4*3/2), dpi=200)

    ax1 = f1.add_subplot(3,2,2)
    ax1.plot(fy1, Nn_CC_GBR_1, color='#e31a1c',linewidth=1.5, label='GBR, Nn',linestyle='solid')
    ax1.plot(fy1, Nd_CC_GBR_1, color='#0070C0',linewidth=1.5, label='GBR, Nd',linestyle='solid')
    fy_loc=np.where(fy1==fy)[0].item()
    ax1.scatter(fy,Nn_CC_GBR_1[fy_loc],marker='o',facecolors='#e31a1c')
    ax1.scatter(fy,Nd_CC_GBR_1[fy_loc],marker='o',facecolors='#0070C0')
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('Resistance (kN)', fontsize=10)
    ax1.set_xlabel('fy (MPa)', fontsize=10)
    
    ax2 = f1.add_subplot(3,2,1)
    ax2.plot(fc2, Nn_CC_GBR_2, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax2.plot(fc2, Nd_CC_GBR_2, color='#0070C0',linewidth=1.5, linestyle='solid')
    fc_loc=np.where(fc2==fc)[0].item()
    ax2.scatter(fc,Nn_CC_GBR_2[fc_loc],marker='o',facecolors='#e31a1c')
    ax2.scatter(fc,Nd_CC_GBR_2[fc_loc],marker='o',facecolors='#0070C0')
    ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax2.set_ylabel('Resistance (kN)', fontsize=10)
    ax2.set_xlabel("f'c (MPa)", fontsize=10)
    
    ax3 = f1.add_subplot(3,2,4)
    ax3.plot(d3, Nn_CC_GBR_3, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax3.plot(d3, Nd_CC_GBR_3, color='#0070C0',linewidth=1.5, linestyle='solid')
    d_loc=np.where(d3==d)[0].item()
    ax3.scatter(d,Nn_CC_GBR_3[d_loc],marker='o',facecolors='#e31a1c')
    ax3.scatter(d,Nd_CC_GBR_3[d_loc],marker='o',facecolors='#0070C0')
    ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax3.set_ylabel('Resistance (kN)', fontsize=10)
    ax3.set_xlabel('D (mm)', fontsize=10)
    
    ax4 = f1.add_subplot(3,2,3)
    ax4.plot(t4, Nn_CC_GBR_4, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax4.plot(t4, Nd_CC_GBR_4, color='#0070C0',linewidth=1.5, linestyle='solid')
    t_loc=np.where(t4==t)[0].item()
    ax4.scatter(t,Nn_CC_GBR_4[t_loc],marker='o',facecolors='#e31a1c')
    ax4.scatter(t,Nd_CC_GBR_4[t_loc],marker='o',facecolors='#0070C0')
    ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax4.set_ylabel('Resistance (kN)', fontsize=10)
    ax4.set_xlabel('t (mm)', fontsize=10)
    
    ax5 = f1.add_subplot(3,2,5)
    ax5.plot(l5, Nn_CC_GBR_5, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax5.plot(l5, Nd_CC_GBR_5, color='#0070C0',linewidth=1.5, linestyle='solid')
    l_loc=np.where(l5==l)[0].item()
    ax5.scatter(l,Nn_CC_GBR_5[l_loc],marker='o',facecolors='#e31a1c')
    ax5.scatter(l,Nd_CC_GBR_5[l_loc],marker='o',facecolors='#0070C0')
    ax5.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax5.set_ylabel('Resistance (kN)', fontsize=10)
    ax5.set_xlabel('L (mm)', fontsize=10)
    
    f1.legend(ncol=2, fontsize=10, bbox_to_anchor=(0.52, -0.07), loc='lower center')
    f1.tight_layout()
    st.pyplot(f1)
    
    st.subheader('Nomenclature')
    st.write("D is the outside diameter of circular column cross section; GBR is gradient boosting regressor; L is column length; Nn and Nd are the nominal and design resistances of columns; f'c is concrete compressive strength; fy is steel yield strength; t is tube wall thickness.")

elif column_type=='Circular Beam-Column':
    fy1=np.arange(fy_CBC_min,fy_CBC_max+1,50)
    fy1=fy1.reshape(len(fy1),1)
    fc1=np.full((len(fy1),1),fc)
    d1=np.full((len(fy1),1),d)
    t1=np.full((len(fy1),1),t)
    l1=np.full((len(fy1),1),l)
    e1=np.full((len(fy1),1),e)
    X_ML_CBC_1=np.concatenate((d1, t1, l1, fy1, fc1, e1, e1), axis=1)
    X_ML_CBC_XB_1=CBC_XB_sc.transform(X_ML_CBC_1)
    Nn_CBC_XB_1=CBC_XB.predict(X_ML_CBC_XB_1)
    Nd_CBC_XB_1=Nn_CBC_XB_1*phi_CBC_XB

    fc2=np.arange(fc_CBC_min,fc_CBC_max+1,10)
    fc2=fc2.reshape(len(fc2),1)
    fy2=np.full((len(fc2),1),fy)
    d2=np.full((len(fc2),1),d)
    t2=np.full((len(fc2),1),t)
    l2=np.full((len(fc2),1),l)
    e2=np.full((len(fc2),1),e)    
    X_ML_CBC_2=np.concatenate((d2, t2, l2, fy2, fc2, e2, e2), axis=1)
    X_ML_CBC_XB_2=CBC_XB_sc.transform(X_ML_CBC_2)
    Nn_CBC_XB_2=CBC_XB.predict(X_ML_CBC_XB_2)
    Nd_CBC_XB_2=Nn_CBC_XB_2*phi_CBC_XB 

    d3=np.arange(d_CBC_min,d_CBC_max+1,10.0)
    d3=d3.reshape(len(d3),1)
    fy3=np.full((len(d3),1),fy)    
    fc3=np.full((len(d3),1),fc)
    t3=np.full((len(d3),1),t)
    l3=np.full((len(d3),1),l)
    e3=np.full((len(d3),1),e)
    X_ML_CBC_3=np.concatenate((d3, t3, l3, fy3, fc3, e3, e3), axis=1)
    X_ML_CBC_XB_3=CBC_XB_sc.transform(X_ML_CBC_3)
    Nn_CBC_XB_3=CBC_XB.predict(X_ML_CBC_XB_3)
    Nd_CBC_XB_3=Nn_CBC_XB_3*phi_CBC_XB    
    
    t4=np.arange(t_CBC_min,t_CBC_max+0.1,0.5)
    t4=t4.reshape(len(t4),1)
    fy4=np.full((len(t4),1),fy)   
    fc4=np.full((len(t4),1),fc)
    d4=np.full((len(t4),1),d)
    l4=np.full((len(t4),1),l)
    e4=np.full((len(t4),1),e)
    X_ML_CBC_4=np.concatenate((d4, t4, l4, fy4, fc4, e4, e4), axis=1)
    X_ML_CBC_XB_4=CBC_XB_sc.transform(X_ML_CBC_4)
    Nn_CBC_XB_4=CBC_XB.predict(X_ML_CBC_XB_4)
    Nd_CBC_XB_4=Nn_CBC_XB_4*phi_CBC_XB    

    l5=np.arange(l_CBC_min,l_CBC_max+1,100.0)
    l5=l5.reshape(len(l5),1)
    fy5=np.full((len(l5),1),fy)
    fc5=np.full((len(l5),1),fc)
    d5=np.full((len(l5),1),d)
    t5=np.full((len(l5),1),t)
    e5=np.full((len(l5),1),e)
    X_ML_CBC_5=np.concatenate((d5, t5, l5, fy5, fc5, e5, e5), axis=1)
    X_ML_CBC_XB_5=CBC_XB_sc.transform(X_ML_CBC_5)
    Nn_CBC_XB_5=CBC_XB.predict(X_ML_CBC_XB_5)
    Nd_CBC_XB_5=Nn_CBC_XB_5*phi_CBC_XB    

    e6=np.arange(e_CBC_min,e_CBC_max+0.1,5.0)
    e6=e6.reshape(len(e6),1)
    fy6=np.full((len(e6),1),fy)
    fc6=np.full((len(e6),1),fc)
    d6=np.full((len(e6),1),d)
    t6=np.full((len(e6),1),t)
    l6=np.full((len(e6),1),l)
    X_ML_CBC_6=np.concatenate((d6, t6, l6, fy6, fc6, e6, e6), axis=1)
    X_ML_CBC_XB_6=CBC_XB_sc.transform(X_ML_CBC_6)
    Nn_CBC_XB_6=CBC_XB.predict(X_ML_CBC_XB_6)
    Nd_CBC_XB_6=Nn_CBC_XB_6*phi_CBC_XB  

    f1 = plt.figure(figsize=(6.75,4*3/2), dpi=200)

    ax1 = f1.add_subplot(3,2,2)
    ax1.plot(fy1, Nn_CBC_XB_1, color='#e31a1c',linewidth=1.5, label='XGBoost, Nn',linestyle='solid')
    ax1.plot(fy1, Nd_CBC_XB_1, color='#0070C0',linewidth=1.5, label='XGBoost, Nd',linestyle='solid')
    fy_loc=np.where(fy1==fy)[0].item()
    ax1.scatter(fy,Nn_CBC_XB_1[fy_loc],marker='o',facecolors='#e31a1c')
    ax1.scatter(fy,Nd_CBC_XB_1[fy_loc],marker='o',facecolors='#0070C0')
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('Resistance (kN)', fontsize=10)
    ax1.set_xlabel('fy (MPa)', fontsize=10)
    
    ax2 = f1.add_subplot(3,2,1)
    ax2.plot(fc2, Nn_CBC_XB_2, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax2.plot(fc2, Nd_CBC_XB_2, color='#0070C0',linewidth=1.5, linestyle='solid')
    fc_loc=np.where(fc2==fc)[0].item()
    ax2.scatter(fc,Nn_CBC_XB_2[fc_loc],marker='o',facecolors='#e31a1c')
    ax2.scatter(fc,Nd_CBC_XB_2[fc_loc],marker='o',facecolors='#0070C0')
    ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax2.set_ylabel('Resistance (kN)', fontsize=10)
    ax2.set_xlabel("f'c (MPa)", fontsize=10)
    
    ax3 = f1.add_subplot(3,2,4)
    ax3.plot(d3, Nn_CBC_XB_3, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax3.plot(d3, Nd_CBC_XB_3, color='#0070C0',linewidth=1.5, linestyle='solid')
    d_loc=np.where(d3==d)[0].item()
    ax3.scatter(d,Nn_CBC_XB_3[d_loc],marker='o',facecolors='#e31a1c')
    ax3.scatter(d,Nd_CBC_XB_3[d_loc],marker='o',facecolors='#0070C0')
    ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax3.set_ylabel('Resistance (kN)', fontsize=10)
    ax3.set_xlabel('D (mm)', fontsize=10)
    
    ax4 = f1.add_subplot(3,2,3)
    ax4.plot(t4, Nn_CBC_XB_4, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax4.plot(t4, Nd_CBC_XB_4, color='#0070C0',linewidth=1.5, linestyle='solid')
    t_loc=np.where(t4==t)[0].item()
    ax4.scatter(t,Nn_CBC_XB_4[t_loc],marker='o',facecolors='#e31a1c')
    ax4.scatter(t,Nd_CBC_XB_4[t_loc],marker='o',facecolors='#0070C0')
    ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax4.set_ylabel('Resistance (kN)', fontsize=10)
    ax4.set_xlabel('t (mm)', fontsize=10)
    
    ax5 = f1.add_subplot(3,2,5)
    ax5.plot(l5, Nn_CBC_XB_5, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax5.plot(l5, Nd_CBC_XB_5, color='#0070C0',linewidth=1.5, linestyle='solid')
    l_loc=np.where(l5==l)[0].item()
    ax5.scatter(l,Nn_CBC_XB_5[l_loc],marker='o',facecolors='#e31a1c')
    ax5.scatter(l,Nd_CBC_XB_5[l_loc],marker='o',facecolors='#0070C0')
    ax5.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax5.set_ylabel('Resistance (kN)', fontsize=10)
    ax5.set_xlabel('L (mm)', fontsize=10)
    
    ax6 = f1.add_subplot(3,2,6)
    ax6.plot(e6, Nn_CBC_XB_6, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax6.plot(e6, Nd_CBC_XB_6, color='#0070C0',linewidth=1.5, linestyle='solid')
    e_loc=np.where(e6==e)[0].item()
    ax6.scatter(e,Nn_CBC_XB_6[e_loc],marker='o',facecolors='#e31a1c')
    ax6.scatter(e,Nd_CBC_XB_6[e_loc],marker='o',facecolors='#0070C0')
    ax6.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax6.set_ylabel('Resistance (kN)', fontsize=10)
    ax6.set_xlabel('e (mm)', fontsize=10)
    
    f1.legend(ncol=2, fontsize=10, bbox_to_anchor=(0.52, -0.07), loc='lower center')
    f1.tight_layout()
    st.pyplot(f1)
    st.subheader('Nomenclature')
    st.write("D is the outside diameter of circular column cross section; L is column length; Nn and Nd are the nominal and design resistances of beam-columns; XGBoost is extreme gradient boosting regressor; e is load eccentricity; f'c is concrete compressive strength; fy is steel yield strength; t is tube wall thickness.")    

elif column_type=='Rectangular Column':
    fy1=np.arange(fy_RC_min,fy_RC_max+1,50)
    fy1=fy1.reshape(len(fy1),1)
    fc1=np.full((len(fy1),1),fc)
    b1=np.full((len(fy1),1),b)
    h1=np.full((len(fy1),1),h)
    t1=np.full((len(fy1),1),t)
    l1=np.full((len(fy1),1),l)
    X_ML_RC_1=np.concatenate((b1, h1, t1, l1, fy1, fc1), axis=1)
    X_ML_RC_CB_1=RC_CB_sc.transform(X_ML_RC_1)
    Nn_RC_CB_1=RC_CB.predict(X_ML_RC_CB_1)
    Nd_RC_CB_1=Nn_RC_CB_1*phi_RC_CB
    
    fc2=np.arange(fc_RC_min,fc_RC_max+1,10)
    fc2=fc2.reshape(len(fc2),1)
    fy2=np.full((len(fc2),1),fy)
    b2=np.full((len(fc2),1),b)
    h2=np.full((len(fc2),1),h)
    t2=np.full((len(fc2),1),t)
    l2=np.full((len(fc2),1),l)   
    X_ML_RC_2=np.concatenate((b2, h2, t2, l2, fy2, fc2), axis=1)
    X_ML_RC_CB_2=RC_CB_sc.transform(X_ML_RC_2)
    Nn_RC_CB_2=RC_CB.predict(X_ML_RC_CB_2)
    Nd_RC_CB_2=Nn_RC_CB_2*phi_RC_CB 

    b3=np.arange(b_RC_min,b_RC_max+1,10.0)
    b3=b3.reshape(len(b3),1)
    h3=np.full((len(b3),1),h)
    fy3=np.full((len(b3),1),fy)    
    fc3=np.full((len(b3),1),fc)
    t3=np.full((len(b3),1),t)
    l3=np.full((len(b3),1),l)
    X_ML_RC_3=np.concatenate((b3, h3, t3, l3, fy3, fc3), axis=1)
    X_ML_RC_CB_3=RC_CB_sc.transform(X_ML_RC_3)
    Nn_RC_CB_3=RC_CB.predict(X_ML_RC_CB_3)
    Nd_RC_CB_3=Nn_RC_CB_3*phi_RC_CB

    h4=np.arange(h_RC_min,h_RC_max+1,10.0)
    h4=h4.reshape(len(h4),1) 
    fy4=np.full((len(h4),1),fy)
    fc4=np.full((len(h4),1),fc)
    b4=np.full((len(h4),1),b)
    t4=np.full((len(h4),1),t)
    l4=np.full((len(h4),1),l)
    X_ML_RC_4=np.concatenate((b4, h4, t4, l4, fy4, fc4), axis=1)
    X_ML_RC_CB_4=RC_CB_sc.transform(X_ML_RC_4)
    Nn_RC_CB_4=RC_CB.predict(X_ML_RC_CB_4)
    Nd_RC_CB_4=Nn_RC_CB_4*phi_RC_CB    
    
    t5=np.arange(t_RC_min,t_RC_max+0.1,0.5)
    t5=t5.reshape(len(t5),1)
    fy5=np.full((len(t5),1),fy)   
    fc5=np.full((len(t5),1),fc)
    b5=np.full((len(t5),1),b)
    h5=np.full((len(t5),1),h)
    l5=np.full((len(t5),1),l)
    X_ML_RC_5=np.concatenate((b5, h5, t5, l5, fy5, fc5), axis=1)
    X_ML_RC_CB_5=RC_CB_sc.transform(X_ML_RC_5)
    Nn_RC_CB_5=RC_CB.predict(X_ML_RC_CB_5)
    Nd_RC_CB_5=Nn_RC_CB_5*phi_RC_CB    

    l6=np.arange(l_RC_min,l_RC_max+1,100.0)
    l6=l6.reshape(len(l6),1)
    fy6=np.full((len(l6),1),fy)
    fc6=np.full((len(l6),1),fc)
    b6=np.full((len(l6),1),b)
    h6=np.full((len(l6),1),h)
    t6=np.full((len(l6),1),t)
    X_ML_RC_6=np.concatenate((b6, h6, t6, l6, fy6, fc6), axis=1)
    X_ML_RC_CB_6=RC_CB_sc.transform(X_ML_RC_6)
    Nn_RC_CB_6=RC_CB.predict(X_ML_RC_CB_6)
    Nd_RC_CB_6=Nn_RC_CB_6*phi_RC_CB    
    
    f1 = plt.figure(figsize=(6.75,4*3/2), dpi=200)

    ax1 = f1.add_subplot(3,2,2)
    ax1.plot(fy1, Nn_RC_CB_1, color='#e31a1c',linewidth=1.5, label='CatBoost, Nn',linestyle='solid')
    ax1.plot(fy1, Nd_RC_CB_1, color='#0070C0',linewidth=1.5, label='CatBoost, Nd',linestyle='solid')
    fy_loc=np.where(fy1==fy)[0].item()
    ax1.scatter(fy,Nn_RC_CB_1[fy_loc],marker='o',facecolors='#e31a1c')
    ax1.scatter(fy,Nd_RC_CB_1[fy_loc],marker='o',facecolors='#0070C0')
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('Resistance (kN)', fontsize=10)
    ax1.set_xlabel('fy (MPa)', fontsize=10)
    
    ax2 = f1.add_subplot(3,2,1)
    ax2.plot(fc2, Nn_RC_CB_2, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax2.plot(fc2, Nd_RC_CB_2, color='#0070C0',linewidth=1.5, linestyle='solid')
    fc_loc=np.where(fc2==fc)[0].item()
    ax2.scatter(fc,Nn_RC_CB_2[fc_loc],marker='o',facecolors='#e31a1c')
    ax2.scatter(fc,Nd_RC_CB_2[fc_loc],marker='o',facecolors='#0070C0')
    ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax2.set_ylabel('Resistance (kN)', fontsize=10)
    ax2.set_xlabel("f'c (MPa)", fontsize=10)
    
    ax3 = f1.add_subplot(3,2,4)
    ax3.plot(b3, Nn_RC_CB_3, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax3.plot(b3, Nd_RC_CB_3, color='#0070C0',linewidth=1.5, linestyle='solid')
    b_loc=np.where(b3==b)[0].item()
    ax3.scatter(b,Nn_RC_CB_3[b_loc],marker='o',facecolors='#e31a1c')
    ax3.scatter(b,Nd_RC_CB_3[b_loc],marker='o',facecolors='#0070C0')
    ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax3.set_ylabel('Resistance (kN)', fontsize=10)
    ax3.set_xlabel('B (mm)', fontsize=10)
    
    ax6 = f1.add_subplot(3,2,5)
    ax6.plot(h4, Nn_RC_CB_4, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax6.plot(h4, Nd_RC_CB_4, color='#0070C0',linewidth=1.5, linestyle='solid')
    h_loc=np.where(h4==h)[0].item()
    ax6.scatter(h,Nn_RC_CB_4[h_loc],marker='o',facecolors='#e31a1c')
    ax6.scatter(h,Nd_RC_CB_4[h_loc],marker='o',facecolors='#0070C0')
    ax6.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax6.set_ylabel('Resistance (kN)', fontsize=10)
    ax6.set_xlabel('H (mm)', fontsize=10)    
    
    ax4 = f1.add_subplot(3,2,3)
    ax4.plot(t5, Nn_RC_CB_5, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax4.plot(t5, Nd_RC_CB_5, color='#0070C0',linewidth=1.5, linestyle='solid')
    t_loc=np.where(t5==t)[0].item()
    ax4.scatter(t,Nn_RC_CB_5[t_loc],marker='o',facecolors='#e31a1c')
    ax4.scatter(t,Nd_RC_CB_5[t_loc],marker='o',facecolors='#0070C0')
    ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax4.set_ylabel('Resistance (kN)', fontsize=10)
    ax4.set_xlabel('t (mm)', fontsize=10)
    
    ax5 = f1.add_subplot(3,2,6)
    ax5.plot(l6, Nn_RC_CB_6, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax5.plot(l6, Nd_RC_CB_6, color='#0070C0',linewidth=1.5, linestyle='solid')
    l_loc=np.where(l6==l)[0].item()
    ax5.scatter(l,Nn_RC_CB_6[l_loc],marker='o',facecolors='#e31a1c')
    ax5.scatter(l,Nd_RC_CB_6[l_loc],marker='o',facecolors='#0070C0')
    ax5.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax5.set_ylabel('Resistance (kN)', fontsize=10)
    ax5.set_xlabel('L (mm)', fontsize=10)
    
    f1.legend(ncol=2, fontsize=10, bbox_to_anchor=(0.52, -0.07), loc='lower center')
    f1.tight_layout()
    st.pyplot(f1)
    st.subheader('Nomenclature')
    st.write("B and H are the width and height of rectangular column cross section; CatBoost is categorical gradient boosting regressor; L is column length; Nn and Nd are the nominal and design resistances of columns; f'c is concrete compressive strength; fy is steel yield strength; t is tube wall thickness.")    

elif column_type=='Rectangular Beam-Column':
    fy1=np.arange(fy_RBC_min,fy_RBC_max+1,50)
    fy1=fy1.reshape(len(fy1),1)
    fc1=np.full((len(fy1),1),fc)
    b1=np.full((len(fy1),1),b)
    h1=np.full((len(fy1),1),h)
    t1=np.full((len(fy1),1),t)
    l1=np.full((len(fy1),1),l)
    e1=np.full((len(fy1),1),e)
    X_ML_RBC_1=np.concatenate((b1, h1, t1, l1, fy1, fc1, e1, e1), axis=1)
    X_ML_RBC_CB_1=RBC_CB_sc.transform(X_ML_RBC_1)
    Nn_RBC_CB_1=RBC_CB.predict(X_ML_RBC_CB_1)
    Nd_RBC_CB_1=Nn_RBC_CB_1*phi_RBC_CB
    
    fc2=np.arange(fc_RBC_min,fc_RBC_max+1,10)
    fc2=fc2.reshape(len(fc2),1)
    fy2=np.full((len(fc2),1),fy)
    b2=np.full((len(fc2),1),b)
    h2=np.full((len(fc2),1),h)
    t2=np.full((len(fc2),1),t)
    l2=np.full((len(fc2),1),l) 
    e2=np.full((len(fc2),1),e)    
    X_ML_RBC_2=np.concatenate((b2, h2, t2, l2, fy2, fc2, e2, e2), axis=1)
    X_ML_RBC_CB_2=RBC_CB_sc.transform(X_ML_RBC_2)
    Nn_RBC_CB_2=RBC_CB.predict(X_ML_RBC_CB_2)
    Nd_RBC_CB_2=Nn_RBC_CB_2*phi_RBC_CB 

    b3=np.arange(b_RBC_min,b_RBC_max+1,10.0)
    b3=b3.reshape(len(b3),1)
    h3=np.full((len(b3),1),h)
    fy3=np.full((len(b3),1),fy)    
    fc3=np.full((len(b3),1),fc)
    t3=np.full((len(b3),1),t)
    l3=np.full((len(b3),1),l)
    e3=np.full((len(b3),1),e)
    X_ML_RBC_3=np.concatenate((b3, h3, t3, l3, fy3, fc3, e3, e3), axis=1)
    X_ML_RBC_CB_3=RBC_CB_sc.transform(X_ML_RBC_3)
    Nn_RBC_CB_3=RBC_CB.predict(X_ML_RBC_CB_3)
    Nd_RBC_CB_3=Nn_RBC_CB_3*phi_RBC_CB

    h4=np.arange(h_RBC_min,h_RBC_max+1,10.0)
    h4=h4.reshape(len(h4),1) 
    fy4=np.full((len(h4),1),fy)
    fc4=np.full((len(h4),1),fc)
    b4=np.full((len(h4),1),b)
    t4=np.full((len(h4),1),t)
    l4=np.full((len(h4),1),l)
    e4=np.full((len(h4),1),e)
    X_ML_RBC_4=np.concatenate((b4, h4, t4, l4, fy4, fc4, e4, e4), axis=1)
    X_ML_RBC_CB_4=RBC_CB_sc.transform(X_ML_RBC_4)
    Nn_RBC_CB_4=RBC_CB.predict(X_ML_RBC_CB_4)
    Nd_RBC_CB_4=Nn_RBC_CB_4*phi_RBC_CB    
    
    t5=np.arange(t_RBC_min,t_RBC_max+0.1,0.5)
    t5=t5.reshape(len(t5),1)
    fy5=np.full((len(t5),1),fy)   
    fc5=np.full((len(t5),1),fc)
    b5=np.full((len(t5),1),b)
    h5=np.full((len(t5),1),h)
    l5=np.full((len(t5),1),l)
    e5=np.full((len(t5),1),e)
    X_ML_RBC_5=np.concatenate((b5, h5, t5, l5, fy5, fc5, e5, e5), axis=1)
    X_ML_RBC_CB_5=RBC_CB_sc.transform(X_ML_RBC_5)
    Nn_RBC_CB_5=RBC_CB.predict(X_ML_RBC_CB_5)
    Nd_RBC_CB_5=Nn_RBC_CB_5*phi_RBC_CB    

    l6=np.arange(l_RBC_min,l_RBC_max+1,100.0)
    l6=l6.reshape(len(l6),1)
    fy6=np.full((len(l6),1),fy)
    fc6=np.full((len(l6),1),fc)
    b6=np.full((len(l6),1),b)
    h6=np.full((len(l6),1),h)
    t6=np.full((len(l6),1),t)
    e6=np.full((len(l6),1),e)    
    X_ML_RBC_6=np.concatenate((b6, h6, t6, l6, fy6, fc6, e6, e6), axis=1)
    X_ML_RBC_CB_6=RBC_CB_sc.transform(X_ML_RBC_6)
    Nn_RBC_CB_6=RBC_CB.predict(X_ML_RBC_CB_6)
    Nd_RBC_CB_6=Nn_RBC_CB_6*phi_RBC_CB 

    e7=np.arange(e_RBC_min,e_RBC_max+0.1,5.0)
    e7=e7.reshape(len(e7),1)
    fy7=np.full((len(e7),1),fy)   
    fc7=np.full((len(e7),1),fc)
    b7=np.full((len(e7),1),b)
    h7=np.full((len(e7),1),h)
    t7=np.full((len(e7),1),t)
    l7=np.full((len(e7),1),l)
    X_ML_RBC_7=np.concatenate((b7, h7, t7, l7, fy7, fc7, e7, e7), axis=1)
    X_ML_RBC_CB_7=RBC_CB_sc.transform(X_ML_RBC_7)
    Nn_RBC_CB_7=RBC_CB.predict(X_ML_RBC_CB_7)
    Nd_RBC_CB_7=Nn_RBC_CB_7*phi_RBC_CB     
    
    f1 = plt.figure(figsize=(6.75,4*4/2), dpi=200)

    ax1 = f1.add_subplot(4,2,2)
    ax1.plot(fy1, Nn_RBC_CB_1, color='#e31a1c',linewidth=1.5, label='CatBoost, Nn',linestyle='solid')
    ax1.plot(fy1, Nd_RBC_CB_1, color='#0070C0',linewidth=1.5, label='CatBoost, Nd',linestyle='solid')
    fy_loc=np.where(fy1==fy)[0].item()
    ax1.scatter(fy,Nn_RBC_CB_1[fy_loc],marker='o',facecolors='#e31a1c')
    ax1.scatter(fy,Nd_RBC_CB_1[fy_loc],marker='o',facecolors='#0070C0')
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('Resistance (kN)', fontsize=10)
    ax1.set_xlabel('fy (MPa)', fontsize=10)
    
    ax2 = f1.add_subplot(4,2,1)
    ax2.plot(fc2, Nn_RBC_CB_2, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax2.plot(fc2, Nd_RBC_CB_2, color='#0070C0',linewidth=1.5, linestyle='solid')
    fc_loc=np.where(fc2==fc)[0].item()
    ax2.scatter(fc,Nn_RBC_CB_2[fc_loc],marker='o',facecolors='#e31a1c')
    ax2.scatter(fc,Nd_RBC_CB_2[fc_loc],marker='o',facecolors='#0070C0')
    ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax2.set_ylabel('Resistance (kN)', fontsize=10)
    ax2.set_xlabel("f'c (MPa)", fontsize=10)
    
    ax3 = f1.add_subplot(4,2,4)
    ax3.plot(b3, Nn_RBC_CB_3, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax3.plot(b3, Nd_RBC_CB_3, color='#0070C0',linewidth=1.5, linestyle='solid')
    b_loc=np.where(b3==b)[0].item()
    ax3.scatter(b,Nn_RBC_CB_3[b_loc],marker='o',facecolors='#e31a1c')
    ax3.scatter(b,Nd_RBC_CB_3[b_loc],marker='o',facecolors='#0070C0')
    ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax3.set_ylabel('Resistance (kN)', fontsize=10)
    ax3.set_xlabel('B (mm)', fontsize=10)
    
    ax6 = f1.add_subplot(4,2,5)
    ax6.plot(h4, Nn_RBC_CB_4, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax6.plot(h4, Nd_RBC_CB_4, color='#0070C0',linewidth=1.5, linestyle='solid')
    h_loc=np.where(h4==h)[0].item()
    ax6.scatter(h,Nn_RBC_CB_4[h_loc],marker='o',facecolors='#e31a1c')
    ax6.scatter(h,Nd_RBC_CB_4[h_loc],marker='o',facecolors='#0070C0')
    ax6.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax6.set_ylabel('Resistance (kN)', fontsize=10)
    ax6.set_xlabel('H (mm)', fontsize=10)    
    
    ax4 = f1.add_subplot(4,2,3)
    ax4.plot(t5, Nn_RBC_CB_5, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax4.plot(t5, Nd_RBC_CB_5, color='#0070C0',linewidth=1.5, linestyle='solid')
    t_loc=np.where(t5==t)[0].item()
    ax4.scatter(t,Nn_RBC_CB_5[t_loc],marker='o',facecolors='#e31a1c')
    ax4.scatter(t,Nd_RBC_CB_5[t_loc],marker='o',facecolors='#0070C0')
    ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax4.set_ylabel('Resistance (kN)', fontsize=10)
    ax4.set_xlabel('t (mm)', fontsize=10)
    
    ax5 = f1.add_subplot(4,2,6)
    ax5.plot(l6, Nn_RBC_CB_6, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax5.plot(l6, Nd_RBC_CB_6, color='#0070C0',linewidth=1.5, linestyle='solid')
    l_loc=np.where(l6==l)[0].item()
    ax5.scatter(l,Nn_RBC_CB_6[l_loc],marker='o',facecolors='#e31a1c')
    ax5.scatter(l,Nd_RBC_CB_6[l_loc],marker='o',facecolors='#0070C0')
    ax5.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax5.set_ylabel('Resistance (kN)', fontsize=10)
    ax5.set_xlabel('L (mm)', fontsize=10)
    
    ax7 = f1.add_subplot(4,2,7)
    ax7.plot(e7, Nn_RBC_CB_7, color='#e31a1c',linewidth=1.5, linestyle='solid')
    ax7.plot(e7, Nd_RBC_CB_7, color='#0070C0',linewidth=1.5, linestyle='solid')
    e_loc=np.where(e7==e)[0].item()
    ax7.scatter(e,Nn_RBC_CB_7[e_loc],marker='o',facecolors='#e31a1c')
    ax7.scatter(e,Nd_RBC_CB_7[e_loc],marker='o',facecolors='#0070C0')
    ax7.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax7.set_ylabel('Resistance (kN)', fontsize=10)
    ax7.set_xlabel('e (mm)', fontsize=10)    
    
    f1.legend(ncol=2, fontsize=10, bbox_to_anchor=(0.52, -0.07), loc='lower center')
    f1.tight_layout()
    st.pyplot(f1)

    st.subheader('Nomenclature')
    st.write("B and H are the width and height of rectangular column cross section; CatBoost is categorical gradient boosting regressor; L is column length; Nn and Nd are the nominal and design resistances of beam-columns; e is load eccentricity; f'c is concrete compressive strength; fy is steel yield strength; t is tube wall thickness.")
