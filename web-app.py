import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

st.write("""
# Electrical Characterization Prediction
This app predicts the electrical behaviour of a circuit given some features and some random variability.
Currently only a CMOS inverter is available 
""")

st.sidebar.header("Input Features")

file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if file is not None:
    input = pd.DataFrame(file)
else:
    def user_input():
        voltage = st.sidebar.slider("Voltage (V)",0.6,0.9,0.7)
        p_width = st.sidebar.slider("Pmos width (nm)",70,420,265)
        n_width = st.sidebar.slider("Nmos width (nm)",70,210,120)
        length = st.sidebar.slider("Nmos and Pmos length (nm)",32,40,35)
        temperature = st.sidebar.slider("Temperature (°C)",-25,100,30)
        n_var = st.sidebar.slider("Nmos variability",0.46,0.59,0.51)
        p_var = st.sidebar.slider("Pmos variability",-0.5,-0.4,-0.45)
        data = {
            'voltage': voltage,
            'width_pmos': p_width*(10**-9),
            'width_nmos': n_width*(10**-9),
            'length': length*(10**-9),
            'temper': temperature,
            'nmos@var': n_var,
            'pmos@varp': p_var
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input = user_input()

df = input.copy()
data = pd.read_csv('clean_data.csv')
st.subheader('User Input features')

if file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

scaler = StandardScaler().fit(df)
df_sc = scaler.transform(df)

iint_clf = pickle.load(open('iint.pkl', 'rb'))
iint_pred = iint_clf.predict(df_sc)
st.subheader('Energy Prediction')
st.write(((iint_pred+data.iint.min())*(data.iint.max()-data.iint.min()))*(10**14))

tphl_clf = pickle.load(open('tphl.pkl', 'rb'))
tphl_pred = tphl_clf.predict(df_sc)
st.subheader('High-Low Propagation Prediction')
st.write(((tphl_pred+data.tphl.min())*(data.tphl.max()-data.tphl.min()))*(10**12))

tplh_clf = pickle.load(open('tplh.pkl', 'rb'))
tplh_pred = tplh_clf.predict(df_sc)
st.subheader('Low-High Propagation Prediction')
st.write(((tplh_pred+data.tplh.min())*(data.tplh.max()-data.tplh.min()))*(10**12))