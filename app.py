import streamlit as st
import pandas as pd
import numpy as np
import joblib

with open('pipeline.pkl', 'rb') as file_1:
  pipeline = joblib.load(file_1)


v1 = st.slider('Masukan nilai v1',227.285714, 678.375000)
v2 =  st.slider('Masukan nilai v2',178.800000, 422.812500)
v3 =  st.slider('Masukan nilai v3',348.933333, 722.312500)
v4 = st.slider('Masukan nilai v4',313.733333, 558.500000)
v5 = st.slider('Masukan nilai v5',373.333333, 721.000000)
v6 = st.slider('Masukan nilai v6',189.200000, 415.375000)
v7 = st.slider('Masukan nilai v7',586.266667, 853.466667)
v8 = st.slider('Masukan nilai v8',3725.666667, 5086.375000)
sample_type = st.selectbox('masukkan tipe sample', {'lab 1', 'lab 2'}, index=1)

if st.button('predict target'):

    data_inf = pd.DataFrame({'v1' : v1, 'v2': v2, 'v3': v3, 'v4' : v4, 'v5' : v5, 
                    'v6' : v6, 'v7' : v7, 'v8' : v8, 'sample_type' : sample_type},index=[0])
    target = pipeline.predict(data_inf)
    st.write(target)