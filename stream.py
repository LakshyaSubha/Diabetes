import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn
from imblearn.over_sampling import SMOTE
import streamlit as st

st.set_page_config(layout='wide')


st.markdown("<h1 style='text-align: center; color: red'>**Diabetes Mellitus Prediction **</h1>", unsafe_allow_html=True)

model = pickle.load(open("my_model.pkl","rb"))

file = st.file_uploader("Please upload the csv file with Pregnencies,glucose,BloodPressure,SkinThickness,Insulin,BMI,Diabetes,Age\
     information to the model as input",type=['csv'])

if file is not None:

    df = pd.read_csv(file)
    st.dataframe(df)

    st.text("Note: If the prediction is 1 then Diabetics else Healthy")
      
    if(st.button("Convert")):
        y_pred = model.predict(df)
        st.text(f"Your Diabetes is predicted as {y_pred}")
        

else:
    st.text("Please upload File")