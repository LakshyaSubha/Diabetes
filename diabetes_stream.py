import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import pickle
import streamlit as st

st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: center; color: red;'>**Diabetes**</h1>",unsafe_allow_html=True)

pca= pickle.load(open("pca.pkl","rb"))
model = pickle.load(open("model_3.pkl","rb"))

file = st.file_uploader("Please upload the csv file with Pregnencies,glucose,BloodPressure,SkinThickness,Insulin,BMI,Diabetes,Age,Outcome needs to be given as the input",type=['csv'])

if file is not None:

    df = pd.read_csv(file)
    st.dataframe(df)
      
    if(st.button("Convert")):
        y_pred = model.predict(df)
        st.text(f"Your Diabetes is predicted as {y_pred}")

else:
    st.text("Please upload File")
