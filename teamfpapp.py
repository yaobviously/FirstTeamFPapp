# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:15:19 2021

@author: yaobv
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import shap

DATAURL = ('https://github.com/yaobviously/practice/blob/main/TM3.csv?raw=true')
TM3 = pd.read_csv(DATAURL)


st.write("""
#        Simple Team Fantasy Points App
         
This app will predict a team's expected fantasy points from Vegas Points and Closing Line'
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    
    vegas_total = st.sidebar.slider('Team Vegas', 95, 125, 110), 
    closing_spread = st.sidebar.slider('Closing Line', -18, 18, 2),
    TeamRB = st.sidebar.slider('Team Reb/100', 36, 52, 44),
    TeamAST = st.sidebar.slider('Team AST/100', 18, 28, 24),
    TeamSTL = st.sidebar.slider('Team STL/100', 6, 18, 12), 
    TeamBLK = st.sidebar.slider('Team BLK/100', 5, 18, 9),
    oppAST = st.sidebar.slider('Opponent AST/100 Allowed', 18, 28, 24),
    oppREB = st.sidebar.slider('Opponent REB/100 Allowed', 36, 52, 44),
    oppBLK = st.sidebar.slider('Opponent BLK/100 Allowed', 5, 18, 9)
    
    data = {'TeamVegas' : vegas_total,
            'ClosingSpread' : closing_spread,
            'TRB' : TeamRB,
            'AST' : TeamAST,
            'STL' : TeamSTL,
            'BLK' : TeamBLK,
            'oppAST' : oppAST, 
            'oppTRB' : oppREB,
            'oppBLK' : oppBLK}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Load the RandomForestRegressor model

rfmodel = pickle.load(open('teamfp.pkl', 'rb'))

# Use model to predict

predictions = rfmodel.predict(df)



st.subheader('Prediction')
st.write(predictions)

