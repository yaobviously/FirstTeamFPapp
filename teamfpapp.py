# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:15:19 2021

@author: yaobv
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import shap


st.write("""
#        Simple Team Fantasy Points App
         
This app will predict a team's expected fantasy points'
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    
    vegas_total = st.sidebar.slider('Team Vegas', 95.0, 125.0, 110.0), 
    closing_spread = st.sidebar.slider('Closing Line', -18, 18, 1),
    TeamRB = st.sidebar.slider('Team Reb/100', 36, 54, 45),
    TeamAST = st.sidebar.slider('Team AST/100', 18, 32, 25),
    TeamSTL = st.sidebar.slider('Team STL/100', 4, 12, 6), 
    TeamBLK = st.sidebar.slider('Team BLK/100', 2, 10, 5),
    oppAST = st.sidebar.slider('Opponent AST/100 Allowed', 18, 32, 25),
    oppREB = st.sidebar.slider('Opponent REB/100 Allowed', 36, 54, 45),
    oppBLK = st.sidebar.slider('Opponent BLK/100 Allowed', 2, 10, 5)
    
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

