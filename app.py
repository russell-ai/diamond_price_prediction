import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

pd.set_option("display.max.columns", None)

st.write("""
# Diamond Price Prediction App
This app predicts the **Diamond Price**
""")
st.sidebar.header('Select Diamond Features')
st.sidebar.markdown("""
**Specify diamond properties** 
""")


def user_input_features():
    carat = st.sidebar.slider('Carat', 0.20, 5.01, 0.79)
    cut = st.sidebar.selectbox('Cut', ('Ideal', 'Premium', 'Very Good', 'Good', 'Fair'))
    color = st.sidebar.selectbox('Color', ('G', 'E', 'F', 'H', 'D', 'I', 'J'))
    clarity = st.sidebar.selectbox('Clarity', ('SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1'))
    depth = st.sidebar.slider('Depth', 43., 79., 61.)
    table = st.sidebar.slider('Table', 43., 95., 57.)
    x = st.sidebar.slider('X mm', 0., 10.75, 5.73)
    y = st.sidebar.slider('Y mm', 0., 59., 5.73)
    z = st.sidebar.slider('Z mm', 0., 32., 3.53)

    data = {'carat': carat,
            'cut': cut,
            'color': color,
            'clarity': clarity,
            'depth': depth,
            'table': table,
            'x': x,
            'y': y,
            'z': z,
            }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

dataset = pd.read_csv('diamonds.csv')
df = dataset.drop(columns=['price'])
df = pd.concat([input_df, df], axis=0)

encode = ['cut', 'color', 'clarity']
for col in encode:
    dummy = pd.get_dummies(df[col], drop_first=True)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')
st.write(df)
st.markdown("""---""")

# Read the saved model
with open('GBR_estimator.pkl', 'rb') as file:
    model = pickle.load(file)

# Apply loaded model to make predictions
st.subheader('Diamond Value Prediction')
prediction = model.predict(df)
st.header(f'US ${np.round(prediction,2)}')


st.markdown(
    """
    ---
    
    
    [for more contact](https://www.resulcaliskan.com) 
    """)