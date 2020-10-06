import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Test Simple Iris Prediction

Ngetest aja gan
""")

st.sidebar.header('Input')

def user_input_features():
    sepal_length = st.sidebar.slider('sepal_length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 0.2)
    data = {'SepalLengthCm': sepal_length,
            'SepalWidthCm': sepal_width,
            'PetalLengthCm' : petal_length,
            'PetalWidthCm' : petal_width}
    features = pd.DataFrame(data, index = [0])
    return features

input_df = user_input_features()

iris_raw = pd.read_csv('Iris.csv')
iris = iris_raw.drop(columns = ['Species', 'Id'])
df = pd.concat([input_df, iris], axis = 0)

df = df[:1]

st.subheader('User Input')
st.write(df)


iris_species = np.array(['Iris-setosa','Iris-versicolor','Iris-virginica'])


st.subheader('Class')
st.write(iris_species)

#baca model
load_clf = pickle.load(open('iris_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
st.write(iris_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
