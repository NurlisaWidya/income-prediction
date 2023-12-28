import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

@st.cache_data()
def load_data():
    df = pd.read_csv('data2.csv')
    x = df[['age', 'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week']]
    y = df[['income']]
    return df, x, y

@st.cache_data()
def train_model(x,y):
  model = DecisionTreeClassifier(max_depth=3, criterion="entropy")
  model.fit(x,y)

  score = model.score(x,y)

  return model, score

def predict(x, y, features):
  model, score = train_model(x,y)

  pred = model.predict(np.array(features).reshape(1,-1))

  return pred, score

