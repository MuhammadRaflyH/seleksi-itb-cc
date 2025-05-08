import streamlit as st
import kagglehub
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide"
)

# load dataset
df = pd.read_csv('data/train.csv')

# title
st.title("Tugas Seleksi ITB CC - Titanic Dataset")

# data container
data_c = st.container()
data_c.header("Data")
dataframe_col, description_col = data_c.columns(2)

with description_col:
  description_c = description_col.container()
  data_dictionary_c = description_col.container()

# dataframe column
with dataframe_col:
  st.subheader("Dataframe")
  st.dataframe(df)

# description container
description_c.subheader("Description")
description_c.write("👋🛳️ Ahoy, welcome to Kaggle! You’re in the right place. This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works. The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.")

# data dictionary container
data_dictionary_c.subheader("Data Dictionary")
data_dictionary_c.json(
  {
      'survival': {
          'definition': 'Survival',
          'key': '0 = No, 1 = Yes'
      },
      'pclass': {
          'definition': 'Ticket class',
          'key': '1 = 1st, 2 = 2nd, 3 = 3rd'
      },
      'sex': {
          'definition': 'Sex',
          'key': None
      },
      'age': {
          'definition': 'Age in years',
          'key': None
      },
      'sibsp': {
          'definition': '# of siblings / spouses aboard the Titanic',
          'key': None
      },
      'parch': {
          'definition': '# of parents / children aboard the Titanic',
          'key': None
      },
      'ticket': {
          'definition': 'Ticket number',
          'key': None
      },
      'fare': {
          'definition': 'Passenger fare',
          'key': None
      },
      'cabin': {
          'definition': 'Cabin number',
          'key': None
      },
      'embarked': {
          'definition': 'Port of Embarkation',
          'key': 'C = Cherbourg, Q = Queenstown, S = Southampton'
      }
 }, expanded = 1)
