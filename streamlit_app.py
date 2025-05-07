import streamlit as st
import kagglehub
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# title
st.title("Tugas Seleksi ITB CC - Titanic Dataset")

# dataset and metadata
path = kagglehub.dataset_download("brendan45774/test-file")
df = pd.read_csv(f"{path}/tested.csv")

col1, col2 = st.columns(2)

with col1:
  df
with col2:
  st.write("👋🛳️ Ahoy, welcome to Kaggle! You’re in the right place. This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works. The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.")
  df.shape
