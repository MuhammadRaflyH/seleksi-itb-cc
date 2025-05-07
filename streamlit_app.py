import streamlit as st
import kagglehub
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load dataset
path = kagglehub.dataset_download("brendan45774/test-file")
df = pd.read_csv(f"{path}/tested.csv")

df
df.shape
