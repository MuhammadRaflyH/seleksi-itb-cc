import streamlit as st
import kagglehub
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load dataset
path = kagglehub.dataset_download("uciml/iris")
df = pd.read_csv(f"{path}/Iris.csv")

# drop ID column
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# extract features and labels
features = df.drop(columns=['Species'])
labels = df['Species']

# apply PCA
pca = PCA(n_components=2)
components = pca.fit_transform(features)

# add components to a DataFrame
pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
pca_df['Species'] = labels

# output dataframe
st.title("Iris Dataset")
st.dataframe(df)

# plot PCA result
st.subheader("2D PCA Scatter Plot")
fig, ax = plt.subplots()
for species in pca_df['Species'].unique():
  subset = pca_df[pca_df['Species'] == species]
  ax.scatter(subset['PC1'], subset['PC2'], label=species)
ax.legend()
st.pyplot(fig)
