import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def detect_outliers_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

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
    dictionary_c = description_col.container()

# dataframe column
with dataframe_col:
    st.subheader("Dataframe")
    st.dataframe(df)

# description container
description_c.subheader("Description")
description_c.write("👋🛳️ Ahoy, welcome to Kaggle! You’re in the right place. This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works. The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.")

# dictionary container
dictionary_c.subheader("Data Dictionary")
dictionary_c.json(
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
      ' parch': {
            'definition': '# of parents / children aboard the Titanic',
            'key': None
        },
      ' ticket': {
            'definition': 'Ticket number',
            'key': None
        },
      ' fare': {
            'definition': 'Passenger fare',
            'key': None
        },
      ' cabin': {
            'definition': 'Cabin number',
            'key': None
        },
      ' embarked': {
            'definition': 'Port of Embarkation',
            'key': 'C = Cherbourg, Q = Queenstown, S = Southampton'
        }
    }, expanded = 1)

# preparation container
preparation_c = st.container()
preparation_c.header("Data Preparation")
summary_tab, missing_tab, duplicate_tab, outlier_tab, transformation_tab, selection_tab = preparation_c.tabs(["Data Summary", "Missing Values", "Duplicate Data", "Outliers", "Data Transformation", "Data Selection"])

# summary tab
with summary_tab:
    st.dataframe(df.describe())

# missing tab
with missing_tab:  
    st.write("Can I delete every row that has missing values?")
    st.code(f"There are {df.isnull().any(axis=1).sum()} rows with missing values out of {len(df)} rows.")
    st.write("How many missing values do each attribute have?")
    st.code(df.isnull().sum())

    st.write("That's alot of missing :blue-badge[Cabin] values and it has such a messy format. Let's get rid of the :blue-badge[Cabin] attribute first.")
    st.code("df.drop(columns=['Cabin'], inplace=True)")
    df.drop(columns=['Cabin'], inplace=True)

    st.write(":blue-badge[Embarked] has so little missing values, that we can impute with the most frequent value because its simple, effective, keeps all rows, and causes minimal distortion.")
    st.code("df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)")
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    st.write("It's a bit trickier with :blue-badge[Age]. Even though it's not very precise, to respect :blue-badge[Age] differences by :blue-badge[PClass] and :blue-badge[Sex], we will impute :blue-badge[Age] using group medians based off :blue-badge[PClass] and :blue-badge[Sex].")
    st.code("df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))")
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# duplicate tab
with duplicate_tab:
    st.write("Let's check for duplicate data.")
    st.code("df.duplicated().sum()")
    st.write("Nice! :thumbsup:")

# outlier tab
with outlier_tab:
    st.write("Let's check for outliers using the IQR method for numeric columns like :blue-badge[Age] or :blue-badge[Fare].")
    outliers_fare = detect_outliers_iqr('Fare')
    outliers_age = detect_outliers_iqr('Age')
    st.code(f"Age outliers: {len(outliers_age)}\n"
            "Fare outliers: {len(outliers_fare)}")

    st.write("We can keep :blue-badge[Age] outliers because old age is realistic and 33 data is not large.")
    st.write(":blue-badge[Fare] on the other hand needs to be log-transformed into :blue-badge[FareLog] to reduce skew.")
    st.code("df['Fare_log'] = np.log1p(df['Fare'])")
    df['FareLog'] = np.log1p(df['Fare'])

# transformation tab
with transformation_tab:
    st.write("Let's now move on to transforming the data to prepare it for modeling.")

    st.write("We begin by encoding categorical variables into numerical format. Since :blue-badge[Sex] is binary, we can use simple label encoding (0 = male, 1 = female).")
    st.write("For :blue-badge[Embarked], which has more than two categories, we'll use one-hot encoding so that the model doesn't assume any ordinal relationship between ports.")
    st.code("df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})\n"
            "df = pd.get_dummies(df, columns=['Embarked'])")
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'])

    st.write("Next, we'll engineer a few new features that can help the model pick up useful patterns.")
    st.write("We create :blue-badge[FamilySize] by adding :blue-badge[SibSp] and :blue-badge[Parch], plus one for the passenger.")
    st.write("Then we create :blue-badge[IsAlone], a binary feature that tells us if the passenger was traveling alone.")
    st.write("We also extract :blue-badge[Title] from the passenger's name (like Mr, Mrs, Miss) and simplify rare titles into a 'Rare' category, before encoding them numerically.")
    st.code("df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n"
            "df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n"
            "# extract and clean Title...\n")
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
    df['Title'] = df['Title'].fillna(0)

    st.write("Finally, we discretize :blue-badge[Age] into :blue-badge[AgeGroup] bins to group people into categories like child, teen, adult, and senior. This may help models that prefer categorical features.")
    st.code("df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])")
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
    age_group_map = {'Child': 1, 'Teen': 2, 'Adult': 3, 'Senior': 4}
    df['AgeGroup'] = df['AgeGroup'].map(age_group_map)

with selection_tab:
    st.write("Now that we've extracted the useful parts of the data, it's time to clean up.")
    st.write("We originally wanted to drop :blue-badge[Name] and :blue-badge[Ticket] because they are high-cardinality, mostly unique identifiers — not useful directly for modeling.")
    st.write("However, we used :blue-badge[Name] to extract :blue-badge[Title], so now we can safely drop it.")
    st.write("Let's go ahead and remove both columns to avoid introducing noise into the model.")
    st.code("df.drop(columns=['Name', 'Ticket'], inplace=True)")
    df.drop(columns=['Name', 'Ticket'], inplace=True)

preparation_c.subheader("Cleaned Data")
preparation_c.dataframe(df)
