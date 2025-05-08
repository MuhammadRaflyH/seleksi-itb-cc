import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import altair as alt

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

# containers
data_c = st.container()
st.divider()
preparation_c = st.container()
st.divider()
visualisation_c = st.container()

# data container
data_c.header("Data")
dataframe_col, description_col = data_c.columns(2)

# description column
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
preparation_c.header("Data Preparation")
summary_tab, missing_tab, duplicate_tab, outlier_tab, transformation_tab, selection_tab = preparation_c.tabs(["Data Summary", "Missing Values", "Duplicate Data", "Outliers", "Data Transformation", "Data Selection"])
cleaned_c = preparation_c.container()
cleaned_dictionary_c = preparation_c.container()

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
    st.write("Now that we've extracted all the useful features from our data, it's time for a final cleanup.")
    st.write("We originally considered dropping :blue-badge[Name] and :blue-badge[Ticket] early on, as they're high-cardinality, mostly unique identifiers.")
    st.write(":blue-badge[Name] was used to extract :blue-badge[Title], which we've already encoded, so it's safe to drop now.")
    st.write(":blue-badge[Ticket] hasn't been used, and doesn't offer any clear value without deep parsing, so we can drop it too.")
    st.write("We'll also drop :blue-badge[PassengerId], which is just an index assigned to each row and holds no predictive power — though you may keep it elsewhere for submission or identification.")
    st.code("df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)")
    df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# cleaned container
cleaned_c.subheader("Cleaned Data")
cleaned_c.dataframe(df)

# cleaned dictionary container
cleaned_dictionary_c.subheader("Cleaned Data Dictionary")
cleaned_dictionary_c.json({
    'Survived': {
        'definition': 'Survival status',
        'key': '0 = Did not survive, 1 = Survived'
    },
    'Pclass': {
        'definition': 'Passenger class',
        'key': '1 = 1st, 2 = 2nd, 3 = 3rd'
    },
    'Sex': {
        'definition': 'Sex (encoded)',
        'key': '0 = Male, 1 = Female'
    },
    'Age': {
        'definition': 'Age in years (with missing values imputed)',
        'key': None
    },
    'SibSp': {
        'definition': '# of siblings or spouses aboard',
        'key': None
    },
    'Parch': {
        'definition': '# of parents or children aboard',
        'key': None
    },
    'Fare': {
        'definition': 'Ticket fare',
        'key': None
    },
    'FareLog': {
        'definition': 'Log-transformed ticket fare',
        'key': 'np.log1p(Fare)'
    },
    'Embarked_C': {
        'definition': 'Embarked from Cherbourg (dummy)',
        'key': '0 = No, 1 = Yes'
    },
    'Embarked_Q': {
        'definition': 'Embarked from Queenstown (dummy)',
        'key': '0 = No, 1 = Yes'
    },
    'Embarked_S': {
        'definition': 'Embarked from Southampton (dummy)',
        'key': '0 = No, 1 = Yes'
    },
    'FamilySize': {
        'definition': 'Total family members aboard (SibSp + Parch + 1)',
        'key': None
    },
    'IsAlone': {
        'definition': 'Passenger was alone',
        'key': '0 = Not alone, 1 = Alone'
    },
    'Title': {
        'definition': 'Passenger title (simplified and encoded)',
        'key': '1 = Mr, 2 = Miss, 3 = Mrs, 4 = Master, 5 = Rare'
    },
    'AgeGroup': {
        'definition': 'Age group (binned and encoded)',
        'key': '1 = Child, 2 = Teen, 3 = Adult, 4 = Senior'
    }
}, expanded = 1)

# visualisation container
visualisation_c.header("Data Visualization")
survival_tab, category_tab, numeric_tab, pca_tab = visualisation_c.tabs([
    "Survival Overview", "Categorical Features", "Numeric Features", "PCA Visualisation"
])

with survival_tab:
    st.subheader("Survival Count")
    surv_counts = df['Survived'].value_counts().rename(index={0: 'Did Not Survive', 1: 'Survived'})
    st.bar_chart(surv_counts)
    st.caption("Shows the total number of passengers who did and did not survive.")

    st.subheader("Overall Survival Rate")
    survival_rate = df['Survived'].mean() * 100
    st.metric(label="Survival Rate", value=f"{survival_rate:.2f}%")
    st.caption("Displays the percentage of passengers who survived overall.")

with category_tab:
    st.subheader("Survival Rate by Sex")
    sex_surv_df = df.groupby('Sex')['Survived'].mean().reset_index()
    sex_surv_df['Sex'] = sex_surv_df['Sex'].map({0: 'Male', 1: 'Female'})
    bar = alt.Chart(sex_surv_df).mark_bar().encode(
        x=alt.X('Sex:N', title='Sex'),
        y=alt.Y('Survived:Q', title='Survival Rate'),
        color=alt.Color('Sex:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']))
    )
    st.altair_chart(bar, use_container_width=True)
    st.caption("Shows survival rate grouped by sex.")

    st.subheader("Survival Rate by Passenger Class")
    class_surv_df = df.groupby('Pclass')['Survived'].mean().reset_index()
    class_surv_df['Pclass'] = class_surv_df['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    st.bar_chart(
        class_surv_df,
        x='Pclass',
        y='Survived',
        x_label='Passenger Class',
        y_label='Survival Rate',
        color='#3399FF'
    )
    st.caption("Shows survival rate by passenger class using vertical bars.")

    st.bar_chart(
        class_surv_df,
        x='Survived',
        y='Pclass',
        x_label='Survival Rate',
        y_label='Passenger Class',
        color='#FF7F0E',
        horizontal=True
    )
    st.caption("Same data as above, shown in a horizontal bar format.")

    st.subheader("Survival Rate by Age Group")
    agegroup_surv = df.groupby('AgeGroup')['Survived'].mean().rename(index={1: 'Child', 2: 'Teen', 3: 'Adult', 4: 'Senior'})
    st.bar_chart(agegroup_surv)
    st.caption("Shows survival rate across different age groups.")

    st.subheader("Survival Rate by Title")
    title_map = {1: 'Mr', 2: 'Miss', 3: 'Mrs', 4: 'Master', 5: 'Rare'}
    title_surv = df.groupby('Title')['Survived'].mean().rename(index=title_map)
    st.bar_chart(title_surv)
    st.caption("Shows how titles (social roles) impacted survival likelihood.")

with numeric_tab:
    st.subheader("Average Fare by Survival Status")
    fare_mean = df.groupby('Survived')['Fare'].mean().rename(index={0: 'Did Not Survive', 1: 'Survived'})
    st.bar_chart(fare_mean)
    st.caption("Average ticket fare grouped by survival status.")

    st.subheader("Average Age by Survival Status")
    age_mean = df.groupby('Survived')['Age'].mean().rename(index={0: 'Did Not Survive', 1: 'Survived'})
    st.bar_chart(age_mean)
    st.caption("Average age of passengers grouped by survival status.")

    st.subheader("Family Size Distribution")
    fam_dist = df['FamilySize'].value_counts().sort_index().reset_index()
    fam_dist.columns = ['FamilySize', 'Count']
    st.bar_chart(
        fam_dist,
        x='FamilySize',
        y='Count',
        x_label='Family Size',
        y_label='Number of Passengers',
        color='#8E44AD'
    )
    st.caption("Shows how many passengers traveled with different family sizes.")

    st.subheader("Fare vs Age (Scatter)")
    scatter_data = df[['Age', 'Fare']].dropna()
    st.scatter_chart(scatter_data)
    st.caption("Displays distribution of passengers by age and fare paid.")

    st.subheader("Survival Trend by Fare")
    fare_trend = df[['Fare', 'Survived']].sort_values('Fare')
    st.line_chart(fare_trend.set_index('Fare'))
    st.caption("Shows survival rate trend based on fare amount.")

    st.subheader("Cumulative Survival by Age (Area Chart)")
    age_trend = df[['Age', 'Survived']].sort_values('Age')
    st.area_chart(age_trend.set_index('Age'))
    st.caption("Shows cumulative survival rate over increasing passenger age.")

with pca_tab:
    numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=['Survived'])
    scaled_data = StandardScaler().fit_transform(numeric_cols)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])

    st.subheader("PCA Projection (Unlabeled)")
    st.scatter_chart(pca_df)
    st.caption("2D projection of features using PCA (not grouped by survival).")

    pca_df['Survived'] = df['Survived'].values
    chart = alt.Chart(pca_df).mark_circle(size=60).encode(
        x='PC1',
        y='PC2',
        color=alt.Color('Survived:N', scale=alt.Scale(scheme='set1')),
        tooltip=['PC1', 'PC2', 'Survived']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption("PCA plot colored by survival status for visual clustering.")
