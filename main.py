import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


# Set page config
st.set_page_config(page_title="Stroke Prediction Model Evaluation")

# Load Data Function with Caching
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/AfsheenMahmood/Stroke-Prediction/main/stroke%20prediction%20dataset.csv"
    return pd.read_csv(url, encoding="utf-8")
df = load_data()

# Title Section
st.title("Stroke Prediction Data Analysis")

# Introduction to Stroke
st.header("What is a Stroke?")
st.write("""
A stroke occurs when there is a problem with the blood supply to the brain. It can be caused by a blockage or the rupture of a blood vessel. 
When blood flow is interrupted, brain cells are deprived of oxygen, leading to brain damage and possible loss of function.
""")

# Dataset Introduction
st.header("About the Stroke Prediction Dataset")
st.write("""
This dataset contains medical data related to stroke prediction. It includes various attributes such as age, gender, hypertension, heart disease, marital status, and more, which are used to predict the likelihood of a stroke.
""")

# Exploratory Data Analysis (EDA) Section
st.header("Exploratory Data Analysis (EDA)")

# Show first 5 rows of data
st.subheader("Dataset Overview")
st.dataframe(df.head())

# Display Shape of Data
st.subheader("Shape of Data")
st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# DataFrame Information
st.subheader("DataFrame Information")
info_df = pd.DataFrame({
    'Column Name': df.columns,
    'Non-Null Count': df.notnull().sum(),
    'Data Type': df.dtypes,
})
st.dataframe(info_df)

# Statistical Summary of Data
st.subheader("Statistical Summary")
st.write(df.describe())

# Missing Values Analysis
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Display Duplicates
st.subheader("Duplicated Values")
st.write(f"Number of duplicated rows: {df.duplicated().sum()}")

# Display Column Names
st.subheader("Column Names")
st.write(df.columns.tolist())

# Data Type of Each Column
st.subheader("Data Types")
st.write(df.dtypes)

# Missing Values Bar Graph
st.subheader("Graph for Missing Values")
missing_percentage = df.isnull().mean() * 100
plt.figure(figsize=(12, 6))
missing_percentage.sort_values(ascending=False).plot(kind='bar', color='salmon')
st.pyplot(plt)

# Fill Missing Values for 'bmi'
df['bmi'].fillna(df['bmi'].median(), inplace=True)
st.write("Missing values in 'bmi' have been filled with the median value.")
st.write(df.isnull().sum())

# Gender Unique Values
st.subheader("Unique Values for Gender")
st.write(df['gender'].value_counts())

# Replace 'Other' with Mode for Gender Column
mode_gender = df['gender'].mode()[0]
df['gender'] = df['gender'].replace('Other', mode_gender)
st.write("Gender column updated with mode value instead of 'Other'.")
st.write(df['gender'].value_counts())

# Function to Create Sorted Intervals for Columns
def create_sorted_intervals(df, column, bin_size):
    min_val = df[column].min()
    max_val = df[column].max()
    bins = range(int(min_val), int(max_val) + bin_size + 1, bin_size)
    intervals = pd.cut(df[column], bins=bins, right=False)
    sorted_labels = sorted(intervals.cat.categories, key=lambda x: x.left)
    sorted_labels = [f"{int(label.left)}-{int(label.right)}" for label in sorted_labels]
    df[f'{column}_group'] = pd.cut(df[column], bins=bins, right=False, labels=sorted_labels)
    return df

# Create age, bmi, and glucose level groups
df = create_sorted_intervals(df, 'age', 15)
df = create_sorted_intervals(df, 'bmi', 15)
df = create_sorted_intervals(df, 'avg_glucose_level', 20)

# Exclude certain columns from display
exclude_columns = ["id", "avg_glucose_level", "bmi", "age"]
columns_to_show = [col for col in df.columns if col not in exclude_columns]

# Select column to view unique values and plot distribution
st.subheader("Unique Values and Distribution")
selected_column = st.selectbox("Select a column to view unique values and distribution:", columns_to_show)
unique_values_counts = df[selected_column].value_counts().reset_index()
unique_values_counts.columns = [selected_column, "Count"]
st.dataframe(unique_values_counts)

# Count plot for selected categorical column
if df[selected_column].dtype == 'object' or df[selected_column].nunique() < 20:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x=selected_column, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Box Plot for Outliers
st.subheader("Box Plot for Outliers")
numeric_columns = df.select_dtypes(include=['float64']).columns
for column in numeric_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x=column)
    st.pyplot(plt)

# Handle Outliers Using Quantile Transformer
columns_to_transform = ['avg_glucose_level', 'bmi']
quantile_transformer = QuantileTransformer(output_distribution='uniform')
df[columns_to_transform] = quantile_transformer.fit_transform(df[columns_to_transform])
st.write("Outliers have been treated using Quantile Transformer.")
st.header("Box Plot for Mean After Scaling")
for column in numeric_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x=column)
    st.pyplot(plt)
st.header("Distribution of Numeric Columns")
numeric_cols = df.select_dtypes(include=['float64','object']).columns
for col in numeric_cols:
    st.write(f"Distribution of {col}:")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with reduced width (increased bins for finer granularity)
    sns.histplot(df[col], kde=False, stat="density", color='skyblue', bins=50, ax=ax)  # Increase bin count for finer bins
    st.pyplot(fig)
# 21. Group By Work Type/Ever Married/Smoking Status
st.header("Group By Selection")
df_col=["work_type", "ever_married","smoking_status"]
# Dropdown for selecting a column to group by
group_by_column = st.selectbox("Select a column to group by:", df_col)

# Get unique values of the selected column
unique_values = df[group_by_column].unique()

# Dropdown for selecting a specific group
selected_group = st.selectbox(f"Select a value from {group_by_column}:", unique_values)

# Filter the DataFrame based on the selected group
filtered_df = df.groupby(group_by_column).get_group(selected_group)

# Display filtered data
st.subheader(f"Data for {group_by_column} = {selected_group}")
st.dataframe(filtered_df)

# # Selectbox for column selection
st.header("Countplot for Various Columns")
countplot_option = st.selectbox(
    "Select a column for countplot:", 
    ['work_type','age_group','bmi_group','avg_glucose_level_group', 'smoking_status',  'Residence_type', 'ever_married', 'gender', 'hypertension', 'heart_disease']
    )

# Create the countplot for the selected column with respect to stroke
plt.figure(figsize=(10, 4))
sns.countplot(data=df, x=countplot_option, hue='stroke')

# Display the plot
st.pyplot(plt)
# 23. ML Model with K-Fold Validation


st.header("Feature Importance for Stroke Prediction")

# Encode categorical columns with LabelEncoder

label_encoder = LabelEncoder()

# Apply LabelEncoder to all categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Define your features and target variable
X = df.drop(columns=['id', 'stroke','age_group','bmi_group','avg_glucose_level_group'])  # Remove target and non-numeric columns
y = df['stroke']

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance'])
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plot the feature importances
fig, ax = plt.subplots(figsize=(8, 6))
feature_importances.plot(kind='bar', color='skyblue', ax=ax)

# Customize plot
ax.set_title('Feature Importance for Stroke Prediction')
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)
st.header("Correlation Analysis After Label Encoding")

# Create a copy of the DataFrame to avoid modifying the original data
df_encoded = df.copy()

# Encode categorical columns with LabelEncoder
label_encoder = LabelEncoder()
categorical_cols = df_encoded.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

# Drop interval-based columns before computing correlation
interval_cols = ['age_group', 'bmi_group', 'avg_glucose_level_group','id']
df_encoded = df_encoded.drop(columns=interval_cols, errors='ignore')

# Compute the correlation matrix
correlation_matrix = df_encoded.corr()

# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix ")

# Display the heatmap in Streamlit
st.pyplot(fig)


# ✅ Move this to the very top
# 🌟 Title
st.title("Model Performance Evaluation with K-Fold Cross-Validation")

# 📊 Sidebar Selection
st.sidebar.header("Settings")
split_ratio = st.sidebar.slider("Test Set Ratio", 0.1, 0.5, 0.2, step=0.05)
num_splits = st.sidebar.slider("Number of K-Folds", 2, 10, 5)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42, stratify=y)

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# K-Fold Cross-Validation
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

# 🎯 Model Evaluation
results = []
confusion_matrices = {}

for name, model in models.items():
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    
    # Train & predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Accuracy & classification report
    test_accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Store confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

    # Append results
    results.append({
        'Model': name,
        'Mean CV Accuracy': cv_scores.mean(),
        'Std CV Accuracy': cv_scores.std(),
        'Test Accuracy': test_accuracy
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# 📈 Plot Model Performance
st.subheader("Model Performance Comparison")

fig = px.bar(
    results_df,
    x='Model',
    y=['Mean CV Accuracy', 'Test Accuracy'],
    barmode='group',
    text_auto='.2f',
    title="Model Performance Comparison",
    labels={"value": "Accuracy", "variable": "Metric"}
)
fig.update_layout(
    xaxis_title="Models",
    yaxis_title="Accuracy",
    legend_title="Metrics",
    height=500
)
st.plotly_chart(fig)

# 🏆 Best Model
best_model = results_df.loc[results_df['Test Accuracy'].idxmax()]
st.success(f"Best Performing Model: **{best_model['Model']}** with **{best_model['Test Accuracy']:.4f}** accuracy!")

# 📌 Detailed Results Table
st.subheader("Model Performance Summary")
st.dataframe(results_df.style.format({
    "Mean CV Accuracy": "{:.4f}",
    "Std CV Accuracy": "{:.4f}",
    "Test Accuracy": "{:.4f}"
}).background_gradient(cmap='Blues', subset=["Mean CV Accuracy", "Test Accuracy"]))

# 📝 Classification Report Section
st.subheader("Classification Reports")

selected_model = st.selectbox("Select a model to view detailed classification report:", results_df['Model'].tolist())

for name, model in models.items():
    if name == selected_model:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        st.text_area(f"Classification Report: {name}", report, height=300)

