# 🧠 Stroke Prediction App

This Streamlit web application uses machine learning models to analyze medical data and predict the likelihood of a stroke. The app includes data preprocessing, exploratory data analysis (EDA), outlier handling, feature importance visualization, and evaluation of multiple machine learning models with K-Fold cross-validation.

🔗 **Live App**: [View on Streamlit](https://afsheenmahmood-stroke-prediction-main-tbjpdw.streamlit.app/)

---

## 📊 Features

- **Interactive EDA**: Explore missing values, distributions, outliers, and grouped data.
- **Preprocessing**: Handles missing values, encodes categories, transforms outliers.
- **Model Training**: Trains multiple models (Random Forest, SVM, Logistic Regression, etc.).
- **K-Fold Cross-Validation**: Compare model performance robustly.
- **Classification Reports**: Detailed precision, recall, F1-score output.
- **Feature Importance**: Identify the most important variables.
- **Visualization**: Interactive charts with Plotly and Seaborn.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Seaborn & Matplotlib
- Plotly
---

## 🚀 Running the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/AfsheenMahmood/stroke-prediction.git
cd stroke-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
