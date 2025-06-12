# Heart-Disease-Prediction

This project was developed by our MSBA cohort using a Kaggle dataset to predict the likelihood of heart disease in patients based on clinical indicators. Our objective was to build and compare multiple machine learning models to identify high-risk patients and support early intervention strategies.

---

## 📊 Project Overview

- **Goal**: Predict the presence of heart disease using clinical data.
- **Dataset**: [Kaggle - Heart Disease UCI Dataset]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset))
- **Target Variable**: `target` (binary classification: 0 = No Heart Disease, 1 = Heart Disease)
- **Team**: MSBA Graduate Cohort

---

## 🧹 Data Cleaning & Preprocessing

- Handled missing values and corrected data types
- Normalized numeric features using StandardScaler
- Performed exploratory data analysis (EDA) to understand variable importance

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Naïve Bayes  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Neural Network (MLP)  
- Ensemble Models (Voting Classifier, Bagging, Boosting)

---

## 🧠 Key Insights

- **Random Forest** and **Ensemble Models** were the **top-performing models**, achieving the best balance between accuracy, precision, recall, and F1-score.
- The **Decision Tree** model had the **lowest performance**.
- Important features included **chest pain type**, **maximum heart rate achieved**, and **resting blood pressure**.
- These models can help detect high-risk individuals and support preventive healthcare initiatives.


---

## 📚 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow / keras (if used)
