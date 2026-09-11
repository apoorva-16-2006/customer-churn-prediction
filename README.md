# customer-churn-prediction
Machine Learning customer churn prediction system using SVM, Decision Tree, and Random Forest with a Streamlit web application and majority voting.


# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn based on customer demographics, tenure, charges, contract type, internet service, and technical support information.

The project includes data analysis, preprocessing, model training, hyperparameter tuning, model evaluation, and a Streamlit web application. The final application combines predictions from **SVM, Decision Tree, and Random Forest** models using a majority voting approach.

---

## 🚀 Project Overview

Customer churn prediction helps businesses identify customers who may leave their service.

In this project, customer information is used to build classification models that predict the `Churn` outcome:

- **Yes** → Customer is likely to churn
- **No** → Customer is unlikely to churn

The project compares multiple classification algorithms and deploys the trained models through an interactive Streamlit application.

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── Customer_Churn_Prediction.ipynb
│
├── SVM_pipeline.pkl
├── decision_Tree_pipeline.pkl
├── random_forest_pipeline.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset contains **1,000 customer records** and **10 columns**.

The features used for prediction are:

| Feature | Description |
|---|---|
| Age | Customer age |
| Gender | Customer gender |
| Tenure | Number of months with the service |
| MonthlyCharges | Customer's monthly charges |
| InternetService | Type of internet service |
| TotalCharges | Total amount charged |
| ContractType | Customer's contract type |
| TechSupport | Whether the customer has technical support |
| Churn | Target variable |

`CustomerID` is used as an identifier and is not included in the prediction features.

The dataset contains missing values in `InternetService`, which are handled by replacing them with `"unknown"`. No duplicate records were found. 

---

## 🔄 Machine Learning Workflow

The project follows an end-to-end Machine Learning workflow:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Transformation
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Save Trained Pipelines
   ↓
Streamlit Deployment
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### Missing Values

Missing values in `InternetService` were replaced with:

```text
unknown
```

### Categorical Features

Categorical variables were encoded using:

```python
OneHotEncoder(drop="first")
```

Categorical features include:

- Gender
- InternetService
- ContractType
- TechSupport

### Numerical Features

Numerical features were standardized using:

```python
StandardScaler()
```

The numerical features include:

- Age
- Tenure
- MonthlyCharges
- TotalCharges

The preprocessing and model are combined into a single Scikit-learn pipeline for deployment.

---

## 🤖 Machine Learning Models

Several classification algorithms were explored during the project.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

Test accuracy:

```text
94.33%
```

### 2. Support Vector Machine

An SVM classifier was tuned using `GridSearchCV`.

The best parameters were:

```text
C = 1
kernel = rbf
```

Test accuracy:

```text
94%
```

### 3. Decision Tree

A Decision Tree classifier was optimized using `GridSearchCV`.

The best parameters found were:

```text
criterion = gini
max_depth = None
min_samples_leaf = 1
min_samples_split = 2
splitter = best
```

Test accuracy:

```text
99.67%
```

### 4. Random Forest

A Random Forest classifier was tuned using `GridSearchCV`.

The best parameters were:

```text
bootstrap = True
max_features = 3
n_estimators = 256
```

Test accuracy:

```text
99.67%
```

---

## 📈 Model Comparison

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 94.33% |
| SVM | 94.00% |
| Decision Tree | 99.67% |
| Random Forest | 99.67% |

The Decision Tree and Random Forest achieved the highest recorded test accuracy in the notebook. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## 🗳️ Majority Voting

The Streamlit application uses three trained models:

```text
SVM
Decision Tree
Random Forest
```

Each model makes an independent prediction.

The final prediction is determined using **majority voting**:

```text
2 or more Churn votes
        ↓
      CHURN

Otherwise
        ↓
   NON-CHURN
```

For example:

```text
SVM            → Churn
Decision Tree  → Non-Churn
Random Forest  → Churn

Final Result   → Churn
```

This voting logic is implemented directly in the Streamlit application. :contentReference[oaicite:6]{index=6}

---

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit interface called:

**Customer Churn Prediction**

Users can enter:

- Age
- Gender
- Tenure
- Monthly Charges
- Internet Service
- Total Charges
- Contract Type
- Tech Support

The application then sends the input to all three trained models and displays:

- Final churn prediction
- Churn vote count
- Non-churn vote count
- Individual SVM prediction
- Individual Decision Tree prediction
- Individual Random Forest prediction
- Customer information

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📁 Trained Models

The repository contains three trained model pipelines:

```text
SVM_pipeline.pkl
decision_Tree_pipeline.pkl
random_forest_pipeline.pkl
```

The Streamlit application loads these models using Joblib when the application starts. :contentReference[oaicite:9]{index=9}

---

## 🎯 Project Objectives

The main objectives of this project are:

- Understand customer churn data
- Perform exploratory data analysis
- Handle missing values
- Preprocess numerical and categorical features
- Train multiple classification algorithms
- Compare model performance
- Perform hyperparameter tuning
- Save trained ML pipelines
- Build an interactive prediction application
- Combine multiple models using majority voting

---

## 🔮 Future Improvements

- Add churn probability instead of only the final class
- Add feature importance visualization
- Add model performance dashboard
- Add customer retention recommendations
- Improve class imbalance handling
- Deploy the Streamlit application online
- Add a batch prediction feature using CSV uploads
- Add monitoring for model performance after deployment

---

## 👨‍💻 Author

**Apoorva Jaliminchi**

BCA – Data Analytics & AI

---

⭐ If you found this project useful, consider giving the repository a star!
