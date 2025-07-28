
# 🚨 Fraud Detection System

## 📋 Table of Contents
1. [Overview](#overview)  
2. [Problem](#problem)  
3. [Project Structure](#project-structure)  
4. [Datasets](#datasets)  
5. [Key Steps](#key-steps)  
6. [Tech Stack](#tech-stack)  
7. [Testing & CI/CD](#testing--cicd)  
8. [Running the Project](#running-the-project)  
9. [Results](#results)  
10. [Conclusion](#conclusion)  
11. [Author](#author)

---

## 🧠 Overview  
This project builds a system to detect fraudulent transactions using machine learning. It uses two types of data:  
- Online purchase activities  
- Credit card transactions  

The goal is to build models that can detect fraud accurately and explain why a transaction is suspicious.

### 💼 Why it matters:  
Fraud causes major financial losses. This system helps businesses detect fraud early while reducing false alarms. That means better protection and a smoother experience for customers.

---

## ❗ Problem  
Fraud is hard to detect because fake transactions are very rare. This project solves that by using smart algorithms that can detect unusual behavior, even in very imbalanced datasets.

---

## 🗂️ Project Structure  

```
FraudDetection/
├── data/                  # Datasets
├── notebooks/             # Jupyter Notebooks for each step
├── src/                   # Python code for data and model logic
├── tests/                 # Unit tests
├── .github/workflows/     # CI/CD workflow using GitHub Actions
├── requirements.txt       # Python libraries needed
├── pyproject.toml         # Project settings
└── README.md              # This file
```

### Key Notebooks:
- `eda.ipynb`: Data analysis and cleaning  
- `modeling.ipynb`: Train and evaluate models  
- `explainability.ipynb`: Show which features matter using SHAP

---

## 📊 Datasets  
1. **Fraud_Data.csv** – Online user data (signup time, device, purchase info, etc.)  
2. **IpAddress_to_Country.csv** – Maps IPs to countries  
3. **creditcard.csv** – Credit card transactions with anonymized features

---

## ⚙️ Key Steps

### 🔍 Task 1: Data Analysis
- Clean the data and handle missing values  
- Add features like time between signup and purchase, user behavior, etc.  
- Add country info using IP addresses  
- Analyze and visualize patterns

### 🤖 Task 2: Model Training
- Train Logistic Regression (simple) and LightGBM (advanced)  
- Use SMOTE to handle imbalance  
- Evaluate with F1-Score, AUC-ROC, AUC-PR, and confusion matrix  
- **LightGBM gave the best results**

### 🧠 Task 3: Model Explainability
- Use SHAP to explain model decisions  
- Show most important features  
- Help understand what causes fraud

---

## 🛠️ Tech Stack  
- Python  
- Pandas, NumPy  
- Scikit-learn, LightGBM  
- Imbalanced-learn (SMOTE)  
- SHAP  
- Matplotlib, Seaborn  

---

## ✅ Testing & CI/CD

### Unit Testing
- Uses `pytest` to check that the code (especially feature engineering) works properly  
- Tests like IP-to-country mapping and time calculations are included

### Continuous Integration
- GitHub Actions runs tests and Jupyter notebooks automatically whenever changes are pushed  
- Makes sure the project stays stable and reproducible

---

## 🏃 Running the Project

### 1. Set up environment:
```bash
git clone https://github.com/Samrizee/fraud-cases-for-e-commerce-and-bank-transactions.git
cd fraud-cases-for-e-commerce-and-bank-transactions
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

### 2. Add data:
Put the following files into the `data/` folder:  
- `Fraud_Data.csv`  
- `IpAddress_to_Country.csv`  
- `creditcard.csv`

### 3. Run tests:
```bash
pytest
```

### 4. Run Notebooks:
```bash
jupyter notebook
```
Then open each notebook (`task-1-EDA.ipynb`, `Task-2-Modeling.ipynb`, `Task-3-Explainability.ipynb`) and run all cells.

---

## 📈 Results

| Dataset         | Model               | F1  | AUC-ROC | AUC-PR |
|-----------------|---------------------|-----|---------|--------|
| Fraud_Data      | Logistic Regression | 0.63 | 0.84    | 0.66   |
| Fraud_Data      | LightGBM            | 0.69 | 0.84    | 0.71   |
| CreditCard_Data | Logistic Regression | 0.10 | 0.96    | 0.72   |
| CreditCard_Data | LightGBM            | 0.85 | 0.97    | 0.80   |

**LightGBM** performed best and was selected for final use and explanation.

---

## 🧾 Conclusion  
This project successfully built a working fraud detection system. It includes strong data prep, smart modeling (LightGBM + SMOTE), and interpretability (SHAP). The modular code structure, tests, and CI pipeline make it reliable and easy to maintain.

---

## 👤 Author  
[Samrizee on GitHub](https://github.com/Samrizee/fraud-cases-for-e-commerce-and-bank-transactions.git) 