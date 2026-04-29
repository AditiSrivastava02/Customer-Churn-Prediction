# Customer-Churn-Prediction
Customer churn prediction using machine learning to identify high-risk customers and improve retention strategies.

## Overview

This project focuses on predicting customer churn for a telecom company using machine learning techniques. The objective is to identify customers who are likely to leave the service, enabling proactive retention strategies.

## Dataset

Dataset source: [Customer Churn Prediction](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information such as:

* Demographics (gender, senior citizen, etc.)
* Account details (tenure, contract type, payment method)
* Service usage (internet service, streaming, etc.)
* Billing information (monthly charges, total charges)

Target variable:

* **Churn (Yes/No)**

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* TensorFlow / ANN
* Imbalanced-learn (SMOTE)
* Matplotlib, Seaborn

## Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Handling class imbalance using SMOTE

## Models Implemented

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Artificial Neural Network (ANN)

## Model Performance (Test Set - 20%)

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest       | 0.7523   | 0.5235    | 0.7433 | 0.6144   | 0.8376  |
| ANN                 | 0.5429   | 0.3636    | 0.9626 | 0.5279   | 0.8341  |
| Logistic Regression | 0.7395   | 0.5068    | 0.6979 | 0.5872   | 0.8283  |
| XGBoost             | 0.7119   | 0.4755    | 0.8289 | 0.6043   | 0.8212  |
| Decision Tree       | 0.7374   | 0.5038    | 0.7139 | 0.5907   | 0.8159  |

## Best Model

Random Forest achieved the best overall performance based on ROC-AUC.

* Accuracy: 0.7523
* Precision: 0.5235
* Recall: 0.7433
* F1 Score: 0.6144
* ROC-AUC: 0.8376

## Customer Risk Segmentation

Customers are categorized into risk tiers based on churn probability:

| Risk Tier | Customers | Actual Churners | Avg Churn Prob | Avg Monthly Charges | Avg Tenure | Churn Rate (%) | Revenue at Risk |
| --------- | --------- | --------------- | -------------- | ------------------- | ---------- | -------------- | --------------- |
| Critical  | 224       | 156             | 0.842          | $78.00              | 7.5        | 69.6           | $17,472         |
| High      | 307       | 122             | 0.624          | $73.34              | 20.9       | 39.7           | $22,517         |
| Medium    | 285       | 65              | 0.369          | $72.20              | 32.3       | 22.8           | $20,576         |
| Low       | 593       | 31              | 0.099          | $50.15              | 46.7       | 5.2            | $29,738         |

## Key Insights

* Random Forest provides the best balance between recall and precision
* ANN achieves very high recall but low precision, leading to more false positives
* Customers with low tenure and high monthly charges are more likely to churn
* Risk segmentation helps prioritize retention strategies

## Outputs

All visualizations and results (confusion matrix, ROC curves, feature importance) are available in the `outputs/` folder.

## Project Structure

```bash
Customer-Churn-Prediction/
│
├── outputs/
├── Customer Churn Prediction.zip/
├── Customer_Churn_Prediction.ipynb/
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-ml.git
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the notebook:

```bash
jupyter notebook
```

## Author

Aditi Srivastava
