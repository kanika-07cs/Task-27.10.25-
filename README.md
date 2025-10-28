
# 🏦 Bank Term Deposit Subscription Prediction

## 📘 Project Overview

This project predicts whether a client will **subscribe to a term deposit** (binary classification) based on demographic, economic, and campaign-related features from a Portuguese bank’s marketing dataset.

It uses a **Neural Network built with TensorFlow** and integrates with **Streamlit** for real-time prediction through an interactive web interface.

Key highlights:

* Comprehensive **data preprocessing** (handling skewness, outliers, encoding, scaling)
* **SMOTE balancing** for imbalanced data
* Deep learning model trained using **Keras Sequential API**
* End-to-end deployment using **Streamlit**

**Streamlit Link** : https://kanika-07cs-task-27-10-25--app-cxbx6d.streamlit.app/

**Source:** https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
## 📊 Dataset Description
The dataset contains **41,188 records** and **21 attributes**, describing both client and marketing campaign details.

| Feature        | Description                                       | Type        |
| -------------- | ------------------------------------------------- | ----------- |
| age            | Age of the client                                 | Numeric     |
| job            | Type of job                                       | Categorical |
| marital        | Marital status                                    | Categorical |
| education      | Education level                                   | Categorical |
| default        | Has credit in default?                            | Categorical |
| housing        | Has housing loan?                                 | Categorical |
| loan           | Has personal loan?                                | Categorical |
| contact        | Contact communication type                        | Categorical |
| month          | Last contact month                                | Categorical |
| day_of_week    | Last contact day of the week                      | Categorical |
| duration       | Last contact duration (seconds)                   | Numeric     |
| campaign       | Number of contacts performed during this campaign | Numeric     |
| pdays          | Days since last contact from a previous campaign  | Numeric     |
| previous       | Number of contacts before this campaign           | Numeric     |
| poutcome       | Outcome of the previous campaign                  | Categorical |
| emp.var.rate   | Employment variation rate                         | Numeric     |
| cons.price.idx | Consumer price index                              | Numeric     |
| cons.conf.idx  | Consumer confidence index                         | Numeric     |
| euribor3m      | Euribor 3 month rate                              | Numeric     |
| nr.employed    | Number of employees                               | Numeric     |
| **y**          | Client subscribed to term deposit? (`yes`/`no`)   | Target      |

## ⚙️ Data Preprocessing Steps

1. **Handling Skewness:**

   * Applied **Yeo-Johnson Power Transformation** to normalize skewed numeric features.

2. **Outlier Treatment:**

   * Used **IQR-based clipping** to limit extreme outlier values.

3. **Categorical Encoding:**

   * Converted categorical columns using **Label Encoding** and saved encoders as `label_encoders.pkl`.

4. **Scaling:**

   * Standardized numeric features using **StandardScaler**, stored as `scaler.pkl`.

5. **Class Imbalance Handling:**

   * Balanced the dataset using **SMOTE (Synthetic Minority Over-sampling Technique)**.

6. **Train-Test Split:**

   * 80% training and 20% testing using `train_test_split` with stratification on target `y`.


## 🧠 Model Development

A **Neural Network (Sequential Model)** was built using **TensorFlow Keras**:

```python
model = Sequential([
    Dense(64, input_dim=X_train_res.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Training Configuration:**

* Optimizer: `Adam`
* Loss Function: `Binary Crossentropy`
* Metrics: `Accuracy`
* Epochs: `50`
* Batch Size: `32`

## 📈 Evaluation

Evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **Precision, Recall, F1-score**
* **ROC-AUC Curve**


## 💡 Results & Insights

* After applying **SMOTE**, the model achieved **balanced prediction performance**.
* **Duration**, **Euribor rate**, and **Employment variation rate** had the highest influence on predictions.
* The **ROC-AUC score of ~0.95** indicates excellent model discrimination ability.

## 🚀 How to Run

1. Install Requirements
- pip install -r requirements.txt

2. Run the Streamlit App
- streamlit run app.py

## 🧾 Conclusion

This project demonstrates how to build a **complete end-to-end machine learning pipeline**:

* Data cleaning, encoding, scaling, and balancing
* Deep learning model development and evaluation
* Model deployment using **Streamlit**


Would you like me to include a ready `requirements.txt` and `project folder structure` section (for GitHub setup)?
It’ll make your README fully deployment-ready.
