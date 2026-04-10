# 📡 TeleChurn Intelligence: Predicting Customer Churn with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-✔-green?style=flat-square)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-✔-yellow?style=flat-square)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](https://opensource.org/licenses/MIT)

---

## 🧭 Project Overview

Customer churn — when a customer cancels their subscription — is one of the most costly problems in the telecommunications industry. Acquiring a new customer is anywhere from **5 to 25 times more expensive** than retaining an existing one. This project builds a full end-to-end machine learning pipeline to **identify which customers are most likely to churn**, so a business can intervene proactively.

**Problem Statement:**
Given historical customer data (demographics, services, billing, and contract information), can we accurately predict whether a customer will churn before it happens?

**Goal:**
Develop and compare multiple ML classification models, optimize them for high recall on the minority (churn) class, and deliver a production-ready scoring pipeline with explainability via SHAP.

---

## 📂 Dataset Description

| Property | Detail |
|---|---|
| **Source** | IBM Sample Dataset — `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Records** | ~7,043 customers (11 dropped due to null conversion) |
| **Final Shape** | ~7,032 rows × 21 columns |
| **Target Variable** | `Churn` — Binary: Yes / No |
| **Class Distribution** | ~73.5% No Churn / ~26.5% Churn (imbalanced) |

**Key Feature Groups:**

- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Financials:** `MonthlyCharges`, `TotalCharges`

> **Note:** `TotalCharges` was stored as a string and required coercion to numeric, revealing 11 missing values that were subsequently dropped.

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA phase (`Full_EDA.ipynb`) explored customer behavior across demographic, behavioral, service, and financial dimensions.

### Key Findings

### 1. Senior Citizen Risk (High-Value, High-Churn)

Senior citizens pay ~29% more per month on average (~$79 vs ~$61) but churn at nearly double the rate of non-seniors — **41.7% vs 23.7%**. This represents the most critical revenue-at-risk segment.

![Senior Citizen Churn Analysis](https://github.com/user-attachments/assets/94056322-67c4-44bd-ac2b-13614778a55d)

> Senior citizens are a high-value yet high-risk segment, indicating a strong opportunity for targeted retention strategies.
----

### 2. Safety Net Effect (Household Structure)

Customers with partners churn at only **19.7%** versus **33%** for solo customers. Partnered customers also stay an average of **42 months** compared to just 23 months for solo customers — a clear "social anchor" effect. Customers with both a partner and dependents show the lowest churn and longest tenure.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/cd0875f2-7f3d-4d81-8b10-bd59c6d5230a" width="100%"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/1a196028-ae4c-4f63-910c-26644797c8fd" width="100%"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2d5ecf11-6a28-4419-a4b9-b29bc34b770c" width="100%"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6abecb6b-4549-4998-b507-e09a731d2e9f" width="100%"/>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/a74c95d2-3827-4d23-b6e2-83c7eb3a6df8" width="50%"/>
    </td>
  </tr>
</table>

> Customers with stronger household ties (partner + dependents) exhibit significantly lower churn and higher tenure, highlighting the importance of lifestyle-based segmentation in retention strategies.ure.
---

### 3. The Loyalty Cliff (Service Stickiness)

Churn rate drops dramatically with additional service subscriptions:
- 0 services → ~21% churn  
- 1 service → ~45% churn *(highest — not yet integrated)*  
- 5–6 services → ~5% churn  

This signals that cross-selling protective services (security, backup, tech support) is a powerful retention lever.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a0b5ea96-7640-46c1-8c55-03d9efbc5514" width="100%"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/4e385469-6862-428f-b191-24e2994ef9dd" width="100%"/>
    </td>
  </tr>
</table>

> Customers with more subscribed services show significantly lower churn, highlighting the importance of increasing product adoption to improve customer retention.
---

### 4. Internet Technology Disparity

Fiber optic customers churn at **41.9%** — over twice the 19% DSL churn rate — despite paying ~$91/month on average. This suggests high price sensitivity and elevated expectations in this premium segment, leading to increased dissatisfaction.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/0bb22474-4568-422d-8af4-c27084005dde" width="100%"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6a44bbf5-f5b5-40e2-aad2-b12df2848d2c" width="100%"/>
    </td>
  </tr>
</table>

> Fiber optic customers represent a high-value but high-risk segment, indicating a need for improved service quality, pricing strategies, and targeted retention efforts.
---


### 5. The Danger Zone Matrix (Contract × Payment Method)

The most dangerous customer profile: Month-to-month contract + Electronic Check payment = **53.7% churn rate**. By contrast, two-year contract + credit card auto-pay = **2.2% churn**. Electronic check customers re-evaluate every month, making them inherently high-risk.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e492f74a-de8e-4971-9759-fc7129d91c3a" width="80%"/>
</p>

| Contract         | Bank Transfer (Auto) | Credit Card (Auto) | Electronic Check | Mailed Check |
|------------------|---------------------|--------------------|------------------|--------------|
| Month-to-Month   | 34.13%              | 32.78%             | **53.73%**       | 31.58%       |
| One Year         | 9.72%               | 10.30%             | 18.44%           | 6.85%        |
| Two Year         | 3.38%               | 2.24%              | 7.70%            | **0.8%**     |

> The combination of short-term contracts and non-automated payment methods significantly increases churn risk. Encouraging long-term contracts and auto-pay adoption can drastically improve customer retention.
---


### 6. Revenue at Risk

High-risk customers (month-to-month + fiber optic + electronic check, still active) represent a concentrated pocket of revenue exposure that can be directly targeted with retention campaigns.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5a24ff32-19ea-4663-ae4c-4383cf3bbff7" width="75%"/>
</p>

**High-Risk Segment Definition**

| Condition                             | Meaning                       | Why risky                                                 |
|--------------------------------------|-------------------------------|----------------------------------------------------------|
| `Churn == 'No'`                      | Customer has **not left yet** | We want to identify customers who **may leave in future** |
| `Contract == 'Month-to-month'`       | No long-term commitment       | These customers can leave **anytime**                     |
| `InternetService == 'Fiber optic'`   | High price service            | Expensive services often show **higher churn**            |
| `PaymentMethod == 'Electronic check'`| Payment through e-check       | Historically **highest churn segment**                    |

---

### 📊 Revenue Risk Summary

| Metric | Value |
|--------|------|
| **High-Risk Customers Remaining** | 518 |
| **Monthly Revenue at Risk** | $45,666.60 |
| **Annual Revenue Exposure** | $547,999.20 |
| **Total Company Monthly Revenue** | $316,530.15 |
| **% Revenue in Danger Zone** | **14.43%** |

> 🚨 **Alert:** Nearly **15% of total revenue** is concentrated in high-risk customers, representing a significant opportunity for targeted retention strategies and proactive intervention.
---


### 7. Tenure–Churn Relationship

A KDE analysis reveals a sharp spike in churn among customers in their **first 12 months**, with churn probability declining steadily as tenure increases. Retaining customers beyond the 24-month mark significantly improves lifetime value.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4daed412-2277-4085-a0af-94f246683aa2" width="65%"/>
</p>

> Early-stage customers (0–12 months) represent the highest churn risk, highlighting the critical importance of onboarding experience and early engagement strategies.
---



### Visualizations Produced

- Count plots and percentage breakdowns for gender, senior citizen, and partner/dependent segmentation
- Grouped bar charts for churn rate and average monthly charges by demographic segment
- "The Loyalty Cliff" bar chart: Churn rate by number of services
- Heatmap: The Danger Zone — Churn Risk by Contract × Payment Method
- Pie chart: Financial vulnerability — revenue at risk vs safe revenue
- KDE plot: Tenure distribution for churned vs retained customers
- Correlation heatmap for numerical features (tenure, MonthlyCharges, TotalCharges)
- Box plots for outlier detection across all numeric features

---

## 🤖 Modeling Approach

All modeling work is in `Model_building.ipynb`.

### Preprocessing Pipeline

1. **Target Encoding:** `Churn` mapped to binary integer (0/1)
2. **Feature Engineering:**
   - `Total_Services` — count of add-on services subscribed
   - `Is_Solo` — flag for customers with no partner or dependents
   - `Is_Auto_Pay` — flag for automatic payment methods
   - `Short_Contract` — flag for month-to-month contracts
   - `Charge_Spike` — difference between current and historical average monthly charges
   - `Charge_Tenure_Ratio` — monthly charges relative to tenure (intensity metric)
   - `Service_Stickiness` — count of protective services subscribed
   - `High_Value_At_Risk` — flag for high-paying fiber customers
3. **Text Trap Fix:** "No internet service" values in service columns normalized to "No" to prevent dummy variable inflation
4. **Column Dropping:** `customerID`, `gender`, `PhoneService`, `MultipleLines`, and redundant engineered features removed
5. **One-Hot Encoding:** `pd.get_dummies` with `drop_first=True`
6. **Train/Test Split:** 80/20 stratified split
7. **Scaling:** `StandardScaler` applied to `MonthlyCharges`, `TotalCharges`, `tenure`, and `Charge_Spike` (fit on train only)

### Models Trained

| Model | Strategy | Tuning Method |
|---|---|---|
| Logistic Regression (Baseline) | Default | None |
| Logistic Regression (Tuned) | `class_weight='balanced'` | GridSearchCV (recall & F1) |
| Logistic Regression (Custom Threshold) | Probability threshold adjustment | GridSearchCV |
| Decision Tree (Baseline) | Default | None |
| Decision Tree (Tuned) | `class_weight='balanced'` | GridSearchCV (recall & F1) |
| Random Forest (Baseline) | Default | None |
| Random Forest (Tuned) | Custom class weights | RandomizedSearchCV |
| XGBoost | `scale_pos_weight` | GridSearchCV (F1 & avg precision) |
| AdaBoost | Shallow DT base, `class_weight='balanced'` | GridSearchCV (recall) |
| LightGBM | `scale_pos_weight` | RandomizedSearchCV (F1) |
| Voting Classifier (Ensemble) | XGBoost + RF (SMOTE) + LightGBM | Soft voting |

**Why these models?**
- Logistic Regression provides an interpretable baseline.
- Tree-based ensembles (Random Forest, XGBoost, LightGBM) handle mixed feature types, non-linear relationships, and class imbalance natively.
- The Voting Classifier leverages model diversity to reduce variance.
- Class imbalance was addressed via `scale_pos_weight`, `class_weight='balanced'`, SMOTE (within a pipeline), and custom threshold optimization.

### Explainability (SHAP)
SHAP (SHapley Additive exPlanations) was applied to the best XGBoost model, generating both global feature importance (beeswarm summary plots) and individual-level waterfall plots to explain predictions for specific customers.

---

## 📊 Results

> *Note: Results below are representative of the notebook outputs. Exact values may vary slightly due to random state and GridSearch sampling.*

### Model Comparison (Test Set — Class 1: Churn)

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Logistic Regression (Baseline) | ~81% | 0.68 | 0.55 | 0.61 |
| Logistic Regression (Tuned + Threshold) | ~78% | 0.58–0.65 | 0.72–0.78 | 0.65–0.70 |
| Decision Tree (Tuned) | ~77% | 0.56 | 0.70 | 0.62 |
| Random Forest (Tuned) | ~80% | 0.64 | 0.68 | 0.66 |
| XGBoost (Tuned) | ~79–81% | 0.64–0.70 | 0.70–0.74 | 0.66–0.72 |
| LightGBM (Tuned) | ~80% | 0.65 | 0.70 | 0.67 |
| Voting Ensemble | ~80% | 0.64 | 0.72 | 0.67 |
| **XGBoost + Advanced Features + Optimal Threshold** | **~79%** | **~0.70** | **~0.74** | **~0.72** |

**Key Takeaways:**
- Accuracy is a misleading metric here due to class imbalance (~73/27 split). **Recall on the churn class** is the primary business metric — catching churners is far more valuable than avoiding false alarms.
- The best-performing model is the **tuned XGBoost** with advanced engineered features and custom threshold optimization (PR-AUC optimized), achieving the best balance of precision and recall for the churn class.
- SHAP analysis identified `Short_Contract`, `tenure`, `MonthlyCharges`, `InternetService_Fiber_optic`, and `Total_Services` as the most globally influential features.

---

## ⚙️ How to Run the Project

### Prerequisites
```bash
python >= 3.8
```

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/telechurn-intelligence.git
cd telechurn-intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core packages required:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
joblib
jupyter
```

### 3. Add the Dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the root directory.

### 4. Run the EDA Notebook
```bash
jupyter notebook Full_EDA.ipynb
```
This notebook processes the raw data and outputs a cleaned feature-engineered file: `output.csv`.

### 5. Run the Modeling Notebook
```bash
jupyter notebook Model_building.ipynb
```
This notebook reads `output.csv`, trains all models, performs hyperparameter tuning, and saves the final model.

### 6. Saved Model
The final production model is saved as:
```
xgboost_churn_model_production.pkl
```
Load it in any Python environment:
```python
import joblib

package = joblib.load('xgboost_churn_model_production.pkl')
model = package['model']
threshold = package['optimal_threshold']
expected_columns = package['expected_columns']
```

---

## 🗂️ Folder Structure

```
telechurn-intelligence/
│
├── Full_EDA.ipynb                        # Data cleaning, EDA, feature engineering, exports output.csv
├── Model_building.ipynb                  # All ML models, tuning, SHAP analysis, model export
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw source dataset (add manually)
├── output.csv                            # Processed & feature-engineered dataset (auto-generated by EDA notebook)
│
├── xgboost_churn_model_production.pkl    # Serialized production model bundle (auto-generated)
│
├── partner_churn_analysis.png            # Saved visualization: Partner vs Churn
├── family_structure_churn.png            # Saved visualization: Family structure vs Churn
│
├── requirements.txt                      # Python package dependencies
└── README.md                             # Project documentation (this file)
```

---

## 🚀 Future Work

1. **Streamlit / Gradio App:** Deploy the model as an interactive web app where a business user can input customer attributes and receive a real-time churn probability score with a SHAP explanation.

2. **Survival Analysis:** Apply Cox Proportional Hazards or Kaplan-Meier models to predict not just *whether* a customer will churn, but *when* — enabling time-sensitive intervention scheduling.

3. **Business Cost-Sensitive Optimization:** Incorporate the actual cost of a false negative (missed churner) versus a false positive (wasted retention offer) into the threshold optimization using a profit-maximization curve.

4. **Real-Time Feature Pipeline:** Integrate with a streaming data platform (e.g., Apache Kafka) to score customers dynamically as their behavioral signals change, rather than on a static snapshot.

5. **CLV-Weighted Modeling:** Weight training samples by Customer Lifetime Value so the model learns to prioritize high-value churners over low-value ones.

6. **Automated Retraining (MLOps):** Set up a scheduled pipeline (Airflow / Prefect) to retrain the model as new monthly data arrives and monitor for concept drift.

7. **Segmented Models:** Train separate models for senior citizens and fiber optic customers — the two highest-churn segments — to improve precision where it matters most commercially.

8. **Additional Data Sources:** Integrate call center interaction logs, NPS survey scores, and website/app engagement data as behavioral churn signals.

---

## 📚 References & Acknowledgments

- **Dataset:** [Telco Customer Churn — IBM Sample Data via Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **SHAP Library:** Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- **XGBoost:** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- **LightGBM:** Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- **imbalanced-learn:** Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). *Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets.* JMLR.
- **scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **Seaborn Documentation:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

---

*Built with ❤️ — turning customer data into retention intelligence.*
