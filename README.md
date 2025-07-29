🕵️‍♂️ Fraud Detection - Interim Submission Report
This repository contains my progress for Task 1 – Data Analysis and Preprocessing of the fraud detection project. It includes comprehensive data cleaning, exploratory data analysis (EDA), feature engineering, and class imbalance handling techniques to prepare the dataset for fraud prediction modeling.

📁 Repository Structure

# Clone the repo
git clone https://github.com/bobdeve/fraud-detection.git
cd fraud-detection

# Set up a virtual environment (optional but recommended)
python -m venv .fraud
source .fraud/bin/activate  # or .fraud\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/task1_data_cleaning_eda.ipynb
jupyter notebook notebooks/task2_model_training.ipynb
fraud-detection/
│
├── data/                   # Raw and processed data files (if shared)
├── notebooks/              # Jupyter notebooks for EDA and preprocessing
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_feature_engineering.ipynb
│
├── utils/                  # Python scripts for reusable functions
│   └── data_utils.py
│
├── outputs/                # Saved figures and visualizations from EDA
│
├── README.md               # Project overview and instructions
└── requirements.txt        # Required Python packages
🚀 Getting Started
📦 Installation
To run this project locally:


git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
🧼 1. Data Cleaning and Preprocessing
The dataset underwent the following preprocessing steps (see data_utils.py and 01_data_cleaning.ipynb):

✅ Missing Value Handling: Detected and imputed missing values using median imputation for numerical features. Columns with excessive NaNs (e.g., time_since_last_transaction) were dropped.

✅ Duplicates Removal: Removed duplicate rows to prevent data leakage and model bias.

✅ Data Type Correction: Ensured datetime features were parsed correctly and numerical types were standardized.

📊 2. Exploratory Data Analysis (EDA)
Performed in 02_eda.ipynb:

📈 Visualizations include:

Fraud vs. Non-Fraud transaction amounts

Hourly and daily transaction distribution

Customer sign-up and transaction time gaps

💡 Insights:

Fraudulent transactions often occur in unusual timeframes or with extreme amounts.

Newer accounts are slightly more prone to fraud.

All plots are annotated and saved in /outputs/.

🧠 3. Feature Engineering
Done in 03_feature_engineering.ipynb and applied in data_utils.py:

Created new predictive features like:

time_since_signup

avg_transaction_amount_per_user

✨ Hypothesis: Fraudulent users may:

Transact immediately after signing up

Show abnormal frequency or amount behavior compared to regular users

⚖️ 4. Handling Class Imbalance
Initial class distribution showed severe imbalance (~1–3% fraud cases).

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the training set.

Distribution and sampling results are shown in both plots and console outputs.

Code is implemented in transform_and_balance() function inside data_utils.py.

📎 How to Reproduce

from utils.data_utils import transform_and_balance

X_train, X_test, y_train, y_test = transform_and_balance(X, y)
Or, run the Jupyter notebooks in order from the /notebooks directory.

📌 Next Steps
Move to model development and evaluation (Task 2).

Explore additional feature interactions and time-based fraud patterns.

Perform hyperparameter tuning after baseline models are tested.



📌 Task 2 - Model Building and Training
✅ Task Completion
This notebook implements and compares two models on the cleaned and preprocessed fraud dataset:

Logistic Regression: A simple, interpretable baseline model.

XGBoost & Random Forest: Powerful ensemble models well-suited for imbalanced classification.

Data Handling:

Separated features and target (class) from the dataset.

Performed stratified train-test split to preserve class distribution.

Used SMOTE to oversample the minority (fraud) class during training.

Model Evaluation:

Each model was evaluated using metrics tailored for imbalanced classification tasks:

Area Under Precision-Recall Curve (AUC-PR)

F1-Score

Confusion Matrix

Classification Report (Precision, Recall, F1, Support)

Key Insights:

XGBoost and Random Forest outperformed Logistic Regression significantly in terms of F1-score and AUC-PR.

Random Forest showed the best balance of precision and recall, making it the strongest candidate for fraud detection on this dataset.

🧪 Documentation & Reproducibility
All code is well-commented, modularized, and logically ordered.

Each step of the modeling pipeline (data loading, splitting, balancing, training, and evaluation) is clearly explained.

Complex operations like SMOTE balancing and metric evaluation are wrapped in reusable functions to ensure maintainability.

The notebook runs cleanly from top to bottom without errors.

📁 Repository Organization


📊 Metric Summary
Model	AUC-PR	F1-Score	Precision (Fraud)	Recall (Fraud)
LogisticRegression	0.4197	0.2807	0.18	0.70
Random Forest	0.6295	0.6988	1.00	0.54
XGBoost	0.6174	0.6898	0.96	0.54

## 🔍 Task 3: Model Explainability using SHAP

In this task, we used **SHAP (SHapley Additive exPlanations)** to interpret our best-performing model — the **XGBoost Classifier** — and understand what features drive fraud predictions.

### 🎯 Objective

To uncover **global** and **local** feature importance:
- Understand which features contribute most to fraud detection.
- Visualize individual prediction explanations.
- Reveal feature interactions and their effects on model output.

---

### 🧠 SHAP Workflow

1. **Explainer Setup:**
   We used `shap.Explainer()` on the trained XGBoost model and test data:

   ```python
   explainer = shap.Explainer(model, X_train)
   shap_values = explainer(X_test)

   Summary Plot (Global Importance):


What it shows: The average impact of each feature on the model’s output.

Insight: Features like V14, V10, V12, and Amount have the most influence in identifying fraud.

Color bar: Indicates feature value (red = high, blue = low).

Bar Plot of Top 10 Features:


Mean absolute SHAP values were calculated to rank features.

Fraud prediction is most sensitive to extreme values in features like V14 and V10.

Force Plot (Local Explanation):


What it shows: How individual feature values push a prediction toward fraud (1) or not fraud (0).

Usage: Helps explain specific flagged transactions for auditing or debugging.

Interaction Plot (Optional Advanced Analysis):

Analyzed interaction effects between pairs of features.

Found that Time interacts with anonymized features like V1 in contributing to fraud risk.

🗝️ Key Takeaways from SHAP
V14, V10, and V12 are top fraud indicators.

Higher transaction amounts (Amount) tend to push predictions toward fraud.

Local force plots reveal the reasoning behind a single fraud prediction, increasing transparency.

SHAP provides both trust and debugging power to your model pipeline.

📂 Output Files
File	Description
outputs/shap_summary.png	Summary plot showing global feature impact
outputs/fraud_feature_importance.png	Bar plot of top 10 important features
outputs/shap_force_plot_example.png	Local explanation for a single sample

