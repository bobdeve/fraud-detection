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

bash
Copy
Edit
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

🚀 How to Reproduce


