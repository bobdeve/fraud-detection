🕵️‍♂️ Fraud Detection - Interim Submission Report
This repository contains my progress for Task 1 – Data Analysis and Preprocessing of the fraud detection project. It includes comprehensive data cleaning, exploratory data analysis (EDA), feature engineering, and class imbalance handling techniques to prepare the dataset for fraud prediction modeling.

📁 Repository Structure
bash
Copy
Edit
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
python
Copy
Edit
from utils.data_utils import transform_and_balance

X_train, X_test, y_train, y_test = transform_and_balance(X, y)
Or, run the Jupyter notebooks in order from the /notebooks directory.

📌 Next Steps
Move to model development and evaluation (Task 2).

Explore additional feature interactions and time-based fraud patterns.

Perform hyperparameter tuning after baseline models are tested.



Task 2 - Model Building and Training
Overview
In Task 2, we focused on building, training, and evaluating machine learning models to detect fraud in the datasets. The main goal was to develop models that can effectively handle the class imbalance inherent in fraud detection.

Data Preparation
Loaded the cleaned and preprocessed data from Task 1.

Separated features (X) and target (y) variables.

Performed a train-test split with stratification to maintain class distribution.

Applied SMOTE on the training set to balance the minority class.

Model Selection
We built and compared the following models:

Logistic Regression

Served as a simple and interpretable baseline model.

Random Forest Classifier

A powerful ensemble model that handles non-linearity and interactions well.

XGBoost Classifier

A gradient boosting model known for its high performance in classification tasks.

Model Training and Evaluation
All models were trained on the balanced training data.

Performance was evaluated on the test set using metrics appropriate for imbalanced classification:

Area Under the Precision-Recall Curve (AUC-PR)

F1-Score

Confusion Matrix

Classification Report (precision, recall, f1-score)

Results and Insights
Random Forest and XGBoost models significantly outperformed Logistic Regression in terms of AUC-PR and F1-Score.

Logistic Regression showed lower recall and F1-Score on the minority (fraud) class.

Random Forest achieved the best balance between precision and recall, making it the preferred model for this task.

The evaluation demonstrated the importance of using ensemble methods and balancing techniques in fraud detection.

