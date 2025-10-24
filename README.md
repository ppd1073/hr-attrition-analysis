# 👩‍💼 HR Attrition Analysis & Employee Risk Scoring

## 🎯 Objective
Predict and explain **employee attrition** using machine learning, and assign a measurable **Attrition Risk Score** for each employee to guide retention planning.

---

## 🗂️ Data Source
The dataset is fetched automatically from Kaggle using `kagglehub`:
> **HR Analytics: Employee Attrition and Performance**  
> Author — mahmoudemadabdallah  
> [Kaggle Dataset Link](https://www.kaggle.com/datasets/mahmoudemadabdallah/hr-analytics-employee-attrition-and-performance)

The notebook downloads and unzips the latest dataset dynamically — no manual upload required.

---

## 🧰 Tech Stack
- **Python:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Machine Learning:** scikit-learn (RandomForestClassifier, train/test split, metrics)  
- **Automation:** kagglehub (dataset fetch)  

To install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
🧱 Workflow Overview

Data Acquisition

Dataset pulled via kagglehub.dataset_download()

Files listed and read into a pandas DataFrame

Data Quality & EDA

Identify employees missing performance reviews

Compare the HireYear distribution for missing vs. non-missing

Analyze Department and YearsAtCompany differences

Visualize hiring patterns and performance data gaps

Feature Engineering

Convert Attrition → binary target (Yes = 1 / No = 0)

One-hot encode categorical columns:
['Gender',' Department','JobRole','EducationField','MaritalStatus','BusinessTravel','OverTime','State','Ethnicity']

Drop IDs, names, and date fields that could leak information

Modeling

RandomForestClassifier with class_weight='balanced'

Train/test split (70 / 30, stratified)

Evaluate using precision, recall, F1, ROC-AUC, and confusion matrix

Risk Scoring

Compute AttritionRiskScore = rf.predict_proba(X)[:,1]

Flag HighRisk = True for score > 0.70

Create a risk_table with employee ID, name, role, department, and score

Role-wise Insights

Boxplots of AttritionRiskScore by JobRole

Compare risk spread and median score per role

📊 Key Insights

Performance review absence and OverTime frequency show a strong correlation with attrition.

Employees in high-stress or travel-intensive roles rank highest on risk score.

Tenure and HireYear explain retention variation across departments.

📈 Visual Outputs

Save visuals under outputs/visuals/ and reference them here:

Visualization	Description

	Distribution of missing performance reviews

	Attrition rate by department

	Top predictors from Random Forest

	Boxplot of risk scores by job role
📁 Repository Structure
hr-attrition-analysis/
├── README.md
├── notebooks/
│   └── HR Analytics.ipynb
├── outputs/
│   └── visuals/
├── requirements.txt
└── .gitignore


.gitignore

data/*
*.csv
*.xlsx
.ipynb_checkpoints/
.DS_Store

▶️ How to Run
# 1. Clone repository
git clone https://github.com/<your-username>/hr-attrition-analysis.git
cd hr-attrition-analysis

# 2. (optional) Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open notebook
jupyter notebook notebooks/"HR Analytics.ipynb"

🧩 Next Steps

Add SHAP/Explainable AI layer for local feature attribution.

Build a Streamlit web interface for uploading employee records and visualizing risk scores.

Introduce cross-validation & hyperparameter tuning (GridSearchCV).

Integrate model tracking (e.g., MLflow).

Author: Prajakta Deshpande

📧 ppdpune@gmail.com

📍 Gujarat | Open to Remote / Relocation
