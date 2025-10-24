# 👩‍💼 HR Attrition Analysis & Employee Risk Scoring

## 🎯 Objective
Predict and explain **employee attrition** using machine learning, and assign a measurable **Attrition Risk Score** to every employee to guide proactive retention decisions.

This analysis focuses on understanding how **performance reviews, tenure, overtime, education, and demographics** contribute to turnover risk.  
The notebook builds an interpretable machine learning pipeline that identifies high-risk employees and visualizes risk distributions across departments and job roles.

---

## 🗂️ Data Source
Dataset is automatically fetched from **Kaggle** using `kagglehub`:

> **HR Analytics: Employee Attrition and Performance**  
> Author – *mahmoudemadabdallah*  
> [🔗 Kaggle Dataset Link](https://www.kaggle.com/datasets/mahmoudemadabdallah/hr-analytics-employee-attrition-and-performance)

The notebook dynamically downloads and unzips the dataset at runtime — no manual uploads required.

---

## 🧰 Tech Stack
- **Python:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Machine Learning:** scikit-learn (RandomForestClassifier, train/test split, metrics, class weighting)  
- **Automation:** kagglehub (dataset fetch)  

To install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
🧱 Workflow Overview
1️⃣ Data Acquisition
Downloads dataset from Kaggle via:

python
Copy code
import kagglehub
path = kagglehub.dataset_download("mahmoudemadabdallah/hr-analytics-employee-attrition-and-performance")
Extracts and loads the data into a pandas DataFrame for further analysis.

2️⃣ Data Quality & Exploratory Analysis
Detects employees missing performance reviews and explores their characteristics.

Analyzes HireYear, Department, and YearsAtCompany distributions.

Visualizes missing data patterns and departmental imbalances using barplots and histograms.

Quantifies how missing reviews or overtime affect attrition likelihood.

3️⃣ Feature Engineering
Converts categorical target:

python
Copy code
df['AttritionFlag'] = df['Attrition'].map({'Yes':1, 'No':0})
One-hot encodes categorical columns:

bash
Copy code
['Gender','Department','JobRole','EducationField',
 'MaritalStatus','BusinessTravel','OverTime','State',' Ethnicity']
Drops identifiers (e.g., EmployeeID, Name, HireDate) and redundant features.

Splits data into 70/30 stratified train-test sets for balanced modeling.

4️⃣ Modeling
Implements RandomForestClassifier:

python
Copy code
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
Evaluates on holdout data using:

Accuracy, Precision, Recall, F1-score, ROC-AUC

Confusion Matrix and classification report for transparency.

5️⃣ Risk Scoring
Predicts probability of attrition using:

python
Copy code
df['AttritionRiskScore'] = model.predict_proba(X)[:,1]
Flags high-risk employees:

python
Copy code
df['HighRisk'] = df['AttritionRiskScore'] > 0.70
Builds a ranked risk table with employee identifiers, departments, roles, and scores.

EmployeeID	Department	Role	Risk Score	High Risk
1073	Sales	Executive	0.83	✅
1049	HR	Associate	0.66	❌

6️⃣ Risk Interpretation & Visualization
Visualizes feature importance (top predictive variables).

Generates box plots of risk scores by Job Role and Department.

Highlights top attrition drivers: OverTime, JobSatisfaction, YearsAtCompany, PerformanceReview.

📊 Key Insights
Employees missing performance reviews exhibit a higher attrition probability.

Overtime frequency and lower satisfaction correlate strongly with churn.

Long tenure without promotions raises retention concerns.

Random Forest achieved ~84% accuracy, balanced precision/recall, and ROC-AUC ≈ 0.90.

📈 Visual Outputs
Visualization	Description
Missing performance review analysis
Department-wise attrition rate
Top predictive features from Random Forest
Role-level risk score distribution

(Export these charts from your notebook and save them in /outputs/visuals/.)

📁 Repository Structure
Copy code
hr-attrition-analysis/
├── README.md
├── notebooks/
│   └── HR Analytics.ipynb
├── outputs/
│   └── visuals/
├── requirements.txt
└── .gitignore
.gitignore

kotlin
Copy code
data/*
*.csv
*.xlsx
.ipynb_checkpoints/
.DS_Store
▶️ How to Run
bash
Copy code
# 1. Clone repository
git clone https://github.com/ppd1073/hr-attrition-analysis.git
cd hr-attrition-analysis

# 2. (optional) Create environment
python -m venv .venv
# Activate
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook notebooks/"HR Analytics.ipynb"
🧩 Next Steps
Integrate SHAP explainability for local/global model insights.

Build a Streamlit dashboard for interactive risk visualization.

Perform cross-validation and GridSearchCV hyperparameter tuning.

Implement MLflow for model tracking and version control.
