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
Splits data into 70/30 stratified train-test sets for balanced modeling.

4️⃣ Modeling
Implements RandomForestClassifier:
Evaluates on holdout data using:

Accuracy, Precision, Recall, F1-score, ROC-AUC

Confusion Matrix and classification report for transparency.

5️⃣ Risk Scoring
Predicts the probability of attrition using:
Builds a ranked risk table with employee identifiers, departments, roles, and scores.

6️⃣ Risk Interpretation & Visualization
Visualizes feature importance (top predictive variables).

Generates box plots of risk scores by Job Role and Department.

Highlights top attrition drivers: OverTime, JobSatisfaction, YearsAtCompany, PerformanceReview.

📊 Key Insights
Employees missing performance reviews exhibit a higher attrition probability.

Overtime frequency and lower satisfaction correlate strongly with churn.

Long tenure without promotions raises retention concerns.

Random Forest achieved ~84% accuracy, balanced precision/recall, and ROC-AUC ≈ 0.90

🧩 Next Steps
Integrate SHAP explainability for local/global model insights.

Build a Streamlit dashboard for interactive risk visualization.

Perform cross-validation and GridSearchCV hyperparameter tuning.

Implement MLflow for model tracking and version control.
