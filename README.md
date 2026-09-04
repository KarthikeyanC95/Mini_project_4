# Employee Attrition and Performance Rating Analysis with Prediction

## 📌 Project Overview

Employee attrition and employee performance are important factors that directly affect an organization's productivity, workforce stability, recruitment costs, and long-term growth.

This project, **Employee Attrition and Performance Rating Analysis with Prediction**, uses data analysis, visualization, feature engineering, feature selection, and machine learning techniques to understand the factors associated with employee turnover and performance.

The main objective is to analyze employee information, identify important factors influencing attrition, compare multiple machine learning models, and develop a predictive approach that can help organizations identify employees who may be at risk of leaving.

The project combines:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Outlier treatment
* Feature engineering
* Feature selection
* Categorical encoding
* Machine learning model training
* Model evaluation
* Attrition prediction
* Performance analysis
* Model deployment preparation

> **Predict the risk. Understand the factors. Improve employee retention.**

---

# 🎯 Business Problem

Employee attrition can create significant challenges for organizations, including:

* Recruitment costs
* Employee onboarding expenses
* Training costs
* Productivity loss
* Loss of experienced employees
* Knowledge transfer issues
* Workforce planning difficulties

A machine learning model can help organizations identify employees who are potentially at higher risk of leaving.

The insights obtained from this project can support HR teams in developing proactive employee-retention strategies.

### Business Use Cases

#### 1. Employee Retention

Identify employees who may be at higher risk of attrition and allow HR teams to take preventive action.

#### 2. Cost Optimization

Reduce costs associated with recruiting, onboarding, training, and replacing experienced employees.

#### 3. Workforce Planning

Use predictive insights to improve workforce planning and employee-management strategies.

#### 4. Employee Satisfaction

Analyze factors such as job satisfaction, overtime, income, tenure, work-life balance, and other employee characteristics that may influence attrition.

#### 5. Performance Analysis

Analyze employee performance-related attributes and understand their relationship with employee retention and organizational outcomes.

---

# 📊 Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

The dataset contains employee-related information covering demographics, job characteristics, compensation, satisfaction, work experience, and performance-related attributes.

### Example Features

* Age
* Gender
* Department
* Business Travel
* Distance From Home
* Education
* Education Field
* Job Satisfaction
* Job Role
* Monthly Income
* Marital Status
* OverTime
* Total Working Years
* Years at Company
* Years in Current Role
* Years Since Last Promotion
* Years With Current Manager
* Number of Companies Worked
* Work-Life Balance
* Performance-related attributes
* And other employee characteristics

### Target Variable

The primary target variable is:

```text
Attrition
```

It contains two classes:

```text
Yes → Employee left the organization
No  → Employee stayed in the organization
```

---

# 🔎 Initial Data Analysis

The dataset contains:

```text
1,470 employee records
35 original columns
```

Missing-value analysis was performed using:

```python
df.isnull().sum().sum()
```

Result:

```text
0
```

Therefore, the original dataset contained no missing values.

Where required in preprocessing experiments, missing-value handling techniques such as mean or median imputation were also considered.

---

# 🧹 Data Preprocessing

Data preprocessing was performed to prepare the employee dataset for machine learning.

The major preprocessing steps included:

* Missing-value analysis
* Numerical feature transformation
* Outlier treatment
* Categorical encoding
* Feature engineering
* Feature selection
* Feature scaling

---

# 📌 Outlier Handling

Outliers were examined using boxplots for numerical variables such as:

* Monthly Income
* Total Working Years
* Years at Company
* Other numerical employee attributes

Several techniques were explored for handling extreme values.

### 1. Log Transformation

Log transformation was considered for highly skewed numerical variables to reduce skewness and the influence of extreme values.

### 2. Square Root Transformation

Square root transformation was also explored.

This technique can be more effective when working with a larger dataset because the current dataset contains a relatively limited number of observations.

### 3. Winsorization

Extreme values were capped using the 5th and 95th percentiles.

This approach reduces the effect of extreme observations without completely removing them.

### 4. IQR Method

The Interquartile Range method was used to identify and cap extreme values.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The boundaries are:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these boundaries can be capped using:

```python
df[column].clip(lower_bound, upper_bound)
```

For the attrition model, processed columns included:

```text
handled_MonthlyIncome
outliersTotalWorkingyear
```

---

# 🧩 Feature Engineering

Additional features were created to improve the representation of employee information and provide machine learning models with more meaningful predictive variables.

Feature engineering considered factors such as:

* Employee tenure
* Job experience
* Income
* Job satisfaction
* Work-life balance
* Number of companies worked
* Overtime
* Department
* Marital status
* Career progression

The purpose of feature engineering was to transform the raw employee information into features that could better represent potential attrition patterns.

---

# 🔤 Categorical Encoding

The dataset contains both categorical and numerical variables.

Categorical variables were transformed into numerical representations so that they could be used by machine learning algorithms.

## One-Hot Encoding

Categorical variables such as:

```text
Department
Marital Status
```

were encoded using `OneHotEncoder`.

### Department

Possible values:

```text
Human Resources
Research & Development
Sales
```

Generated features:

```text
Department_Human Resources
Department_Research & Development
Department_Sales
```

### Marital Status

Possible values:

```text
Divorced
Married
Single
```

Generated features:

```text
MaritalStatus_Divorced
MaritalStatus_Married
MaritalStatus_Single
```

The encoder was saved using Joblib for reuse during deployment:

```python
joblib.dump(
    ohe_for_emp_att,
    r'..\Models\onehotencoderforemp_attri.pkl'
)
```

---

## Binary Encoding

The `OverTime` variable contains two categories:

```text
Yes
No
```

It was transformed using `LabelEncoder`.

The resulting representation was:

```text
No  → 0
Yes → 1
```

The encoder was saved for deployment:

```python
joblib.dump(
    le_for_emp_att,
    r'..\Models\labelforemployeeattri.pkl'
)
```

The saved encoders allow the same transformation process to be applied when new employee information is provided to the prediction application.

---

# 📊 Feature Selection

Multiple feature-selection techniques were explored to identify the variables most relevant to employee attrition and performance analysis.

### Chi-Square Test

The Chi-Square (`chi2`) test was used to identify categorical features that had a significant relationship with the target variable.

### ANOVA

ANOVA was used to identify numerical variables that showed significant differences across target classes.

### Mutual Information

Mutual Information was used to measure the dependency between individual features and the target variable.

### Recursive Feature Elimination

Recursive Feature Elimination (RFE) was used to iteratively remove less important features and retain a more useful feature subset.

### Correlation Analysis

Correlation heatmaps were used to identify highly correlated numerical features and reduce multicollinearity.

---

# 🏗️ Final Feature Set for Attrition Prediction

One of the final feature sets used for employee attrition prediction contained:

```text
Age
handled_MonthlyIncome
JobSatisfaction
YearsAtCompany
OverTime
NumCompaniesWorked
Department_Human Resources
Department_Research & Development
Department_Sales
MaritalStatus_Divorced
MaritalStatus_Married
MaritalStatus_Single
```

The original categorical columns:

```text
Department
MaritalStatus
```

were removed after one-hot encoding.

---

# 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand employee characteristics and identify patterns associated with attrition.

The analysis included:

* Bar plots
* Histograms
* Box plots
* Distribution analysis
* Correlation heatmaps
* Attrition-based comparisons
* Numerical feature analysis
* Categorical feature analysis

Important variables investigated included:

* Age
* Monthly Income
* Job Satisfaction
* Work-Life Balance
* Years at Company
* Overtime
* Marital Status
* Department
* Number of Companies Worked

---

# 🤖 Machine Learning Models

Multiple classification algorithms were evaluated for employee attrition prediction.

The primary models included:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier

The dataset was divided into training and testing sets using stratification:

```python
train_test_split(
    feature_of_Employee_atr,
    target_of_emp_att,
    stratify=target_of_emp_att
)
```

Stratification helps maintain a similar class distribution between the training and testing datasets.

---

# 📊 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

Because employee attrition is an imbalanced classification problem, accuracy alone was not considered sufficient.

Particular attention was given to:

```text
Recall for Attrition = Yes
Precision for Attrition = Yes
F1 Score for Attrition = Yes
```

---

# 📋 Model Performance

## Logistic Regression

Test results from the experiment:

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 85.05% |
| Precision         | 82.41% |
| Recall            | 85.05% |
| Weighted F1 Score | 80.96% |

Classification performance:

```text
              precision    recall    f1-score

No               0.86       0.98       0.92
Yes              0.64       0.15       0.25
```

Logistic Regression achieved approximately **85% test accuracy**.

However, recall for the `Yes` class was only **15%**, meaning that a large proportion of employees who actually left were incorrectly predicted as staying.

---

## Decision Tree Classifier

Test results:

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 79.62% |
| Precision         | 78.91% |
| Recall            | 79.62% |
| Weighted F1 Score | 79.25% |

Training performance:

```text
Accuracy = 100%
Precision = 100%
Recall = 100%
F1 Score = 100%
```

The difference between training and testing performance indicates **overfitting**.

Therefore, perfect training accuracy does not necessarily mean that the Decision Tree is the best production model.

---

## Random Forest Classifier

Test results:

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 83.15% |
| Precision         | 79.27% |
| Recall            | 83.15% |
| Weighted F1 Score | 80.18% |

The Random Forest model achieved approximately **83% test accuracy**.

The model also achieved 100% training accuracy in the experiment, indicating some degree of overfitting.

Random Forest can potentially be improved through hyperparameter tuning and class-balancing techniques.

---

## XGBoost

XGBoost was also evaluated as part of the modeling process.

During the initial experiment, the model produced an error because the target variable contained string labels:

```text
Yes
No
```

while the XGBoost configuration expected numerical labels:

```text
0
1
```

The error was:

```text
ValueError:
Invalid classes inferred from unique values of `y`.
Expected: [0 1], got ['No' 'Yes']
```

The target variable can be encoded using `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(
    target_of_emp_att
)
```

This converts:

```text
No  → 0
Yes → 1
```

The target encoder should also be saved if XGBoost is used during deployment.

---

# ⚠️ Class Imbalance

Employee attrition is an imbalanced classification problem.

There are substantially more employees in the:

```text
No
```

class than in the:

```text
Yes
```

class.

As a result, a model can achieve high overall accuracy while still failing to identify employees who actually leave.

For example:

```text
Actual: Yes
Predicted: No
```

is a **False Negative (FN)**.

False negatives are particularly important from an HR perspective because the model failed to identify an employee who may have been at risk of leaving.

Therefore, the project emphasizes:

* Recall
* Precision
* F1 Score
* Confusion Matrix
* ROC-AUC
* PR-AUC
* False-negative analysis
* Business cost of prediction errors

---

# 🎯 Model Selection Strategy

Although Logistic Regression achieved the highest test accuracy among the models tested in the current experiment, its recall for the `Yes` class was only:

```text
15%
```

Therefore, selecting a model purely based on accuracy may not be appropriate for this business problem.

The main objective is:

> **Identify as many potential attrition cases as possible while maintaining an acceptable level of precision.**

Possible improvements include:

* Class weighting
* Hyperparameter tuning
* Stratified cross-validation
* Random Forest class balancing
* XGBoost class weighting
* SMOTE
* Threshold tuning
* Precision-Recall analysis

For example:

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

can give additional importance to the minority attrition class.

---

# 🧪 Prediction Workflow

The prediction workflow is:

```text
Employee Input
      ↓
Input Validation
      ↓
Data Preprocessing
      ↓
Categorical Encoding
      ↓
Feature Combination
      ↓
Trained ML Model
      ↓
Attrition Prediction
      ↓
Yes / No
```

### Example Employee

```text
Age: 37
Monthly Income: 2090
Job Satisfaction: 3
Years at Company: 0
OverTime: No
Num Companies Worked: 6
Department: Research & Development
Marital Status: Single
```

The categorical inputs are transformed using the saved encoders.

The resulting feature vector is then passed to the trained model:

```python
employee_attrition = decisionforemployeeattri.predict(
    combined_input
)[0]
```

The prediction can then be displayed as:

```text
Employee is likely to leave
```

or:

```text
Employee is likely to stay
```

---

# 💾 Saved Models and Preprocessing Objects

The project saves preprocessing objects and trained models using Joblib.

### One-Hot Encoder

```text
Models/onehotencoderforemp_attri.pkl
```

### Label Encoder

```text
Models/labelforemployeeattri.pkl
```

### Decision Tree Model

```text
Models/decisionforemployeeattri.pkl
```

These objects can be loaded during deployment:

```python
import joblib

model = joblib.load(
    "Models/decisionforemployeeattri.pkl"
)

ohe = joblib.load(
    "Models/onehotencoderforemp_attri.pkl"
)

label_encoder = joblib.load(
    "Models/labelforemployeeattri.pkl"
)
```

---

# 📁 Project Structure

A recommended GitHub project structure is:

```text
Employee-Attrition-and-Performance-Rating-Analysis/
│
├── Data/
│   ├── Employee-Attrition.csv
│   ├── Preprocessed_data.csv
│   ├── feature0femployeeattrion.csv
│   └── target0femployeeattrion.csv
│
├── Models/
│   ├── onehotencoderforemp_attri.pkl
│   ├── labelforemployeeattri.pkl
│   └── decisionforemployeeattri.pkl
│
├── notebooks/
│   └── Employee_Attrition_and_Performance_Analysis.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

> **Note:** If the project contains sensitive or proprietary employee information, raw employee data should not be committed to a public GitHub repository.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Seaborn
* Matplotlib
* Joblib
* Jupyter Notebook
* Streamlit

---

# 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd Employee-Attrition-and-Performance-Rating-Analysis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

If a Streamlit application is included, run:

```bash
streamlit run app.py
```

The application can accept employee information such as:

```text
Age
Monthly Income
Job Satisfaction
Years at Company
OverTime
Number of Companies Worked
Department
Marital Status
```

The application can then return the predicted attrition status.

Example:

```text
Prediction: Employee is likely to leave
```

or:

```text
Prediction: Employee is likely to stay
```

---

# 🔮 Future Improvements

The project can be further improved through the following approaches.

### 1. Hyperparameter Tuning

Use:

* GridSearchCV
* RandomizedSearchCV

to optimize model parameters.

### 2. Cross-Validation

Use Stratified K-Fold Cross-Validation to obtain more reliable estimates of model performance.

### 3. Class Imbalance Handling

Evaluate:

* `class_weight='balanced'`
* SMOTE
* Random under-sampling
* Cost-sensitive learning

### 4. Threshold Optimization

Instead of using the default probability threshold of `0.5`, optimize the threshold according to the business objective.

A lower threshold may increase recall and help identify more potential attrition cases.

### 5. Explainable AI

Use techniques such as:

* SHAP
* Feature Importance
* Partial Dependence Analysis

to explain why an employee is classified as having a higher attrition risk.

### 6. Employee Feedback and Sentiment Analysis

Future versions could incorporate:

* Employee surveys
* Feedback data
* Sentiment analysis
* Engagement scores

to improve attrition-risk prediction.

### 7. Model Monitoring

After deployment, monitor:

* Prediction performance
* Data drift
* Class distribution
* False negatives
* False positives
* Feature distribution changes

to ensure the model remains reliable over time.

---

# 📌 Key Findings

The analysis produced several important observations:

* Employee attrition can be analyzed and predicted using machine learning.
* Job satisfaction and work-life balance can be important indicators of employee turnover.
* Compensation and monthly income can influence employee attrition.
* Employees with shorter tenure may show higher attrition risk.
* Overtime can be an important factor associated with employee turnover.
* Categorical encoding is necessary for machine learning models that require numerical input.
* Outlier treatment can improve the quality of numerical features.
* Feature selection helps identify the most relevant variables.
* Logistic Regression achieved approximately **85% test accuracy** in the current experiment.
* Random Forest achieved approximately **83% test accuracy**.
* The Decision Tree achieved **100% training accuracy**, but its lower test performance indicates overfitting.
* Accuracy alone is not sufficient because employee attrition is an imbalanced classification problem.
* Recall for the `Yes` class is particularly important because false negatives represent employees who may leave but were not identified by the model.

---

# 📚 Learnings

Through this project, the following concepts were explored:

### Data Preprocessing

* Missing-value analysis
* Numerical transformations
* Categorical encoding
* Feature scaling
* Data preparation

### Outlier Handling

* IQR method
* Winsorization
* Log transformation
* Square root transformation

### Feature Selection

* Chi-Square Test
* ANOVA
* Mutual Information
* Recursive Feature Elimination
* Correlation analysis

### Feature Engineering

Created additional meaningful features to improve the representation of employee information.

### Exploratory Data Analysis

Used:

* Bar plots
* Histograms
* Box plots
* Correlation heatmaps

to identify patterns and relationships within employee data.

### Machine Learning

Compared:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

and evaluated their performance using multiple classification metrics.

### Model Performance

Learned that a high accuracy score does not necessarily mean a model is suitable for a business problem, especially when the target variable is imbalanced.

---

# ✅ Conclusion

The **Employee Attrition and Performance Rating Analysis with Prediction** project demonstrates how data analytics and machine learning can be used to understand employee behavior and predict potential attrition.

The project combines exploratory analysis, preprocessing, outlier handling, feature engineering, feature selection, machine learning, and model evaluation to create an end-to-end employee attrition analysis workflow.

The analysis shows that factors such as:

* Job satisfaction
* Work-life balance
* Monthly income
* Overtime
* Years at company
* Number of companies worked
* Department
* Marital status

can provide valuable information for understanding employee attrition.

The project also highlights the importance of selecting machine learning models based on the **actual business objective**, rather than relying only on accuracy.

The ultimate goal is to provide organizations with actionable insights that can help HR teams identify potential attrition risks early and develop appropriate employee-retention strategies.

> **Predict the risk. Understand the reason. Take action before valuable employees leave.**

---

# 👨‍💻 Author

**Karthikeyan**
