---
### Contributors: Brian Waweru, Start-Date        : 08th May, 2025
---
# 0.1 : Working Libraries and Preliminaries
# Python Libraries
import pandas as pd
# sci-kit libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# SMOTE 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#### Importing the `dataset`:
# dataset location
file = "churn_in_telecoms_dataset.csv"

# creating a dataframe
df = pd.read_csv(file)

# shape of the dataset
print(df.shape)

# snapshot
df.head(3)
# 0.2 : Feature Engineering and Preprocessing
# General information of each column
# Including entry types
df.info()
# drop the 'phone number' column
df = df.drop(columns='phone number')
# columns
print(df.columns)
# finding out entries of the 'churn' column
print(f"The Unique entries in teh 'churn' column are: {df.churn.unique()}")
# >>> array([False,  True])
df['churn'] = df['churn'].astype(int)
# Convert boolean values in the 'churn' column to integers: False → 0, True → 1
print(f"After conversion, the new converted entries for ease of classfication are: {df.churn.unique()}")
# Column Description
df.describe()
# Categorical Columns
missing_columns = [col for col in df.columns if col not in df.describe().columns]
print("Columns missing from df.describe():", missing_columns)
# printing them out
df[['state', 'international plan', 'voice mail plan']].head(3)
# Shape of the datset
print(f"The Shape of the dataset is: {df.shape}")
# 'Area Code' column
print(f"The 'area code has only 3 entries: {df['area code'].unique()}") # 3 
# 1.0 : Overview
## 1.1: Overview-Introduction: Customer Churn Prediction for SyriaTel
This project aims to predict customer churn for `SyriaTel`, a telecommunications company, using a sample of their historical customer data. By building a binary classification model since the customer either churns '1' or does not '0', we shall aim to identify patterns and factors that influence whether a customer will leave the company. The predictive model will assist the company in targeting at-risk customers with `retention strategies`, thereby reducing customer attrition and preserving revenue.

The overall project pipeline consists of:

1. **Business Understanding**: Understanding churn's impact on SyriaTel’s business.

2. **Data Understanding and Preparation**: Exploring the structure and distribution of data. Cleaning, transforming, and encoding the dataset.

3. **Exploratory Data Analysis (EDA)**: Finding patterns and feature relationships with churn.

4. **Model Building**: Training and tuning classifiers such as Logistic Regression, Decision Trees or Random Forests.

5. **Evaluation**: Measuring performance using metrics like accuracy, precision, recall, F1-score, and AUC as well as ROC.

6. **Interpretation**: Identifying key drivers of churn.

7. **Recommendations and Actionable Insights**: Informing business interventions to reduce churn. Provide recommendations for customer retention based on analytical findings.
## 1.2: Project-Objectives

Here are the key Objectives in this project:-

1. **Build a Predictive Model for Churn**  
2. **Identify Key Drivers of Churn**  
3. **Improve Churn Prediction Accuracy**  
4. **Support Retention Strategy Development**  
5. **Then Communicate Findings Clearly**: Present model insights and business recommendations in a format accessible to both technical and non-technical stakeholders.

However, if time-allows it is also important to explore these other secondary objectives

1. **Understand Customer Behavior**  
2. **Segment At-Risk Customers**  
3. **Evaluate Cost-Benefit Trade-offs**  
   Analyze which churn-prone customers are most valuable to retain based on their potential lifetime value.
4. **Develop a Repeatable ML Pipeline**  
   Build a clean and modular workflow that can be reused with updated customer data in the future.
# 2.0 : Business and Data Understanding
## 2.1: Business Understanding 

Customer churn is a critical business challenge for telco compianes such as SyriaTel. In a highly competitive and saturated market, retaining existing customers is often more cost-effective than acquiring new ones. Churn not only impacts immediate revenue but also affects long-term customer lifetime value, brand loyalty, and operational efficiency. Understanding why customers leave — and more importantly, identifying who is likely to leave — can empower SyriaTel to take timely, targeted actions. These may include `personalized marketing campaigns`, `service improvements`, or `tailored retention offers`. 

The core business goal of this project is to reduce churn by building a predictive model that accurately flags at-risk customers. This enables SyriaTel to shift from reactive to proactive customer retention, thereby reducing revenue loss and enhancing customer satisfaction.

The project aligns with SyriaTel's strategic priorities:

1. `Preserving revenue` by minimizing customer loss.

2. `Improving customer loyalty` through better engagement.

3. `Increasing the return on investment (ROI)` of marketing and support efforts.

4. `Leveraging data` to drive smarter, faster business decisions.

Ultimately, this project supports SyriaTel’s mission to build lasting customer relationships in a competitive telecom landscape.
## 2.2: Data Understanding

The dataset provided by SyriaTel consists of over 3300 customer records and 21 features, each capturing various aspects of a customer's interaction with the company's service. The target variable is `churn`, which indicates whether a customer has discontinued service or not. Understanding the composition and behavior of these column-features is critical in helping us build an effective churn prediction model.

1. Several features describe customer demographics and account information, such as `state`, `area code`, and `account length`. While these may not directly cause churn, they can help identify regional trends or the effect of customer tenure on loyalty.

2. Other features capture service plans (`international plan`, `voice mail plan`), indicate whether a customer is subscribed to specific services. These features may influence customer satisfaction and costs, potentially affecting their decision to stay or leave.

3. A significant portion of the dataset focuses on usage behavior, including `call minutes`, `number of calls`, and `charges` during the day, evening, night, and for international calls. These metrics are split into separate fields for minutes, calls, and charges. This could allow an examinantion of customer engagement and how it relates to churn. However, since charges are typically derived from minutes, some of these columns may be redundant.

4. The dataset also includes features such as the number of `customer service calls`, which can be a strong indicator of dissatisfaction—customers who contact support frequently may be more likely to churn.

5. Importantly, the dataset is clean, with `no missing values`, and the data types are appropriate for analysis—numerical for continuous variables and object or boolean for categorical ones. However, some preprocessing will be necessary, including encoding categorical variables and dropping non-informative columns like phone number, as done previously, which acts only as an identifier.

Through a Thorough EDA, we aim to understand the relationships between these features and the likelihood of churn. Identifying patterns, such as whether certain service plans correlate with higher churn, or whether customers with higher international usage are more likely to leave, will help us build a predictive model and generate actionable business insights.
# 3: Data Preparation 
### 3.1 : Handle Categorical Variables
# Categories i.e. classify the values in the 'international plan' as either 1 or 0
df['international plan'] = df['international plan'].map({'yes': 1, 'no': 0})
# Categories i.e. classify the values in the 'voice mail plan' as either 1 or 0
df['voice mail plan'] = df['voice mail plan'].map({'yes': 1, 'no': 0})
# print feedback that it's done
print('Success: It is done!')
### 3.2 : Drop Irrelevant or Redundant Columns

As mentioned earlier in the overview and business understanding, features like `total day charge` might be redundant if `total day minutes` already provides similar information. You might choose to drop one.

- `note:` The feature `phone number` had been dropped already. This is because it only serves as a customer identifier.
- `note:` As noted earlier, there are no missing values. See section under `df.info()`.
# redundant columns
redundant_columns = ['total day charge', 'total eve charge', 'total night charge', 'total intl charge']
# dropping them
df.drop(redundant_columns, axis=1, inplace=True)
# print feedback that it's done
print('Success: It is done!')
### 3.3 : Standardization 

Now, we transform features in the datasset i.e. `total day minutes`, `number vmail messages`, `total eve minutes` so that they have a common scale. 
# Create a scaler object
scaler = StandardScaler()

# Choice of columns to standadise
cols_to_standardize = ['total day minutes', 'number vmail messages', 'total eve minutes']

# Apply standardization to relevant columns
df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

# print feedback that it's done
print('Success: It is done!')
### 3.4 : Feature Engineering

- Since the dataset is riddled with `minutes`, suppose we have `Total Call Usage` i.e the sum of `total day minutes`, `total eve minutes`, `total night minutes`, and `total intl minutes`. 

- Also, we can have `Average Call Duration` i.e. for Average of `day`, `evening`, `night`, and `international minutes`.
# TOTAL CALL USAGE:
df['total minutes'] = df['total day minutes'] + df['total eve minutes'] + df['total night minutes'] + df['total intl minutes']
# AVERAGE CALL DURATION:
df['average call duration'] = df[['total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes']].mean(axis=1)
# print feedback that it's done
print('Success: It is done!')
### 3.5 : Choosing `Target` and `Feature` column(s)

It is obvious that the choice of our Target column is `churn` while the rest are automatically the `Features`.

Now, Splitting the Data into `Training` and `Test` dataSets
# Target Feature ## Dependent Feature
y = df.churn
# Other Features ## independent Features
X = df.drop('churn', axis=1)
# split-test-code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# print feedback that it's done
print('Success: It is done!')
### 3.6 : Check for Class Imbalance

Let's ensure that the model doesn’t learn misleading patterns — especially because we have binary classification problem.
# first, let's check class distribution
print(f"The Value counts are: \n{df.churn.value_counts()}", end = '\n\n')
# The proportions are:
print('Essentially, that is:-', end = '\n')
print(f"{df['churn'].value_counts(normalize=True)}", end = '\n')
# Inference and Conclusion 
churn_perc = round(df['churn'].value_counts(normalize=True)[1] * 100, 2)
# print the result
print(' ')
print(f"INFERENCE: So, only {churn_perc}% of the customers churn — this shows class imbalance.")
#### `Inference`: 

This class imbalance refers to the fact that one class (non-churn) significantly outweighs the other `churning` group. This imbalance can affect the performance of machine learning model. They may become biased toward predicting the majority class, the `non-churning`, which could result in misleading accuracy scores.
# RE_DONE
X_encoded = pd.get_dummies(X, drop_first=True)  # drop_first avoids dummy trap

# 
le = LabelEncoder()
X['state'] = le.fit_transform(X['state'])


# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)
# Step 1: Encode categorical variables (e.g., 'State', 'Gender', etc.)
X_encoded = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, stratify=y, test_size=0.2, random_state=42
)

# Step 3: Apply SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 4: Train a Random Forest Classifier with class weights to handle class imbalance
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train_res, y_train_res)

# Step 5: Make predictions on the test set
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Step 6: Evaluate the model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# 4: Modeling
#### 4.1 : Logistic Regression
Let us start with a `simple Logistic Regression` model, before we try others like Random Forest.
model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Use class_weight if you didn't use SMOTE
model.fit(X_train, y_train)
#### 4.2 : Others
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# 1. Prepare features and target
X = df.drop('churn', axis=1)
y = df['churn']

# 2. One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Feature Scaling (optional, but needed for SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=6, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

# 6. Train and evaluate each model
for name, model in models.items():
    print(f"\n----- {name} -----")
    if name in ["SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 5: Evaluation
# 6: Conclusions and Recommendations
