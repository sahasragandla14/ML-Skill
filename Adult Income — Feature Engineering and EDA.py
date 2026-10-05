# Adult Income - Feature Engineering and EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

df = pd.read_csv(
    url,
    names=columns,
    skipinitialspace=True
)

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Basic Information

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Replace ? with NaN

df = df.replace("?", np.nan)

print("\nMissing Values After Replacement:")
print(df.isnull().sum())

# 3. Remove Missing Values

df = df.dropna()

print("\nShape After Cleaning:", df.shape)

# 4. EDA

# Income distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="income", data=df)
plt.title("Income Distribution")
plt.show()

# Income by gender
plt.figure(figsize=(7, 5))
sns.countplot(x="sex", hue="income", data=df)
plt.title("Income by Gender")
plt.show()

# Income by education
plt.figure(figsize=(12, 6))
sns.countplot(
    y="education",
    hue="income",
    data=df
)
plt.title("Income by Education")
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    df["age"],
    bins=30,
    kde=True
)
plt.title("Age Distribution")
plt.show()

# Working hours
plt.figure(figsize=(8, 5))
sns.histplot(
    df["hours_per_week"],
    bins=30,
    kde=True
)
plt.title("Working Hours per Week")
plt.show()

# Age vs Income
plt.figure(figsize=(7, 5))
sns.boxplot(
    x="income",
    y="age",
    data=df
)
plt.title("Age vs Income")
plt.show()

# 5. Feature Engineering

# Create age groups

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 65, 100],
    labels=[
        "Young",
        "Adult",
        "Middle_Aged",
        "Senior",
        "Very_Senior"
    ]
)

# Create work hour category

df["work_hours_category"] = pd.cut(
    df["hours_per_week"],
    bins=[0, 30, 40, 60, 100],
    labels=[
        "Part_Time",
        "Full_Time",
        "Overtime",
        "Extreme"
    ]
)

print("\nFeature Engineered Data:")
print(df[[
    "age",
    "age_group",
    "hours_per_week",
    "work_hours_category"
]].head())

# 6. Encode Target

df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

# 7. Convert Categorical Features

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# 8. Split Data

X = df_encoded.drop("income", axis=1)
y = df_encoded["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 10. Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 11. Prediction

y_pred = model.predict(X_test)

# 12. Evaluation

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 13. Feature Importance

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features:")
print(feature_importance.head(10))

# 14. Plot Feature Importance

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")
plt.show()

# 15. Save Processed Dataset

df_encoded.to_csv(
    "adult_income_feature_engineered.csv",
    index=False
)

print("\nProcessed dataset saved as:")
print("adult_income_feature_engineered.csv")