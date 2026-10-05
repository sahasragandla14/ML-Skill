# Titanic Survival - EDA and ML Lifecycle Mapping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
df = sns.load_dataset("titanic")

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Understand Dataset
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# 3. EDA

# Survival count
sns.countplot(x="survived", data=df)
plt.title("Titanic Survival Count")
plt.show()

# Survival by gender
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by passenger class
sns.countplot(x="pclass", hue="survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution
sns.histplot(df["age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Age vs Survival
sns.boxplot(x="survived", y="age", data=df)
plt.title("Age vs Survival")
plt.show()

# Correlation
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4. Data Preprocessing

data = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()

# Fill missing values
data["age"] = data["age"].fillna(data["age"].median())
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])

# Convert categorical variables
data = pd.get_dummies(data, columns=["sex", "embarked"], drop_first=True)

print("\nProcessed Data:")
print(data.head())

# 5. Split Features and Target

X = data.drop("survived", axis=1)
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Model Training

model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Prediction

y_pred = model.predict(X_test)

# 9. Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Confusion Matrix Visualization

plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nML Lifecycle:")
print("1. Problem Definition")
print("2. Data Collection")
print("3. Data Understanding")
print("4. Exploratory Data Analysis")
print("5. Data Preprocessing")
print("6. Feature Engineering")
print("7. Model Training")
print("8. Model Evaluation")
print("9. Deployment/Monitoring")