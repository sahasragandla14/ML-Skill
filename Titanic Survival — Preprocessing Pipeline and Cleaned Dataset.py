# Titanic Survival - Preprocessing Pipeline

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset

df = sns.load_dataset("titanic")

print("Original Dataset:")
print(df.head())

# 2. Select Useful Features

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]

# 3. Define Column Types

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

# 4. Numeric Pipeline

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 5. Categorical Pipeline

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Combine Pipelines

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 7. Complete ML Pipeline

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

# 8. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train Model

model_pipeline.fit(X_train, y_train)

# 10. Prediction

y_pred = model_pipeline.predict(X_test)

# 11. Evaluation

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Create Cleaned Dataset

cleaned_X = preprocessor.fit_transform(X)

feature_names = preprocessor.get_feature_names_out()

cleaned_df = pd.DataFrame(
    cleaned_X.toarray() if hasattr(cleaned_X, "toarray") else cleaned_X,
    columns=feature_names
)

cleaned_df["survived"] = y.values

print("\nCleaned Dataset:")
print(cleaned_df.head())

print("\nCleaned Dataset Shape:", cleaned_df.shape)

# 13. Save Cleaned Dataset

cleaned_df.to_csv(
    "titanic_cleaned_dataset.csv",
    index=False
)

print("\nCleaned dataset saved as titanic_cleaned_dataset.csv")