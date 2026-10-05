import pandas as pd
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load diabetes dataset
data = load_diabetes(as_frame=True)

X = data.data
y_continuous = data.target

print("Dataset:")
print(X.head())

# Convert continuous target into 3 severity classes
y = pd.qcut(
    y_continuous,
    q=3,
    labels=["Low", "Medium", "High"]
)

print("\nClass Distribution:")
print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Multinomial Logistic Regression
model = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))