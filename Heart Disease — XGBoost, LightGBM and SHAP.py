import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"

df = pd.read_csv(url)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- XGBoost ----------------

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("XGBoost Accuracy:",
      accuracy_score(y_test, xgb_pred))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))


# ---------------- LightGBM ----------------

lgb_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbosity=-1
)

lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)

print("\nLightGBM Accuracy:",
      accuracy_score(y_test, lgb_pred))

print("\nLightGBM Classification Report:")
print(classification_report(y_test, lgb_pred))


# ---------------- SHAP ----------------

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.title("SHAP Feature Importance - XGBoost")
plt.show()

# SHAP bar plot
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")
plt.show()