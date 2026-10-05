import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url, sep=";")

print(df.head())
print("\nShape:", df.shape)

# Create quality classes
def quality_class(x):
    if x <= 4:
        return "Low"
    elif x <= 6:
        return "Medium"
    else:
        return "High"

df["quality_class"] = df["quality"].apply(quality_class)

# Distribution
sns.countplot(
    x="quality_class",
    data=df
)

plt.title("Wine Quality Classes")
plt.show()

# Features and target
X = df.drop(
    ["quality", "quality_class"],
    axis=1
)

y = df["quality_class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- Decision Tree ----------------

dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))

print("\nDecision Tree Report:")
print(classification_report(y_test, dt_pred))


# ---------------- Random Forest ----------------

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred))


# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Wine Quality - Random Forest Feature Importance")
plt.show()