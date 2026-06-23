# Week 2 Classification Assignment

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Select useful columns
df = df[["survived", "pclass", "sex", "age", "fare"]]

# Remove missing values
df = df.dropna()

# Convert categorical values
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# Features and Target
X = df.drop("survived", axis=1)
y = df["survived"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy")
print(accuracy_score(y_test, pred_lr))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred_lr))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)

print("\nDecision Tree Accuracy")
print(accuracy_score(y_test, pred_dt))

print("\nKNN Accuracy")
print(accuracy_score(y_test, pred_knn))

# Decision Tree Tuning
for depth in [2, 3, 5, 7, 10]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print(
        "Max Depth:",
        depth,
        " Accuracy:",
        accuracy_score(y_test, prediction)
    )
