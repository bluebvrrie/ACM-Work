# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

import pickle

# 2. LOAD DATA

df = pd.read_csv("creditcard.csv")

print(df.head())
print(df.info())
df = df.sample(50000, random_state=42)

# 3. EDA (UNDERSTAND DATA)

print(df['Class'].value_counts(normalize=True) * 100)
# Class distribution
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud")
plt.show()

# Correlation heatmap (optional)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4. FEATURE SELECTION

X = df.drop('Class', axis=1)
y = df['Class']

# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_train, y_train = smote.fit_resample(X_train, y_train)

# 6. BUILD PIPELINE

pipeline = Pipeline([
    ('scaler', StandardScaler()),   # scaling
    ('svm', SVC(probability=True))  # model
])

# 7. HYPERPARAMETER TUNING

param_grid = {
    'svm__C': [ 1],
    'svm__kernel': ['rbf'],
    'svm__gamma': ['scale']
}

grid = GridSearchCV(pipeline, param_grid, cv=2, scoring='f1', verbose=2)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# 8. EVALUATION

y_pred = grid.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ROC-AUC Score
y_prob = grid.predict_proba(X_test)[:,1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.show()

# 9. SAVE MODEL

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print("Model saved successfully!")