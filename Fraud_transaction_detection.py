import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
import glob

warnings.filterwarnings('ignore')

# Load all .pkl files in the data folder and concatenate
pkl_files = glob.glob('data/*.pkl')
df = pd.concat([pd.read_pickle(f) for f in pkl_files], ignore_index=True)

# Extract datatime features
df['TX_DAY'] = df['TX_DATETIME'].dt.day
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
df['TX_WEEKDAY'] = df['TX_DATETIME'].dt.weekday

# Rule 1 feature:Amount greater than 220
df['RULE1_HIGH_AMOUNT'] =(df['TX_AMOUNT'] > 220).astype(int)

# Customer level question
customer_stats = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg(['mean', 'count']).reset_index()
customer_stats.columns = ['CUSTOMER_ID', 'CUSTOMER_AVG_AMOUNT', 'CUSTOMER_TX_COUNT']
df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')

# Terminal-level features
terminal_stats = df.groupby('TERMINAL_ID')['TX_AMOUNT'].count().reset_index()
terminal_stats.columns = ['TERMINAL_ID', 'TERMINAL_TX_COUNT']
df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')

# feature selection
features=[
    'TX_AMOUNT',
    'TX_HOUR',
    'TX_WEEKDAY',
    'CUSTOMER_AVG_AMOUNT',
    'CUSTOMER_TX_COUNT',
    'TERMINAL_TX_COUNT'
]
X = df[features]
y = df['TX_FRAUD']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test =train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# Feature importance
importances = model.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, 'Fraud_detector_model.pkl')
