# 🕵️‍♂️ Fraud Transaction Detection

This project uses a simulated transaction dataset to detect fraudulent transactions using machine learning.

---

## 📌 Project Overview

The model classifies transactions as **fraudulent (1)** or **legitimate (0)** based on:

- Transaction amount
- Customer spending patterns
- Terminal activity
- Time-based features

---

## 💡 Fraud Scenarios in the Dataset

1. Transactions over ₹220 are marked as fraud
2. Random terminals are fraud hotspots for 28 days
3. Random customers are targeted for fraud (1/3 of their txns are inflated for 14 days)

---

## 🧠 ML Model Used

- `RandomForestClassifier` from scikit-learn
- Class imbalance handled via `class_weight='balanced'`
- Feature importance plotted for interpretability

---

## 🛠 Features Used

- `TX_AMOUNT`, `TX_HOUR`, `TX_WEEKDAY`
- Customer average amount & transaction count
- Terminal transaction count

---

## 📈 Model Performance

- Precision, Recall, F1-Score
- ROC AUC Score: ~0.73
- Visualized feature importances using Seaborn

---

## 🗃 File Structure
---

## ▶️ How to Run

1. Clone the repo  
2. Install requirements  
3. Run the script:

```bash
python Fraud_transaction_detection.py
