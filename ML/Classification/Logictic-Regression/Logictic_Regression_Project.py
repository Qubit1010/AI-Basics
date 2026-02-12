"""
Logistic Regression Project: Customer Churn Prediction
Complete example with data generation, training, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, roc_auc_score)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. GENERATE SYNTHETIC CUSTOMER DATA
# ============================================================================
print("=" * 60)
print("CUSTOMER CHURN PREDICTION - LOGISTIC REGRESSION")
print("=" * 60)

n_samples = 1000

# Generate features
data = {
    'age': np.random.randint(18, 70, n_samples),
    'monthly_charges': np.random.uniform(20, 150, n_samples),
    'tenure_months': np.random.randint(1, 72, n_samples),
    'support_calls': np.random.randint(0, 10, n_samples),
    'contract_type': np.random.choice([0, 1, 2], n_samples)  # 0: Month, 1: Year, 2: 2-Year
}

df = pd.DataFrame(data)

# Create target variable (churn) based on realistic patterns
# Higher churn probability if: low tenure, high support calls, month-to-month contract
churn_probability = (
    0.1 +  # Base probability
    (df['tenure_months'] < 12) * 0.3 +  # New customers more likely to churn
    (df['support_calls'] > 5) * 0.3 +    # Unhappy customers
    (df['contract_type'] == 0) * 0.25 +  # Month-to-month contracts
    (df['monthly_charges'] > 100) * 0.15  # High charges
)

# Add some randomness
df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nChurn Distribution:")
print(df['churn'].value_counts())
print(f"Churn Rate: {df['churn'].mean():.2%}")

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

# Separate features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (important for logistic regression!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# 3. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ============================================================================
# 5. EVALUATE MODEL
# ============================================================================

print("\nMODEL PERFORMANCE:")
print("-" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# --------------------- Plot 1: Feature Importance ---------------------
ax1 = plt.subplot(3, 3, 1)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
ax1.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
ax1.set_xlabel('Coefficient Value')
ax1.set_title('Feature Importance (Coefficients)', fontweight='bold', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

# --------------------- Plot 2: Confusion Matrix ---------------------
ax2 = plt.subplot(3, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
ax2.set_xticklabels(['No Churn', 'Churn'])
ax2.set_yticklabels(['No Churn', 'Churn'])

# --------------------- Plot 3: ROC Curve ---------------------
ax3 = plt.subplot(3, 3, 3)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve', fontweight='bold', fontsize=12)
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# --------------------- Plot 4: Prediction Distribution ---------------------
ax4 = plt.subplot(3, 3, 4)
ax4.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='No Churn (Actual)', color='blue')
ax4.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Churn (Actual)', color='red')
ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Boundary')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Frequency')
ax4.set_title('Prediction Probability Distribution', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# --------------------- Plot 5: Tenure vs Churn ---------------------
ax5 = plt.subplot(3, 3, 5)
churn_data = df.copy()
churn_data['churn_label'] = churn_data['churn'].map({0: 'No Churn', 1: 'Churn'})
sns.boxplot(data=churn_data, x='churn_label', y='tenure_months', hue='churn_label',
            ax=ax5, palette='Set2', legend=False)
ax5.set_xlabel('Customer Status')
ax5.set_ylabel('Tenure (Months)')
ax5.set_title('Tenure Distribution by Churn Status', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3, axis='y')

# --------------------- Plot 6: Monthly Charges vs Churn ---------------------
ax6 = plt.subplot(3, 3, 6)
sns.violinplot(data=churn_data, x='churn_label', y='monthly_charges', hue='churn_label',
               ax=ax6, palette='Set1', legend=False)
ax6.set_xlabel('Customer Status')
ax6.set_ylabel('Monthly Charges ($)')
ax6.set_title('Monthly Charges by Churn Status', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3, axis='y')

# --------------------- Plot 7: Support Calls Impact ---------------------
ax7 = plt.subplot(3, 3, 7)
support_churn = df.groupby('support_calls')['churn'].mean()
ax7.bar(support_churn.index, support_churn.values, color='coral', edgecolor='black')
ax7.set_xlabel('Number of Support Calls')
ax7.set_ylabel('Churn Rate')
ax7.set_title('Churn Rate by Support Calls', fontweight='bold', fontsize=12)
ax7.grid(True, alpha=0.3, axis='y')

# --------------------- Plot 8: Contract Type Impact ---------------------
ax8 = plt.subplot(3, 3, 8)
contract_labels = {0: 'Monthly', 1: 'Yearly', 2: '2-Year'}
contract_churn = df.groupby('contract_type')['churn'].mean()
contract_names = [contract_labels[i] for i in contract_churn.index]
ax8.bar(contract_names, contract_churn.values, color=['lightcoral', 'lightblue', 'lightgreen'],
        edgecolor='black')
ax8.set_xlabel('Contract Type')
ax8.set_ylabel('Churn Rate')
ax8.set_title('Churn Rate by Contract Type', fontweight='bold', fontsize=12)
ax8.grid(True, alpha=0.3, axis='y')

# --------------------- Plot 9: Age Distribution ---------------------
ax9 = plt.subplot(3, 3, 9)
ax9.hist(df[df['churn'] == 0]['age'], bins=20, alpha=0.6, label='No Churn', color='skyblue')
ax9.hist(df[df['churn'] == 1]['age'], bins=20, alpha=0.6, label='Churn', color='salmon')
ax9.set_xlabel('Age')
ax9.set_ylabel('Frequency')
ax9.set_title('Age Distribution by Churn Status', fontweight='bold', fontsize=12)
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved!")

# ============================================================================
# 7. FEATURE INSIGHTS
# ============================================================================

print("\n" + "=" * 60)
print("FEATURE INSIGHTS")
print("=" * 60)

for feature, coef in zip(X.columns, model.coef_[0]):
    direction = "increases" if coef > 0 else "decreases"
    print(f"{feature:20s}: {coef:+.4f} - {direction} churn probability")

# ============================================================================
# 8. EXAMPLE PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)

# Create sample customers
sample_customers = pd.DataFrame({
    'age': [25, 45, 60],
    'monthly_charges': [150, 50, 75],
    'tenure_months': [3, 36, 48],
    'support_calls': [8, 1, 2],
    'contract_type': [0, 2, 1]
})

sample_scaled = scaler.transform(sample_customers)
sample_predictions = model.predict_proba(sample_scaled)[:, 1]

print("\nCustomer Profiles and Churn Risk:")
for i, prob in enumerate(sample_predictions):
    print(f"\nCustomer {i+1}:")
    print(f"  Age: {sample_customers.iloc[i]['age']}")
    print(f"  Monthly Charges: ${sample_customers.iloc[i]['monthly_charges']:.2f}")
    print(f"  Tenure: {sample_customers.iloc[i]['tenure_months']} months")
    print(f"  Support Calls: {sample_customers.iloc[i]['support_calls']}")
    print(f"  Contract: {['Monthly', 'Yearly', '2-Year'][sample_customers.iloc[i]['contract_type']]}")
    print(f"  → Churn Probability: {prob:.1%} - {'HIGH RISK' if prob > 0.5 else 'LOW RISK'}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)

plt.show()