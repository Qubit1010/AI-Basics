"""
CUSTOMER CHURN PREDICTION PROJECT
==================================
A simple machine learning project using 3 models to predict customer churn

Models Used:
1. Logistic Regression
2. Random Forest
3. Decision Tree

Features:
- Generate realistic customer data
- Train and compare 3 models
- Visualize results
- Make predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

print("=" * 80)
print("CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 80)

# ============================================================================
# 1. GENERATE CUSTOMER DATA
# ============================================================================
print("\n" + "=" * 80)
print("1. GENERATING CUSTOMER DATA")
print("=" * 80)

np.random.seed(42)

# Generate 1000 customers
n_customers = 1000

# Customer features
age = np.random.randint(18, 70, n_customers)
monthly_charges = np.random.uniform(20, 150, n_customers)
total_charges = monthly_charges * np.random.randint(1, 72, n_customers)  # 1-72 months
tenure_months = np.random.randint(1, 72, n_customers)
contract_type = np.random.choice([0, 1, 2], n_customers)  # 0=Month-to-month, 1=One year, 2=Two year
support_calls = np.random.poisson(2, n_customers)  # Average 2 support calls
data_usage_gb = np.random.uniform(1, 50, n_customers)

# Generate churn (target variable)
# Customers are more likely to churn if:
# - Young age, high monthly charges, short tenure, month-to-month contract, many support calls

churn_probability = (
        0.15 +  # Base probability
        (age < 30) * 0.15 +  # Young customers more likely to churn
        (monthly_charges > 100) * 0.2 +  # High charges increase churn
        (tenure_months < 12) * 0.25 +  # New customers more likely to churn
        (contract_type == 0) * 0.2 +  # Month-to-month contract increases churn
        (support_calls > 3) * 0.15  # Many support calls increase churn
)

# Add some randomness
churn_probability = np.clip(churn_probability + np.random.normal(0, 0.1, n_customers), 0, 1)
churn = (np.random.random(n_customers) < churn_probability).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Monthly_Charges': monthly_charges,
    'Total_Charges': total_charges,
    'Tenure_Months': tenure_months,
    'Contract_Type': contract_type,
    'Support_Calls': support_calls,
    'Data_Usage_GB': data_usage_gb,
    'Churn': churn
})

# Add contract type labels
contract_labels = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
df['Contract_Label'] = df['Contract_Type'].map(contract_labels)

print(f"\n✓ Generated data for {n_customers} customers")
print(f"\nFirst 5 customers:")
print(df[['Age', 'Monthly_Charges', 'Tenure_Months', 'Contract_Label', 'Churn']].head())

print(f"\nChurn Distribution:")
churn_counts = df['Churn'].value_counts()
print(f"  No Churn (0): {churn_counts[0]} customers ({churn_counts[0] / n_customers * 100:.1f}%)")
print(f"  Churned (1):  {churn_counts[1]} customers ({churn_counts[1] / n_customers * 100:.1f}%)")

# Save data
df.to_csv('customer_data.csv', index=False)
print(f"\n✓ Data saved to: customer_data.csv")

# ============================================================================
# 2. DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA ANALYSIS")
print("=" * 80)

print("\n📊 Statistical Summary:")
print(df.describe())

print("\n📈 Churn Rate by Contract Type:")
churn_by_contract = df.groupby('Contract_Label')['Churn'].agg(['sum', 'count', 'mean'])
churn_by_contract.columns = ['Churned', 'Total', 'Churn_Rate']
print(churn_by_contract)

# ============================================================================
# 3. PREPARE DATA FOR MACHINE LEARNING
# ============================================================================
print("\n" + "=" * 80)
print("3. PREPARING DATA")
print("=" * 80)

# Features (X) and Target (y)
X = df[['Age', 'Monthly_Charges', 'Total_Charges', 'Tenure_Months',
        'Contract_Type', 'Support_Calls', 'Data_Usage_GB']]
y = df['Churn']

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature Scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled using StandardScaler")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n" + "=" * 80)
print("4. TRAINING MODELS")
print("=" * 80)

# Dictionary to store results
results = {}

# MODEL 1: Logistic Regression
print("\n--- Model 1: Logistic Regression ---")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
results['Logistic Regression'] = {
    'model': lr_model,
    'predictions': y_pred_lr,
    'accuracy': accuracy_lr
}
print(f"✓ Trained successfully")
print(f"  Accuracy: {accuracy_lr:.4f} ({accuracy_lr * 100:.2f}%)")

# MODEL 2: Random Forest
print("\n--- Model 2: Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
results['Random Forest'] = {
    'model': rf_model,
    'predictions': y_pred_rf,
    'accuracy': accuracy_rf
}
print(f"✓ Trained successfully")
print(f"  Accuracy: {accuracy_rf:.4f} ({accuracy_rf * 100:.2f}%)")

# MODEL 3: Decision Tree
print("\n--- Model 3: Decision Tree ---")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
results['Decision Tree'] = {
    'model': dt_model,
    'predictions': y_pred_dt,
    'accuracy': accuracy_dt
}
print(f"✓ Trained successfully")
print(f"  Accuracy: {accuracy_dt:.4f} ({accuracy_dt * 100:.2f}%)")

# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("5. MODEL COMPARISON")
print("=" * 80)

print("\n🏆 Model Performance Ranking:")
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

for rank, (name, data) in enumerate(sorted_results, 1):
    print(f"{rank}. {name:<25} Accuracy: {data['accuracy']:.4f} ({data['accuracy'] * 100:.2f}%)")

# Best model
best_model_name = sorted_results[0][0]
best_model_data = sorted_results[0][1]
print(f"\n🥇 Best Model: {best_model_name}")

# ============================================================================
# 6. DETAILED EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("6. DETAILED EVALUATION (Best Model)")
print("=" * 80)

print(f"\nModel: {best_model_name}")
print(f"\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, best_model_data['predictions'])
print(cm)
print(f"\nInterpretation:")
print(f"  True Negatives (Correctly predicted no churn): {cm[0, 0]}")
print(f"  False Positives (Incorrectly predicted churn): {cm[0, 1]}")
print(f"  False Negatives (Incorrectly predicted no churn): {cm[1, 0]}")
print(f"  True Positives (Correctly predicted churn): {cm[1, 1]}")

print(f"\n--- Classification Report ---")
print(classification_report(y_test, best_model_data['predictions'],
                            target_names=['No Churn', 'Churned']))

# ============================================================================
# 7. FEATURE IMPORTANCE (for Random Forest)
# ============================================================================
print("\n" + "=" * 80)
print("7. FEATURE IMPORTANCE")
print("=" * 80)

if best_model_name == 'Random Forest':
    importances = best_model_data['model'].feature_importances_
    feature_names = X.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\n📊 Feature Importance Ranking:")
    print(feature_importance_df.to_string(index=False))

    print(f"\n💡 Most Important Feature: {feature_importance_df.iloc[0]['Feature']}")

# ============================================================================
# 8. MAKE PREDICTIONS FOR NEW CUSTOMERS
# ============================================================================
print("\n" + "=" * 80)
print("8. MAKING PREDICTIONS FOR NEW CUSTOMERS")
print("=" * 80)

# Create 3 sample new customers
new_customers = pd.DataFrame({
    'Age': [25, 45, 60],
    'Monthly_Charges': [120, 50, 80],
    'Total_Charges': [1200, 3000, 5760],
    'Tenure_Months': [10, 60, 72],
    'Contract_Type': [0, 2, 2],  # Month-to-month, Two year, Two year
    'Support_Calls': [5, 1, 0],
    'Data_Usage_GB': [30, 15, 10]
})

print("\n🆕 New Customers to Predict:")
print(new_customers)

# Scale features
new_customers_scaled = scaler.transform(new_customers)

# Make predictions with best model
if best_model_name == 'Logistic Regression':
    predictions = best_model_data['model'].predict(new_customers_scaled)
    probabilities = best_model_data['model'].predict_proba(new_customers_scaled)
else:
    predictions = best_model_data['model'].predict(new_customers)
    probabilities = best_model_data['model'].predict_proba(new_customers)

print(f"\n🔮 Predictions using {best_model_name}:")
for i in range(len(new_customers)):
    churn_label = "WILL CHURN" if predictions[i] == 1 else "Will Stay"
    churn_prob = probabilities[i][1] * 100
    print(f"  Customer {i + 1}: {churn_label:<15} (Churn probability: {churn_prob:.1f}%)")

# ============================================================================
# 9. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("9. CREATING VISUALIZATIONS")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')

# Visualization 1: Model Comparison
print("\n📊 Creating model comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))

models = [name for name, _ in sorted_results]
accuracies = [data['accuracy'] for _, data in sorted_results]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Churn Prediction Model Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
            f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('churn_viz_1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: churn_viz_1_model_comparison.png")

# Visualization 2: Confusion Matrix
print("\n📊 Creating confusion matrix heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Churn', 'Churned'],
            yticklabels=['No Churn', 'Churned'],
            annot_kws={'size': 16, 'weight': 'bold'})

ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('churn_viz_2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: churn_viz_2_confusion_matrix.png")

# Visualization 3: Feature Importance
if best_model_name == 'Random Forest':
    print("\n📊 Creating feature importance chart...")
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_idx = feature_importance_df['Importance'].argsort()
    pos = np.arange(len(sorted_idx))

    ax.barh(pos, feature_importance_df['Importance'].iloc[sorted_idx],
            color='skyblue', edgecolor='navy', linewidth=1.5)
    ax.set_yticks(pos)
    ax.set_yticklabels(feature_importance_df['Feature'].iloc[sorted_idx])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance for Churn Prediction', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('churn_viz_3_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: churn_viz_3_feature_importance.png")

# Visualization 4: Churn Distribution
print("\n📊 Creating churn distribution chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Overall churn distribution
churn_counts = df['Churn'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c']
ax1.pie(churn_counts, labels=['No Churn', 'Churned'], autopct='%1.1f%%',
        colors=colors_pie, startangle=90, explode=(0.05, 0.05))
ax1.set_title('Overall Churn Distribution', fontsize=14, fontweight='bold')

# Churn by contract type
churn_contract = df.groupby(['Contract_Label', 'Churn']).size().unstack(fill_value=0)
churn_contract.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'], alpha=0.8)
ax2.set_title('Churn by Contract Type', fontsize=14, fontweight='bold')
ax2.set_xlabel('Contract Type', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax2.legend(['No Churn', 'Churned'], loc='upper right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('churn_viz_4_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: churn_viz_4_distribution.png")

# Visualization 5: Customer Insights
print("\n📊 Creating customer insights dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age vs Churn
axes[0, 0].scatter(df[df['Churn'] == 0]['Age'],
                   df[df['Churn'] == 0]['Monthly_Charges'],
                   alpha=0.5, s=30, c='green', label='No Churn')
axes[0, 0].scatter(df[df['Churn'] == 1]['Age'],
                   df[df['Churn'] == 1]['Monthly_Charges'],
                   alpha=0.5, s=30, c='red', label='Churned')
axes[0, 0].set_xlabel('Age', fontweight='bold')
axes[0, 0].set_ylabel('Monthly Charges', fontweight='bold')
axes[0, 0].set_title('Age vs Monthly Charges', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Tenure vs Churn
axes[0, 1].hist([df[df['Churn'] == 0]['Tenure_Months'],
                 df[df['Churn'] == 1]['Tenure_Months']],
                bins=20, label=['No Churn', 'Churned'],
                color=['green', 'red'], alpha=0.6)
axes[0, 1].set_xlabel('Tenure (Months)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Tenure Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Support Calls vs Churn
support_churn = df.groupby(['Support_Calls', 'Churn']).size().unstack(fill_value=0)
support_churn.plot(kind='bar', ax=axes[1, 0], color=['green', 'red'], alpha=0.6)
axes[1, 0].set_xlabel('Number of Support Calls', fontweight='bold')
axes[1, 0].set_ylabel('Number of Customers', fontweight='bold')
axes[1, 0].set_title('Support Calls vs Churn', fontweight='bold')
axes[1, 0].legend(['No Churn', 'Churned'])
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
axes[1, 0].grid(axis='y', alpha=0.3)

# Monthly Charges Distribution
axes[1, 1].hist([df[df['Churn'] == 0]['Monthly_Charges'],
                 df[df['Churn'] == 1]['Monthly_Charges']],
                bins=20, label=['No Churn', 'Churned'],
                color=['green', 'red'], alpha=0.6)
axes[1, 1].set_xlabel('Monthly Charges ($)', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Monthly Charges Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Customer Churn Insights Dashboard', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('churn_viz_5_insights_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: churn_viz_5_insights_dashboard.png")

# ============================================================================
# 10. SAVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("10. GENERATING PROJECT REPORT")
print("=" * 80)

report = f"""
{'=' * 80}
CUSTOMER CHURN PREDICTION - PROJECT REPORT
{'=' * 80}

PROJECT OVERVIEW
{'=' * 80}
Objective: Predict which customers are likely to churn (cancel service)
Dataset: {n_customers} customers with 7 features
Models Trained: 3 (Logistic Regression, Random Forest, Decision Tree)

DATA SUMMARY
{'=' * 80}
Total Customers: {n_customers}
Churned Customers: {churn_counts[1]} ({churn_counts[1] / n_customers * 100:.1f}%)
Active Customers: {churn_counts[0]} ({churn_counts[0] / n_customers * 100:.1f}%)

Training Set: {len(X_train)} samples (70%)
Test Set: {len(X_test)} samples (30%)

FEATURES
{'=' * 80}
1. Age - Customer age (18-70 years)
2. Monthly Charges - Monthly bill amount ($20-$150)
3. Total Charges - Total amount paid over lifetime
4. Tenure Months - How long customer has been with company (1-72 months)
5. Contract Type - Month-to-month (0), One year (1), Two year (2)
6. Support Calls - Number of customer support calls
7. Data Usage GB - Monthly data usage (1-50 GB)

MODEL PERFORMANCE
{'=' * 80}
1. {sorted_results[0][0]:<25} Accuracy: {sorted_results[0][1]['accuracy']:.2%}
2. {sorted_results[1][0]:<25} Accuracy: {sorted_results[1][1]['accuracy']:.2%}
3. {sorted_results[2][0]:<25} Accuracy: {sorted_results[2][1]['accuracy']:.2%}

BEST MODEL: {best_model_name}
{'=' * 80}
Accuracy: {best_model_data['accuracy']:.2%}

Confusion Matrix:
  True Negatives:  {cm[0, 0]} (Correctly predicted no churn)
  False Positives: {cm[0, 1]} (Incorrectly predicted churn)
  False Negatives: {cm[1, 0]} (Missed churn predictions)
  True Positives:  {cm[1, 1]} (Correctly predicted churn)
"""

if best_model_name == 'Random Forest':
    report += f"""
FEATURE IMPORTANCE
{'=' * 80}
Most Important Features (Top 3):
1. {feature_importance_df.iloc[0]['Feature']:<20} {feature_importance_df.iloc[0]['Importance']:.4f}
2. {feature_importance_df.iloc[1]['Feature']:<20} {feature_importance_df.iloc[1]['Importance']:.4f}
3. {feature_importance_df.iloc[2]['Feature']:<20} {feature_importance_df.iloc[2]['Importance']:.4f}
"""

report += f"""
KEY INSIGHTS
{'=' * 80}
• Churn rate is highest for month-to-month contracts
• Customers with short tenure are more likely to churn
• High monthly charges correlate with increased churn
• Frequent support calls indicate potential churn risk

BUSINESS RECOMMENDATIONS
{'=' * 80}
1. Focus retention efforts on month-to-month contract customers
2. Engage with new customers early (first 12 months)
3. Monitor customers with high monthly charges
4. Improve customer support to reduce call frequency
5. Consider loyalty programs for long-tenure customers

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(report)

# Save report
with open('churn_prediction_report.txt', 'w') as f:
    f.write(report)

print("✓ Report saved to: churn_prediction_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT COMPLETE!")
print("=" * 80)

print("\n📊 Models Trained:")
for name in results.keys():
    print(f"  ✓ {name}")

print("\n📁 Files Generated:")
files = [
    "customer_data.csv - Raw customer data (1000 customers)",
    "churn_prediction_report.txt - Comprehensive project report",
    "churn_viz_1_model_comparison.png - Model accuracy comparison",
    "churn_viz_2_confusion_matrix.png - Confusion matrix heatmap",
    "churn_viz_3_feature_importance.png - Feature importance chart",
    "churn_viz_4_distribution.png - Churn distribution analysis",
    "churn_viz_5_insights_dashboard.png - Customer insights dashboard"
]

for file in files:
    print(f"  • {file}")

print(f"\n🏆 Best Model: {best_model_name} with {best_model_data['accuracy']:.2%} accuracy")

if best_model_name == 'Random Forest':
    print(f"\n💡 Key Finding: {feature_importance_df.iloc[0]['Feature']} is the most important predictor of churn")
else:
    print(f"\n💡 Key Finding: Month-to-month contracts have the highest churn rate")

print("\n" + "=" * 80)