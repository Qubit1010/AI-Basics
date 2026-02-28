"""
=============================================================
ANN SCENARIO: Bank Customer Churn Prediction
=============================================================

SCENARIO:
---------
A retail bank "NeoBank" is losing customers (churning) at an alarming rate.
The bank has historical data on 10,000 customers including demographics,
account details, and whether they eventually left the bank.

The bank wants to build an AI system to:
1. Predict which customers are likely to churn
2. Take proactive measures (discounts, personal calls, offers) to retain them

FEATURES:
- CreditScore       : Customer credit score
- Geography         : Country (France, Germany, Spain)
- Gender            : Male/Female
- Age               : Customer age
- Tenure            : Years with bank
- Balance           : Account balance
- NumOfProducts     : Number of bank products used
- HasCrCard         : Has credit card? (0/1)
- IsActiveMember    : Is active member? (0/1)
- EstimatedSalary   : Estimated annual salary

TARGET:
- Exited: 1 = churned, 0 = stayed
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Generate Realistic Synthetic Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  NeoBank Customer Churn Prediction using ANN")
print("=" * 60)
print("\n[STEP 1] Generating synthetic banking dataset...\n")

np.random.seed(42)
n_samples = 10000

# Simulate realistic banking data
geography = np.random.choice(['France', 'Germany', 'Spain'],
                             n_samples, p=[0.5, 0.25, 0.25])
gender = np.random.choice(['Male', 'Female'], n_samples)
age = np.random.normal(38, 10, n_samples).clip(18, 80).astype(int)
tenure = np.random.randint(0, 11, n_samples)
credit_score = np.random.normal(650, 80, n_samples).clip(350, 850).astype(int)
balance = np.where(np.random.random(n_samples) < 0.3, 0,
                   np.random.normal(76000, 62000, n_samples).clip(0))
num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.45, 0.03, 0.02])
has_cr_card = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
is_active = np.random.choice([0, 1], n_samples, p=[0.48, 0.52])
salary = np.random.normal(100000, 50000, n_samples).clip(10000, 300000)

# Churn probability based on features (realistic rules)
churn_prob = (
        0.05
        + 0.15 * (age > 50)
        + 0.10 * (geography == 'Germany')
        - 0.08 * is_active
        + 0.12 * (num_products >= 3)
        - 0.05 * (balance > 0)
        + 0.05 * (credit_score < 550)
        - 0.03 * has_cr_card
        + np.random.normal(0, 0.05, n_samples)
).clip(0, 1)

exited = (np.random.random(n_samples) < churn_prob).astype(int)

df = pd.DataFrame({
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': salary,
    'Exited': exited
})

print(f"Dataset Shape: {df.shape}")
print(f"Churn Rate: {exited.mean() * 100:.1f}%")
print("\nSample Data:")
print(df.head())

# ─────────────────────────────────────────────
# STEP 2: Exploratory Data Analysis
# ─────────────────────────────────────────────
print("\n[STEP 2] Exploratory Data Analysis...\n")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('NeoBank - Customer Churn EDA', fontsize=16, fontweight='bold')

# Churn distribution
churn_counts = df['Exited'].value_counts()
axes[0, 0].pie(churn_counts, labels=['Retained', 'Churned'],
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
               startangle=90, explode=(0, 0.08))
axes[0, 0].set_title('Churn Distribution')

# Age vs Churn
df.boxplot(column='Age', by='Exited', ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Churn')
axes[0, 1].set_xlabel('Churned (1) / Retained (0)')

# Geography vs Churn
geo_churn = df.groupby('Geography')['Exited'].mean() * 100
geo_churn.plot(kind='bar', ax=axes[0, 2], color=['#3498db', '#e74c3c', '#f39c12'])
axes[0, 2].set_title('Churn Rate by Geography (%)')
axes[0, 2].set_xlabel('')
axes[0, 2].tick_params(axis='x', rotation=0)

# Active vs Churn
active_churn = df.groupby('IsActiveMember')['Exited'].mean() * 100
active_churn.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#2ecc71'])
axes[1, 0].set_title('Churn Rate: Active vs Inactive Members')
axes[1, 0].set_xticklabels(['Inactive', 'Active'], rotation=0)

# NumOfProducts vs Churn
prod_churn = df.groupby('NumOfProducts')['Exited'].mean() * 100
prod_churn.plot(kind='bar', ax=axes[1, 1], color='#9b59b6')
axes[1, 1].set_title('Churn Rate by Number of Products')
axes[1, 1].set_xlabel('Number of Products')

# Credit Score distribution
df.groupby('Exited')['CreditScore'].plot(kind='hist', ax=axes[1, 2],
                                         alpha=0.6, bins=30)
axes[1, 2].set_title('Credit Score Distribution')
axes[1, 2].legend(['Retained', 'Churned'])

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plot saved.")

# ─────────────────────────────────────────────
# STEP 3: Data Preprocessing
# ─────────────────────────────────────────────
print("\n[STEP 3] Preprocessing Data...\n")

df_processed = df.copy()

# Encode categorical variables
le_geo = LabelEncoder()
le_gen = LabelEncoder()
df_processed['Geography'] = le_geo.fit_transform(df_processed['Geography'])
df_processed['Gender'] = le_gen.fit_transform(df_processed['Gender'])

X = df_processed.drop('Exited', axis=1).values
y = df_processed['Exited'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")
print(f"Features:     {df.drop('Exited', axis=1).columns.tolist()}")

# ─────────────────────────────────────────────
# STEP 4: Build ANN Model
# ─────────────────────────────────────────────
print("\n[STEP 4] Building ANN Architecture...\n")

model = Sequential([
    # Input Layer
    Dense(64, activation='relu', input_shape=(X_train.shape[1],),
          kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden Layer 1
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.4),

    # Hidden Layer 2
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden Layer 3
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),

    # Output Layer
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ─────────────────────────────────────────────
# STEP 5: Train Model
# ─────────────────────────────────────────────
print("\n[STEP 5] Training the ANN...\n")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# STEP 6: Evaluate Model
# ─────────────────────────────────────────────
print("\n[STEP 6] Evaluating Model Performance...\n")

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"ROC-AUC Score : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# ─────────────────────────────────────────────
# STEP 7: Visualize Results
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ANN Model Results - NeoBank Churn Prediction', fontsize=16, fontweight='bold')

# Training History - Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', color='#3498db', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='#e74c3c',
                linewidth=2, linestyle='--')
axes[0, 0].set_title('Model Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training History - Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train Acc', color='#2ecc71', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc', color='#f39c12',
                linewidth=2, linestyle='--')
axes[0, 1].set_title('Model Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1, 1].plot(fpr, tpr, color='#9b59b6', linewidth=2,
                label=f'AUC = {auc:.4f}')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nModel results plot saved.")

# ─────────────────────────────────────────────
# STEP 8: Predict on a New Customer
# ─────────────────────────────────────────────
print("\n[STEP 8] Real-World Prediction Example...\n")

# New customer data
new_customer = pd.DataFrame({
    'CreditScore': [600],
    'Geography': ['Germany'],  # Germany has higher churn
    'Gender': ['Female'],
    'Age': [55],  # Older = higher churn risk
    'Tenure': [2],
    'Balance': [80000],
    'NumOfProducts': [4],  # Many products = high churn
    'HasCrCard': [1],
    'IsActiveMember': [0],  # Inactive = high risk
    'EstimatedSalary': [90000]
})

new_customer['Geography'] = le_geo.transform(new_customer['Geography'])
new_customer['Gender'] = le_gen.transform(new_customer['Gender'])
new_scaled = scaler.transform(new_customer.values)
churn_probability = model.predict(new_scaled)[0][0]

print("New Customer Profile:")
print("  Geography:      Germany")
print("  Age:            55")
print("  IsActiveMember: No")
print("  NumOfProducts:  4")
print("  CreditScore:    600")
print(f"\nChurn Probability: {churn_probability * 100:.1f}%")
print(
    f"Prediction:        {'⚠️  LIKELY TO CHURN - Intervene NOW!' if churn_probability > 0.5 else '✅  Likely to Stay'}")

print("\n" + "=" * 60)
print("  All outputs saved to ")
print("=" * 60)