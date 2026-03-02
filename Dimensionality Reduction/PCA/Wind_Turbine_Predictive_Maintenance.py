"""
=============================================================
PCA SCENARIO: Wind Turbine Predictive Maintenance
=============================================================

SCENARIO:
---------
A renewable energy company "AeroWind" monitors its wind turbines
using 20 different sensors (vibration, temperature, pressure, etc.).
Processing all 20 sensors in real-time on the edge devices is computationally
expensive, and many of the sensors are highly correlated.

The company wants to build an AI system to:
1. Use PCA to compress the 20 sensor features into a smaller number of
   Principal Components while retaining 95% of the variance.
2. Train a fast machine learning model on these components to predict
   if a turbine is about to fail within the next 24 hours.

FEATURES:
- Sensor_1 to Sensor_20: Continuous readings from various turbine parts.

TARGET:
- Failure: 1 = Imminent Failure, 0 = Healthy
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score, roc_curve)
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Generate Realistic Synthetic Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  AeroWind Predictive Maintenance using PCA")
print("=" * 60)
print("\n[STEP 1] Generating synthetic high-dimensional dataset...\n")

np.random.seed(42)
n_samples = 5000

# To make PCA useful, we simulate 4 "latent" (underlying) factors
# and project them into 20 correlated "observed" sensor features.
latent_factors = np.random.normal(0, 1, (n_samples, 4))

# Create a transformation matrix to map 4 latent factors to 20 sensors
transformation_matrix = np.random.uniform(-1, 1, (4, 20))
X_sensors = np.dot(latent_factors, transformation_matrix)

# Add varying degrees of random noise to each sensor
noise = np.random.normal(0, 0.5, (n_samples, 20))
X_sensors += noise

# Target: Failure is triggered by extreme values in the first two latent factors
failure_prob = (
    0.05
    + 0.3 * (np.abs(latent_factors[:, 0]) > 1.5)
    + 0.4 * (np.abs(latent_factors[:, 1]) > 1.8)
).clip(0, 1)

y_target = (np.random.random(n_samples) < failure_prob).astype(int)

# Create DataFrame
columns = [f'Sensor_{i+1}' for i in range(20)]
df = pd.DataFrame(X_sensors, columns=columns)
df['Failure'] = y_target

print(f"Dataset Shape: {df.shape}")
print(f"Failure Rate:  {df['Failure'].mean() * 100:.1f}%")
print("\nSample Data (First 5 sensors):")
print(df.iloc[:, :5].head())

# ─────────────────────────────────────────────
# STEP 2: Exploratory Data Analysis
# ─────────────────────────────────────────────
print("\n[STEP 2] Exploratory Data Analysis...\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('AeroWind - High Dimensional Sensor EDA', fontsize=16, fontweight='bold')

# Correlation Heatmap - Shows why PCA is needed
corr_matrix = df.drop('Failure', axis=1).corr()
sns.heatmap(corr_matrix, cmap='coolwarm', ax=axes[0], cbar=True,
            xticklabels=False, yticklabels=False)
axes[0].set_title('Sensor Correlation Heatmap\n(Notice the blocks of high correlation)')

# Class Distribution
sns.countplot(x='Failure', data=df, ax=axes[1], palette=['#2ecc71', '#e74c3c'])
axes[1].set_title('Target Distribution (Healthy vs Failure)')
axes[1].set_xticklabels(['Healthy (0)', 'Failure (1)'])

plt.tight_layout()
plt.savefig('pca_eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plot saved.")

# ─────────────────────────────────────────────
# STEP 3: Data Preprocessing (CRUCIAL FOR PCA)
# ─────────────────────────────────────────────
print("\n[STEP 3] Preprocessing Data...\n")

X = df.drop('Failure', axis=1).values
y = df['Failure'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# PCA is heavily sensitive to scale. We MUST standardize features so they have
# a mean of 0 and variance of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled successfully. Ready for Dimensionality Reduction.")

# ─────────────────────────────────────────────
# STEP 4: Apply Principal Component Analysis (PCA)
# ─────────────────────────────────────────────
print("\n[STEP 4] Applying PCA...\n")

# First, let's fit PCA without limiting components to see the explained variance
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Now, let's apply PCA but only keep components that explain 95% of the variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

n_components_kept = pca.n_components_

print(f"Original number of features: 20")
print(f"Reduced number of components (95% variance): {n_components_kept}")
print(f"Data compressed by {(1 - (n_components_kept/20)) * 100:.1f}%")

# ─────────────────────────────────────────────
# STEP 5: Train Classification Model
# ─────────────────────────────────────────────
print("\n[STEP 5] Training Classifier on PCA Data...\n")

# Training a Random Forest on the reduced dimensionality dataset
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_pca, y_train)

# ─────────────────────────────────────────────
# STEP 6: Evaluate Model
# ─────────────────────────────────────────────
print("\n[STEP 6] Evaluating Model Performance...\n")

y_pred = model.predict(X_test_pca)
y_pred_prob = model.predict_proba(X_test_pca)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"ROC-AUC Score : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Failure']))

# ─────────────────────────────────────────────
# STEP 7: Visualize PCA & Results
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('PCA Compression & Predictive Model Results', fontsize=16, fontweight='bold')

# 1. PCA Scree Plot (Explained Variance)
axes[0, 0].plot(range(1, 21), cumulative_variance, marker='o', linestyle='-', color='#3498db')
axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
axes[0, 0].axvline(x=n_components_kept, color='g', linestyle='--', label=f'{n_components_kept} Components')
axes[0, 0].set_title('Cumulative Explained Variance by Components')
axes[0, 0].set_xlabel('Number of Principal Components')
axes[0, 0].set_ylabel('Cumulative Explained Variance')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 2D PCA Projection
# We plot the first two components to see if classes are visually separable
scatter = axes[0, 1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test,
                             cmap='coolwarm', alpha=0.6, edgecolors='w')
axes[0, 1].set_title('2D PCA Projection (PC1 vs PC2)')
axes[0, 1].set_xlabel('Principal Component 1')
axes[0, 1].set_ylabel('Principal Component 2')
legend1 = axes[0, 1].legend(*scatter.legend_elements(), title="Classes")
axes[0, 1].add_artist(legend1)

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Healthy', 'Failure'],
            yticklabels=['Healthy', 'Failure'])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1, 1].plot(fpr, tpr, color='#9b59b6', linewidth=2, label=f'AUC = {auc:.4f}')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPCA visualization and model results plot saved.")

# ─────────────────────────────────────────────
# STEP 8: Predict on a New Sensor Array
# ─────────────────────────────────────────────
print("\n[STEP 8] Real-World Edge Prediction Example...\n")

# Simulate a new incoming array of 20 sensor readings
new_sensor_data = np.random.normal(0, 1.5, (1, 20))

# Preprocessing Pipeline for new data:
# 1. Scale using the fitted StandardScaler
new_scaled = scaler.transform(new_sensor_data)
# 2. Compress using the fitted PCA
new_pca = pca.transform(new_scaled)
# 3. Predict using the RandomForest
failure_probability = model.predict_proba(new_pca)[0][1]

print("New Turbine Sensor Array Received (20 dimensions)")
print(f"Compressed down to {n_components_kept} dimensions for edge processing.")
print(f"\nFailure Probability: {failure_probability * 100:.1f}%")
print(f"Status:              {'⚠️  MAINTENANCE REQUIRED' if failure_probability > 0.5 else '✅  Turbine Healthy'}")

print("\n" + "=" * 60)
print("  All outputs saved to working directory")
print("=" * 60)