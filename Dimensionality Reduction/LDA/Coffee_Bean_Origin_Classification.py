"""
=============================================================
LDA SCENARIO: Coffee Bean Origin Classification
=============================================================

SCENARIO:
---------
A global coffee roaster "GlobalBeans" imports Arabica coffee from
three distinct regions: Brazil, Colombia, and Ethiopia. Counterfeit
beans are a problem in the supply chain. The company uses mass spectrometry
to measure 12 chemical compounds in unlabelled bean batches.

The company wants to build an AI system to:
1. Use LDA to reduce the 12 chemical features down to exactly 2 dimensions
   (since we have 3 classes: 3 - 1 = 2) to maximize class separation.
2. Classify the geographic origin of the beans based on this projection.

FEATURES:
- Chem_1 to Chem_12: Continuous measurements of compounds like
  chlorogenic acid, caffeine, trigonelline, lipids, and volatile aromatics.

TARGET:
- Origin: 0 = Brazil, 1 = Colombia, 2 = Ethiopia
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score)
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Generate Realistic Synthetic Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  GlobalBeans Origin Classification using LDA")
print("=" * 60)
print("\n[STEP 1] Generating synthetic chemical dataset...\n")

np.random.seed(42)
n_samples = 3000
n_features = 12

# Simulate 3 classes (Regions) with slightly different chemical profiles
# We will create overlapping distributions to make the LDA's job realistic
origins = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])

# Base chemical profile for all coffee
X_chem = np.random.normal(10, 3, (n_samples, n_features))

# Add region-specific chemical markers (simulating terroir differences)
for i in range(n_samples):
    if origins[i] == 0:   # Brazil (e.g., lower acidity, higher body)
        X_chem[i, 0:4] += np.random.normal(2, 1, 4)
        X_chem[i, 8:12] -= np.random.normal(1, 0.5, 4)
    elif origins[i] == 1: # Colombia (e.g., balanced, bright)
        X_chem[i, 4:8] += np.random.normal(3, 1, 4)
    else:                 # Ethiopia (e.g., high floral aromatics)
        X_chem[i, 8:12] += np.random.normal(4, 1.5, 4)
        X_chem[i, 0:4] -= np.random.normal(1.5, 0.5, 4)

# Create DataFrame
columns = [f'Chem_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X_chem, columns=columns)
df['Origin'] = origins

origin_names = {0: 'Brazil', 1: 'Colombia', 2: 'Ethiopia'}
df['Origin_Label'] = df['Origin'].map(origin_names)

print(f"Dataset Shape: {df.shape}")
print(f"Class Counts:\n{df['Origin_Label'].value_counts()}")
print("\nSample Data (First 5 chemicals):")
print(df.iloc[:, :5].head())

# ─────────────────────────────────────────────
# STEP 2: Exploratory Data Analysis
# ─────────────────────────────────────────────
print("\n[STEP 2] Exploratory Data Analysis...\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('GlobalBeans - Chemical Profile EDA', fontsize=16, fontweight='bold')

# Feature Distribution (Chem_1) by Origin
sns.kdeplot(data=df, x='Chem_1', hue='Origin_Label', fill=True, ax=axes[0],
            palette=['#27ae60', '#f39c12', '#8e44ad'])
axes[0].set_title('Distribution of Chem_1 by Origin\n(Notice the overlap between classes)')

# Feature Distribution (Chem_9) by Origin
sns.kdeplot(data=df, x='Chem_9', hue='Origin_Label', fill=True, ax=axes[1],
            palette=['#27ae60', '#f39c12', '#8e44ad'])
axes[1].set_title('Distribution of Chem_9 by Origin')

plt.tight_layout()
plt.savefig('lda_eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plot saved.")

# ─────────────────────────────────────────────
# STEP 3: Data Preprocessing
# ─────────────────────────────────────────────
print("\n[STEP 3] Preprocessing Data...\n")

X = df.drop(['Origin', 'Origin_Label'], axis=1).values
y = df['Origin'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standard scaling is highly recommended for LDA so features with
# larger scales don't dominate the distance calculations.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print("Data scaled successfully.")

# ─────────────────────────────────────────────
# STEP 4 & 5: Apply LDA & Train Model
# ─────────────────────────────────────────────
print("\n[STEP 4 & 5] Applying LDA for Dimensionality Reduction and Classification...\n")

# Unlike PCA, LDA takes both X (features) AND y (labels) during the fit process
# It will automatically reduce to min(n_classes - 1, n_features) = 2 components
lda = LinearDiscriminantAnalysis(n_components=2)

# Fit the model and transform the data simultaneously for visualization later
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

print("LDA fitted successfully.")
print(f"Original feature space: {n_features} dimensions")
print(f"LDA projected space:    {X_train_lda.shape[1]} dimensions")
print(f"Explained Variance Ratio by the 2 linear discriminants: {lda.explained_variance_ratio_}")

# ─────────────────────────────────────────────
# STEP 6: Evaluate Model
# ─────────────────────────────────────────────
print("\n[STEP 6] Evaluating LDA Classifier Performance...\n")

# Because sklearn's LDA is also a classifier, we can predict directly from it!
y_pred = lda.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy : {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Brazil', 'Colombia', 'Ethiopia']))

# ─────────────────────────────────────────────
# STEP 7: Visualize LDA Results
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('LDA Projection & Classification Results', fontsize=16, fontweight='bold')

# 1. 2D LDA Scatter Plot
# This perfectly shows how LDA maximizes the distance between the 3 classes
scatter = axes[0].scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test,
                          cmap='viridis', alpha=0.7, edgecolors='w', s=60)
axes[0].set_title('2D LDA Projection (LD1 vs LD2)')
axes[0].set_xlabel(f'Linear Discriminant 1 ({lda.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'Linear Discriminant 2 ({lda.explained_variance_ratio_[1]*100:.1f}%)')

# Custom legend for the scatter plot
handles, _ = scatter.legend_elements()
axes[0].legend(handles, ['Brazil', 'Colombia', 'Ethiopia'], title="Origin")
axes[0].grid(True, alpha=0.3)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['Brazil', 'Colombia', 'Ethiopia'],
            yticklabels=['Brazil', 'Colombia', 'Ethiopia'])
axes[1].set_title('Confusion Matrix')
axes[1].set_ylabel('Actual Origin')
axes[1].set_xlabel('Predicted Origin')

plt.tight_layout()
plt.savefig('lda_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nLDA visualization and model results plot saved.")

# ─────────────────────────────────────────────
# STEP 8: Predict on a New Batch of Beans
# ─────────────────────────────────────────────
print("\n[STEP 8] Real-World Prediction Example...\n")

# Simulate a new, unlabelled batch of coffee beans (12 chemical features)
new_bean_batch = np.random.normal(12, 2.5, (1, 12))

# Preprocessing: Scale the new data
new_scaled = scaler.transform(new_bean_batch)

# Predict using the fitted LDA model
predicted_class = lda.predict(new_scaled)[0]
prediction_probabilities = lda.predict_proba(new_scaled)[0]

print("New Coffee Batch Chemical Profile Received (12 markers)")
print(f"Confidence Scores:")
print(f"  Brazil:   {prediction_probabilities[0] * 100:.1f}%")
print(f"  Colombia: {prediction_probabilities[1] * 100:.1f}%")
print(f"  Ethiopia: {prediction_probabilities[2] * 100:.1f}%")

print(f"\nFinal Classification: ✅ {origin_names[predicted_class]}")

print("\n" + "=" * 60)
print("  All outputs saved to working directory")
print("=" * 60)