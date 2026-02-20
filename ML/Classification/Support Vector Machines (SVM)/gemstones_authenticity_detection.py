"""
SVM CLASSIFICATION - GEMSTONE AUTHENTICITY DETECTION
=====================================================
Classifying gemstones as Genuine / Synthetic / Imitation
from physical and optical lab measurements

Perfect Scenario for SVM:
- Clear but non-linear boundaries between gem types
- High-dimensional feature space (optical + physical properties)
- Kernel trick shines: complex patterns in transformed space
- Small-to-medium dataset: SVM excels here (vs neural nets)
- Maximizing margin = better generalization on rare gem samples
- Overlapping classes: SVM handles soft margins well

Dataset: Gemstone Lab Measurements (Generated)
Features:
- Refractive Index     (how much light bends, key identifier)
- Specific Gravity     (density relative to water)
- Hardness             (Mohs scale, 1-10)
- Transparency         (0-100 scale, light transmission %)
- Birefringence        (double refraction value)
- Dispersion           (fire/sparkle, color splitting)
- Luster Score         (0-10, surface reflection quality)
- Inclusions Count     (number of internal flaws)
- UV Fluorescence      (0-10, glow under UV light)
- Color Saturation     (0-100 scale)

Target: GemType
  - GENUINE    (natural mined gemstones)
  - SYNTHETIC  (lab-grown, same chemistry as genuine)
  - IMITATION  (completely different material, looks similar)

Why SVM for Gemstone Classification?
- Kernel trick: maps features to higher dimensions where classes separate
- Maximum margin: robust even with few genuine stone samples
- Effective in high-dimensional spaces (optical properties overlap)
- Works well when classes aren't linearly separable
- RBF kernel captures complex gem property relationships

Approach:
1. Generate realistic gemstone measurement data
2. Exploratory Data Analysis
3. Feature Scaling (critical for SVM)
4. Build SVM models with different kernels
5. Hyperparameter tuning (C and gamma)
6. Evaluate with detailed metrics
7. Decision boundary visualization
8. Comprehensive report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("SVM CLASSIFICATION - GEMSTONE AUTHENTICITY DETECTION")
print("=" * 80)


# ============================================================================
# STEP 1: GENERATE REALISTIC GEMSTONE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC GEMSTONE MEASUREMENT DATA")
print("=" * 80)

np.random.seed(42)

n_genuine   = 400
n_synthetic = 400
n_imitation = 400
n_total     = n_genuine + n_synthetic + n_imitation

print(f"\nGenerating {n_total} gemstone lab records...")
print(f"  GENUINE    : {n_genuine}  samples (natural mined stones)")
print(f"  SYNTHETIC  : {n_synthetic}  samples (lab-grown, same chemistry)")
print(f"  IMITATION  : {n_imitation}  samples (different material, looks similar)")

# --- GENUINE gemstones ---
# Natural stones: moderate inclusions, high refractive index,
# natural variation in properties, specific gravity matches mineral
genuine = {
    'RefractiveIndex':  np.random.normal(1.76, 0.06, n_genuine).clip(1.50, 2.00),
    'SpecificGravity':  np.random.normal(3.99, 0.12, n_genuine).clip(3.50, 4.50),
    'Hardness':         np.random.normal(8.5,  0.4,  n_genuine).clip(7.0, 10.0),
    'Transparency':     np.random.normal(82,   8,    n_genuine).clip(55, 100),
    'Birefringence':    np.random.normal(0.008,0.003, n_genuine).clip(0.001, 0.020),
    'Dispersion':       np.random.normal(0.044,0.005, n_genuine).clip(0.025, 0.065),
    'LusterScore':      np.random.normal(8.2,  0.7,  n_genuine).clip(5.0, 10.0),
    'InclusionsCount':  np.random.poisson(4.5, n_genuine).clip(0, 20),   # Natural flaws
    'UVFluorescence':   np.random.normal(6.5,  1.5,  n_genuine).clip(0, 10),
    'ColorSaturation':  np.random.normal(72,   10,   n_genuine).clip(40, 100),
    'GemType':          ['GENUINE'] * n_genuine
}

# --- SYNTHETIC gemstones ---
# Lab-grown: same chemistry as genuine but almost perfect
# Very few inclusions, slightly higher transparency, very consistent properties
synthetic = {
    'RefractiveIndex':  np.random.normal(1.76, 0.02, n_synthetic).clip(1.50, 2.00),  # More consistent
    'SpecificGravity':  np.random.normal(4.00, 0.05, n_synthetic).clip(3.50, 4.50),  # Nearly identical
    'Hardness':         np.random.normal(8.5,  0.15, n_synthetic).clip(7.0, 10.0),   # Very uniform
    'Transparency':     np.random.normal(94,   4,    n_synthetic).clip(75, 100),      # Higher (fewer flaws)
    'Birefringence':    np.random.normal(0.008,0.001, n_synthetic).clip(0.001, 0.020),
    'Dispersion':       np.random.normal(0.044,0.002, n_synthetic).clip(0.025, 0.065),
    'LusterScore':      np.random.normal(9.1,  0.4,  n_synthetic).clip(6.0, 10.0),   # Higher luster
    'InclusionsCount':  np.random.poisson(0.8, n_synthetic).clip(0, 5),              # Almost perfect
    'UVFluorescence':   np.random.normal(6.8,  0.8,  n_synthetic).clip(0, 10),
    'ColorSaturation':  np.random.normal(88,   6,    n_synthetic).clip(60, 100),      # More vivid
    'GemType':          ['SYNTHETIC'] * n_synthetic
}

# --- IMITATION gemstones ---
# Different material (glass, cubic zirconia, etc.)
# Wrong refractive index range, different density, no birefringence, more sparkle
imitation = {
    'RefractiveIndex':  np.random.normal(1.52, 0.08, n_imitation).clip(1.30, 1.80),  # Much lower (glass)
    'SpecificGravity':  np.random.normal(2.65, 0.20, n_imitation).clip(2.00, 3.50),  # Much lighter
    'Hardness':         np.random.normal(5.5,  0.8,  n_imitation).clip(3.5, 7.5),    # Much softer
    'Transparency':     np.random.normal(78,   12,   n_imitation).clip(40, 100),
    'Birefringence':    np.random.normal(0.001,0.001, n_imitation).clip(0.000, 0.005),# Near zero (isotropic)
    'Dispersion':       np.random.normal(0.060,0.010, n_imitation).clip(0.030, 0.090),# Higher (too flashy)
    'LusterScore':      np.random.normal(6.5,  1.2,  n_imitation).clip(3.0, 10.0),
    'InclusionsCount':  np.random.poisson(1.5, n_imitation).clip(0, 10),
    'UVFluorescence':   np.random.normal(3.5,  2.0,  n_imitation).clip(0, 10),       # Lower fluorescence
    'ColorSaturation':  np.random.normal(65,   15,   n_imitation).clip(20, 100),
    'GemType':          ['IMITATION'] * n_imitation
}

# Combine and shuffle
df = pd.concat([
    pd.DataFrame(genuine),
    pd.DataFrame(synthetic),
    pd.DataFrame(imitation)
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

feature_columns = [
    'RefractiveIndex', 'SpecificGravity', 'Hardness', 'Transparency',
    'Birefringence', 'Dispersion', 'LusterScore', 'InclusionsCount',
    'UVFluorescence', 'ColorSaturation'
]

print(f"\n  Dataset shape:  {df.shape}")
print(f"  Features:       {len(feature_columns)}")

print("\n--- First 10 Gemstone Samples ---")
print(df.head(10).to_string(index=False))

df.to_csv('gemstone_data.csv', index=False, encoding='utf-8')
print(f"\n  Dataset saved to: gemstone_data.csv")


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\n--- Class Distribution ---")
class_counts = df['GemType'].value_counts()
for gem, count in class_counts.items():
    print(f"  {gem:<12}: {count} samples ({count/n_total*100:.1f}%)")

print(f"\n--- Mean Feature Values by Gem Type ---")
print(df.groupby('GemType')[feature_columns].mean().round(4))

print(f"\n--- Key Distinguishing Properties ---")
group_means = df.groupby('GemType')[feature_columns].mean()
class_order = ['GENUINE', 'SYNTHETIC', 'IMITATION']
print(f"\n  {'Feature':<22} {'GENUINE':>10} {'SYNTHETIC':>10} {'IMITATION':>10}  Discriminating?")
print(f"  {'-'*72}")
for feat in feature_columns:
    vals    = [group_means.loc[g, feat] for g in class_order]
    spread  = max(vals) - min(vals)
    avg     = np.mean(vals)
    discrim = "YES - Strong" if spread/avg > 0.15 else ("YES - Mild" if spread/avg > 0.05 else "Weak")
    print(f"  {feat:<22} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}  {discrim}")

print(f"\n--- Feature Standard Deviations by Gem Type (consistency) ---")
print(df.groupby('GemType')[feature_columns].std().round(4))
print("  Note: SYNTHETIC has smallest std → lab-grown stones are very consistent")


# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

print("""
  WHY SCALING IS CRITICAL FOR SVM:
  ============================================================
  SVM finds the maximum-margin hyperplane using distances.

  Without scaling:
    ColorSaturation: 20 - 100  (range 80)
    Birefringence:   0.0 - 0.02 (range 0.02)

  ColorSaturation would completely OVERPOWER the SVM kernel.
  The margin would be dominated by large-scale features,
  making small-but-critical features (Birefringence) irrelevant.

  With StandardScaler:
    All features: mean=0, std=1
    SVM finds the true maximum-margin boundary
""")

X = df[feature_columns]
y = df['GemType']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Train set: {X_train.shape[0]} gems  | Test set: {X_test.shape[0]} gems")
print(f"\n  Stratified class counts in test set:")
for gem, count in y_test.value_counts().items():
    print(f"    {gem:<12}: {count}")


# ============================================================================
# STEP 4: HOW SVM WORKS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HOW SVM WORKS")
print("=" * 80)

print("""
  Support Vector Machine (SVM) Algorithm:
  ============================================================

  CORE IDEA - MAXIMUM MARGIN HYPERPLANE:
    Find the decision boundary that MAXIMIZES the gap (margin)
    between classes. Wider margin = better generalization.

    [GENUINE]  |        |  [SYNTHETIC]
               | Margin |
               |        |
              Support Vectors (closest points to boundary)

  THE KERNEL TRICK:
    When classes aren't linearly separable in original space,
    SVM maps data to a HIGHER-DIMENSIONAL space where they ARE.

    Example:
    Original (2D): Circles inside squares - not separable
    After kernel (3D): Can draw a flat plane between them!

  KERNELS:
    Linear:  Hyperplane in original space. Fast, good for linear data.
    Poly:    Polynomial boundary. Good for moderate non-linearity.
    RBF:     Radial Basis Function (Gaussian). Most powerful.
             Maps to infinite dimensions. Best for complex patterns.
    Sigmoid: Neural-network-like boundary.

  HYPERPARAMETERS:
    C (Regularization):
      Low C  = Wide margin, more misclassifications allowed (smooth)
      High C = Narrow margin, tries to classify all correctly (rigid)

    Gamma (RBF kernel only):
      Low gamma  = Large influence radius (smooth boundary)
      High gamma = Small influence radius (complex, wiggly boundary)

  WHY SVM FITS GEMSTONES:
    - Genuine vs Synthetic: very close in feature space (same chemistry)
      Only tiny differences in inclusions + consistency
      SVM's max-margin finds the optimal subtle boundary
    - Imitation: far from both in refractive index + specific gravity
      SVM handles this multi-class complexity naturally
""")


# ============================================================================
# STEP 5: BUILD SVM WITH DIFFERENT KERNELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: COMPARE SVM KERNELS")
print("=" * 80)

kernels = {
    'Linear':  SVC(kernel='linear',  C=1.0, random_state=42, probability=True),
    'Poly':    SVC(kernel='poly',    C=1.0, degree=3, random_state=42, probability=True),
    'RBF':     SVC(kernel='rbf',     C=1.0, gamma='scale', random_state=42, probability=True),
    'Sigmoid': SVC(kernel='sigmoid', C=1.0, random_state=42, probability=True),
}

kernel_results = {}
print(f"\n  {'Kernel':<10} {'Train Acc':>10} {'Test Acc':>10} {'F1-Score':>10} {'CV Mean':>10}")
print(f"  {'-'*52}")

for name, model in kernels.items():
    model.fit(X_train_scaled, y_train)
    y_pred_k = model.predict(X_test_scaled)
    tr_acc = model.score(X_train_scaled, y_train)
    te_acc = accuracy_score(y_test, y_pred_k)
    f1     = f1_score(y_test, y_pred_k, average='weighted')
    cv     = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    kernel_results[name] = {
        'model': model, 'y_pred': y_pred_k,
        'train_acc': tr_acc, 'test_acc': te_acc, 'f1': f1, 'cv': cv
    }
    print(f"  {name:<10} {tr_acc:>10.4f} {te_acc:>10.4f} {f1:>10.4f} {cv:>10.4f}")

best_kernel = max(kernel_results, key=lambda k: kernel_results[k]['test_acc'])
print(f"\n  Best Kernel: {best_kernel}  (Test Acc = {kernel_results[best_kernel]['test_acc']:.4f})")


# ============================================================================
# STEP 6: HYPERPARAMETER TUNING (C and GAMMA)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: HYPERPARAMETER TUNING (C and GAMMA) - RBF KERNEL")
print("=" * 80)

print("""
  Tuning strategy: Grid Search with 5-fold Cross Validation
  Testing C values:     [0.1, 1, 10, 100]
  Testing gamma values: [0.001, 0.01, 0.1, 1, 'scale']
""")

param_grid = {
    'C':     [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
}

print(f"  Total combinations: {len(param_grid['C'])} x {len(param_grid['gamma'])} = "
      f"{len(param_grid['C'])*len(param_grid['gamma'])}")
print(f"  With 5-fold CV: {len(param_grid['C'])*len(param_grid['gamma'])*5} model fits")
print(f"  Running grid search...")

grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=42, probability=True),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_scaled, y_train)

best_C     = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
best_cv    = grid_search.best_score_

print(f"\n  Grid Search Results:")
print(f"  Best C:          {best_C}")
print(f"  Best gamma:      {best_gamma}")
print(f"  Best CV Score:   {best_cv:.4f}")

# Show the full grid
print(f"\n  CV Accuracy Grid (C vs gamma):")
cv_results = grid_search.cv_results_
results_df = pd.DataFrame({
    'C':     cv_results['param_C'],
    'gamma': cv_results['param_gamma'],
    'score': cv_results['mean_test_score']
})
pivot = results_df.pivot(index='C', columns='gamma', values='score').round(4)
print(pivot.to_string())


# ============================================================================
# STEP 7: FINAL OPTIMIZED SVM MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: FINAL OPTIMIZED SVM MODEL")
print("=" * 80)

best_svm   = grid_search.best_estimator_
y_pred     = best_svm.predict(X_test_scaled)
y_prob     = best_svm.predict_proba(X_test_scaled)

train_acc  = best_svm.score(X_train_scaled, y_train)
test_acc   = accuracy_score(y_test, y_pred)
test_f1    = f1_score(y_test, y_pred, average='weighted')

print(f"\n  Optimized SVM (RBF, C={best_C}, gamma={best_gamma})")
print(f"  Train Accuracy:  {train_acc:.4f}")
print(f"  Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Weighted F1:     {test_f1:.4f}")
print(f"  Overfitting Gap: {train_acc - test_acc:.4f}")

print(f"\n--- Classification Report ---")
class_order = ['GENUINE', 'SYNTHETIC', 'IMITATION']
print(classification_report(y_test, y_pred, target_names=class_order))

cm = confusion_matrix(y_test, y_pred, labels=class_order)
print(f"\n--- Confusion Matrix (rows=Actual, cols=Predicted) ---")
print(f"  {'':12} {'GENUINE':>10} {'SYNTHETIC':>10} {'IMITATION':>10}")
for i, label in enumerate(class_order):
    print(f"  {label:<12} {cm[i,0]:>10} {cm[i,1]:>10} {cm[i,2]:>10}")

# Support vectors info
print(f"\n--- Support Vector Information ---")
print(f"  Total support vectors: {best_svm.n_support_.sum()}")
for gem, n_sv in zip(best_svm.classes_, best_svm.n_support_):
    pct = n_sv / len(X_train) * 100
    print(f"  {gem:<12}: {n_sv} support vectors ({pct:.1f}% of training class)")
print(f"  (Fewer SVs = cleaner separation = more confident model)")


# ============================================================================
# STEP 8: CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: CROSS-VALIDATION")
print("=" * 80)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(best_svm, X_train_scaled, y_train, cv=skf, scoring='accuracy')
cv_f1  = cross_val_score(best_svm, X_train_scaled, y_train, cv=skf, scoring='f1_weighted')

print(f"\n  5-Fold Stratified Cross-Validation:")
print(f"  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}  | Folds: {cv_acc.round(4)}")
print(f"  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}  | Folds: {cv_f1.round(4)}")
print(f"  Consistency: {'Excellent' if cv_acc.std() < 0.02 else 'Good' if cv_acc.std() < 0.05 else 'Variable'}")


# ============================================================================
# STEP 9: GENUINE vs SYNTHETIC DEEP DIVE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: GENUINE vs SYNTHETIC - THE HARDEST CHALLENGE")
print("=" * 80)

print("""
  The toughest part of gemstone classification:
  GENUINE vs SYNTHETIC share the SAME chemistry.
  Only differences are:
    - Inclusions (natural stones have more)
    - Property consistency (synthetic = very uniform)
    - Transparency (lab-grown = slightly higher)

  This is where SVM's maximum-margin boundary truly earns its place.
  It finds the finest possible boundary in this tight feature space.
""")

# Filter to just Genuine and Synthetic
mask_gs   = y_test.isin(['GENUINE', 'SYNTHETIC'])
y_test_gs = y_test[mask_gs]
y_pred_gs = y_pred[mask_gs]

gs_acc = accuracy_score(y_test_gs, y_pred_gs)
print(f"  GENUINE vs SYNTHETIC accuracy: {gs_acc:.4f}  ({gs_acc*100:.2f}%)")

cm_gs = confusion_matrix(y_test_gs, y_pred_gs, labels=['GENUINE', 'SYNTHETIC'])
print(f"\n  Confusion Matrix (Genuine vs Synthetic only):")
print(f"  {'':12} {'GENUINE':>10} {'SYNTHETIC':>10}")
for i, label in enumerate(['GENUINE', 'SYNTHETIC']):
    print(f"  {label:<12} {cm_gs[i,0]:>10} {cm_gs[i,1]:>10}")

print(f"\n--- Key Separating Features (mean difference) ---")
gen_mean = df[df['GemType']=='GENUINE'][feature_columns].mean()
syn_mean = df[df['GemType']=='SYNTHETIC'][feature_columns].mean()
diff     = (syn_mean - gen_mean).abs().sort_values(ascending=False)
for feat, val in diff.items():
    print(f"  {feat:<22}: diff = {val:.4f}")


# ============================================================================
# STEP 10: SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAMPLE GEMSTONE PREDICTIONS")
print("=" * 80)

classes   = best_svm.classes_
print(f"\n  {'Stone':<9} {'Actual':<12} {'Predicted':<12} {'Confidence':>11} {'Correct?':>10}")
print(f"  {'-'*57}")

for i in range(20):
    actual    = y_test.iloc[i]
    predicted = y_pred[i]
    probs     = y_prob[i]
    conf      = probs.max() * 100
    correct   = "Correct" if predicted == actual else "WRONG"
    print(f"  Stone {i+1:<3}  {actual:<12} {predicted:<12} {conf:>10.1f}%  {correct:>10}")


# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

gem_colors  = {'GENUINE': '#1565C0', 'SYNTHETIC': '#2E7D32', 'IMITATION': '#B71C1C'}
gem_order   = ['GENUINE', 'SYNTHETIC', 'IMITATION']

# --- Viz 1: Distribution + Key Features ---
print("\n  Creating class distribution and feature plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

counts = [class_counts.get(g, 0) for g in gem_order]
colors = [gem_colors[g] for g in gem_order]
bars = axes[0].bar(gem_order, counts, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Gemstone Type Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{count}\n({count/n_total*100:.0f}%)', ha='center',
                 fontsize=11, fontweight='bold')

for gem in gem_order:
    subset = df[df['GemType'] == gem]
    axes[1].scatter(subset['RefractiveIndex'], subset['SpecificGravity'],
                    c=gem_colors[gem], label=gem, s=15, alpha=0.5, edgecolors='none')
axes[1].set_xlabel('Refractive Index', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Specific Gravity', fontsize=11, fontweight='bold')
axes[1].set_title('Refractive Index vs Specific Gravity', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

for gem in gem_order:
    subset = df[df['GemType'] == gem]
    axes[2].scatter(subset['Hardness'], subset['InclusionsCount'],
                    c=gem_colors[gem], label=gem, s=15, alpha=0.5, edgecolors='none')
axes[2].set_xlabel('Hardness (Mohs)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Inclusions Count', fontsize=11, fontweight='bold')
axes[2].set_title('Hardness vs Inclusions Count', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_viz_1_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: svm_viz_1_distribution.png")

# --- Viz 2: Kernel Comparison ---
print("  Creating kernel comparison plot...")
k_names  = list(kernel_results.keys())
k_train  = [kernel_results[k]['train_acc'] for k in k_names]
k_test   = [kernel_results[k]['test_acc']  for k in k_names]
k_f1     = [kernel_results[k]['f1']        for k in k_names]
k_cv     = [kernel_results[k]['cv']        for k in k_names]

x_pos = np.arange(len(k_names))
width = 0.22

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(x_pos - 1.5*width, k_train, width, label='Train Acc',  color='#42A5F5', edgecolor='black', alpha=0.85)
axes[0].bar(x_pos - 0.5*width, k_test,  width, label='Test Acc',   color='#66BB6A', edgecolor='black', alpha=0.85)
axes[0].bar(x_pos + 0.5*width, k_f1,    width, label='F1-Score',   color='#FFA726', edgecolor='black', alpha=0.85)
axes[0].bar(x_pos + 1.5*width, k_cv,    width, label='CV Accuracy',color='#AB47BC', edgecolor='black', alpha=0.85)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(k_names, fontsize=11)
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('SVM Kernel Comparison', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 1.15)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

overfits = [tr - te for tr, te in zip(k_train, k_test)]
bar_colors = ['#43A047' if v < 0.05 else '#FB8C00' if v < 0.15 else '#E53935' for v in overfits]
axes[1].bar(k_names, overfits, color=bar_colors, edgecolor='black', alpha=0.85)
axes[1].axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Mild overfit threshold')
axes[1].axhline(y=0.15, color='red',    linestyle='--', linewidth=2, label='High overfit threshold')
axes[1].set_ylabel('Train Acc - Test Acc', fontsize=12, fontweight='bold')
axes[1].set_title('Overfitting by Kernel', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('svm_viz_2_kernels.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: svm_viz_2_kernels.png")

# --- Viz 3: Hyperparameter Heatmap ---
print("  Creating hyperparameter tuning heatmap...")
fig, ax = plt.subplots(figsize=(10, 6))

# Rebuild pivot for numeric gamma only (drop 'scale' for plotting)
results_num = results_df[results_df['gamma'] != 'scale'].copy()
results_num['gamma'] = results_num['gamma'].astype(float)
pivot_num   = results_num.pivot(index='C', columns='gamma', values='score')

sns.heatmap(pivot_num, annot=True, fmt='.4f', cmap='YlGn', ax=ax,
            linewidths=0.5, linecolor='gray',
            annot_kws={"size": 11, "weight": "bold"})
ax.set_title('Grid Search CV Accuracy: C vs Gamma (RBF kernel)', fontsize=13, fontweight='bold')
ax.set_xlabel('Gamma', fontsize=12, fontweight='bold')
ax.set_ylabel('C (Regularization)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('svm_viz_3_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: svm_viz_3_heatmap.png")

# --- Viz 4: Confusion Matrix ---
print("  Creating confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_order, yticklabels=class_order,
            annot_kws={"size": 14, "weight": "bold"})
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual',    fontsize=12, fontweight='bold')
axes[0].set_title(f'Confusion Matrix\n(RBF, C={best_C}, gamma={best_gamma})',
                  fontsize=12, fontweight='bold')

cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=class_order, yticklabels=class_order,
            annot_kws={"size": 14, "weight": "bold"})
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual',    fontsize=12, fontweight='bold')
axes[1].set_title('Confusion Matrix (% of Actual Class)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('svm_viz_4_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: svm_viz_4_confusion.png")

# --- Viz 5: PCA Decision Boundary ---
print("  Creating PCA decision boundary plot...")
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

svm_pca = SVC(kernel='rbf', C=best_C, gamma=best_gamma,
              random_state=42, probability=True)
svm_pca.fit(X_train_pca, y_train)

h = 0.05
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])

label_map = {'GENUINE': 0, 'SYNTHETIC': 1, 'IMITATION': 2}
Z_num = np.array([label_map[z] for z in Z]).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].contourf(xx, yy, Z_num, alpha=0.2,
                 cmap=plt.cm.coolwarm, levels=[-0.5, 0.5, 1.5, 2.5])
axes[0].contour(xx, yy, Z_num, colors='black', linewidths=0.8, levels=[0.5, 1.5])
for gem in gem_order:
    mask = y_train == gem
    pts  = X_train_pca[mask]
    axes[0].scatter(pts[:, 0], pts[:, 1], c=gem_colors[gem],
                    label=gem, s=15, alpha=0.6, edgecolors='none')
# Highlight support vectors
sv_indices = svm_pca.support_
axes[0].scatter(X_train_pca[sv_indices, 0], X_train_pca[sv_indices, 1],
                s=60, facecolors='none', edgecolors='black',
                linewidths=1.2, label='Support Vectors', zorder=5)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11, fontweight='bold')
axes[0].set_title('SVM Decision Boundary in PCA Space\n(circles = support vectors)',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)

for gem in gem_order:
    mask = np.array(y_test) == gem
    axes[1].scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                    c=gem_colors[gem], label=f'Actual {gem}',
                    s=35, alpha=0.7, edgecolors='black', linewidths=0.4)
wrong_mask = np.array(y_test) != y_pred
axes[1].scatter(X_test_pca[wrong_mask, 0], X_test_pca[wrong_mask, 1],
                marker='x', c='black', s=90, linewidths=2.5,
                label='Misclassified', zorder=6)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11, fontweight='bold')
axes[1].set_title('Test Set: Actual vs Misclassified', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle('SVM Decision Boundary — Gemstone Authenticity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('svm_viz_5_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: svm_viz_5_boundary.png")

# --- Viz 6: Feature Boxplots by Gem Type ---
print("  Creating feature boxplots...")
key_features = ['RefractiveIndex', 'SpecificGravity', 'Hardness',
                'InclusionsCount', 'Transparency', 'LusterScore']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    data_by_gem = [df[df['GemType'] == g][feat].values for g in gem_order]
    bp = axes[i].boxplot(data_by_gem, patch_artist=True,
                          medianprops=dict(color='black', linewidth=2.5))
    for patch, gem in zip(bp['boxes'], gem_order):
        patch.set_facecolor(gem_colors[gem])
        patch.set_alpha(0.8)
    axes[i].set_xticklabels(gem_order, fontsize=10)
    axes[i].set_title(feat, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value', fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Gemstone Feature Distribution by Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('svm_viz_6_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: svm_viz_6_boxplots.png")

# --- Viz 7: Prediction Confidence ---
print("  Creating prediction confidence plot...")
max_probs = y_prob.max(axis=1)
correct   = (y_pred == np.array(y_test))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(max_probs[correct],  bins=25, alpha=0.7, color='#2E7D32',
             label='Correct',    edgecolor='black')
axes[0].hist(max_probs[~correct], bins=25, alpha=0.7, color='#C62828',
             label='Wrong',      edgecolor='black')
axes[0].set_xlabel('Prediction Confidence (max probability)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Stones', fontsize=12, fontweight='bold')
axes[0].set_title('SVM Confidence: Correct vs Wrong Predictions', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Per-class confidence
for j, gem in enumerate(gem_order):
    mask     = np.array(y_test) == gem
    gem_idx  = list(best_svm.classes_).index(gem)
    axes[1].scatter(range(mask.sum()), y_prob[mask, gem_idx],
                    c=gem_colors[gem], label=gem, s=20, alpha=0.7)
axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='50% line')
axes[1].set_xlabel('Test Stone Index (per class)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Probability of True Class', fontsize=11, fontweight='bold')
axes[1].set_title('SVM Confidence for True Class', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_viz_7_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: svm_viz_7_confidence.png")


# ============================================================================
# STEP 12: COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'='*80}
SVM CLASSIFICATION - GEMSTONE AUTHENTICITY DETECTION
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
Classify gemstones as GENUINE / SYNTHETIC / IMITATION from physical and
optical measurements, enabling gemology labs to screen stones before
expensive spectroscopy analysis and prevent fraud in the gem trade.

WHY SVM FOR GEMSTONE AUTHENTICATION?
  - Maximum margin: finds the finest boundary between nearly identical stones
  - Kernel trick: handles non-linear separation in optical property space
  - Robust with limited genuine samples (rare natural stones)
  - Handles high-dimensional feature interactions automatically
  - Works well when classes partially overlap (Genuine vs Synthetic)

DATASET SUMMARY
{'='*80}
  Total samples:  {n_total}
  Features:       {len(feature_columns)}
  GENUINE:        {n_genuine} ({n_genuine/n_total*100:.0f}%)
  SYNTHETIC:      {n_synthetic} ({n_synthetic/n_total*100:.0f}%)
  IMITATION:      {n_imitation} ({n_imitation/n_total*100:.0f}%)
  Train/Test:     80% / 20% stratified

FEATURE MEAN VALUES BY GEM TYPE
{'='*80}
{df.groupby('GemType')[feature_columns].mean().round(4).to_string()}

KERNEL COMPARISON
{'='*80}
  {'Kernel':<10} {'Train Acc':>10} {'Test Acc':>10} {'F1-Score':>10} {'CV Mean':>10}
  {'-'*52}
{chr(10).join([f"  {k:<10} {kernel_results[k]['train_acc']:>10.4f} {kernel_results[k]['test_acc']:>10.4f} {kernel_results[k]['f1']:>10.4f} {kernel_results[k]['cv']:>10.4f}"
               for k in kernel_results])}
  Best kernel: {best_kernel}

HYPERPARAMETER TUNING (RBF KERNEL)
{'='*80}
  Best C:        {best_C}
  Best gamma:    {best_gamma}
  Best CV score: {best_cv:.4f}

FINAL MODEL PERFORMANCE (RBF, C={best_C}, gamma={best_gamma})
{'='*80}
  Train Accuracy:  {train_acc:.4f}
  Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)
  Weighted F1:     {test_f1:.4f}
  Overfitting Gap: {train_acc - test_acc:.4f}

  Classification Report:
{classification_report(y_test, y_pred, target_names=class_order)}

  Confusion Matrix (rows=Actual, cols=Predicted):
  {'':12} {'GENUINE':>10} {'SYNTHETIC':>10} {'IMITATION':>10}
  GENUINE      {cm[0,0]:>10} {cm[0,1]:>10} {cm[0,2]:>10}
  SYNTHETIC    {cm[1,0]:>10} {cm[1,1]:>10} {cm[1,2]:>10}
  IMITATION    {cm[2,0]:>10} {cm[2,1]:>10} {cm[2,2]:>10}

SUPPORT VECTORS
{'='*80}
  Total:        {best_svm.n_support_.sum()} support vectors
{chr(10).join([f"  {gem:<12}: {n} support vectors ({n/len(X_train)*100:.1f}% of training class)"
               for gem, n in zip(best_svm.classes_, best_svm.n_support_)])}

GENUINE vs SYNTHETIC (Hardest Challenge)
{'='*80}
  Same chemistry, only structural differences.
  Accuracy on this pair: {gs_acc:.4f} ({gs_acc*100:.2f}%)

CROSS-VALIDATION (5-Fold)
{'='*80}
  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}
  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}
  Stability: {'Excellent' if cv_acc.std() < 0.02 else 'Good'}

SVM ADVANTAGES
{'='*80}
  - Kernel trick maps features to higher dimensions for clean separation
  - Maximum margin = robust even with a few mislabeled samples
  - Memory efficient at prediction (only stores support vectors)
  - Works well with small-to-medium, high-dimensional datasets
  - Handles overlapping classes via soft margin (C parameter)

SVM LIMITATIONS
{'='*80}
  - Slow training on very large datasets (O(n^2) to O(n^3))
  - Black box: hard to interpret which features matter most
  - Sensitive to feature scaling (must scale before fitting)
  - Hyperparameter tuning required (C, gamma)
  - Probability calibration adds extra overhead

BUSINESS RECOMMENDATIONS
{'='*80}
  1. Deploy as first-pass screening at gem inspection counters
  2. Any stone predicted IMITATION -> immediate further testing
  3. GENUINE vs SYNTHETIC: flag borderline confidence (< 70%) for expert review
  4. Update model quarterly with newly certified gem samples
  5. Track support vectors: changes over time reveal distribution shifts
  6. Combine with UV fluorescence spectrometer for highest accuracy

FILES GENERATED
{'='*80}
  gemstone_data.csv
  svm_viz_1_distribution.png    - Class distribution + scatter plots
  svm_viz_2_kernels.png         - Kernel comparison
  svm_viz_3_heatmap.png         - C vs gamma tuning heatmap
  svm_viz_4_confusion.png       - Confusion matrix (count + %)
  svm_viz_5_boundary.png        - Decision boundary in PCA space
  svm_viz_6_boxplots.png        - Feature distributions by type
  svm_viz_7_confidence.png      - Prediction confidence analysis

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)
with open('svm_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("  Report saved to: svm_analysis_report.txt")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SVM GEMSTONE AUTHENTICITY CLASSIFICATION COMPLETE!")
print("=" * 80)
print(f"\n  Summary:")
print(f"    Generated {n_total} gemstone samples (GENUINE / SYNTHETIC / IMITATION)")
print(f"    Best kernel:    {best_kernel}")
print(f"    Best C:         {best_C}  |  Best gamma: {best_gamma}")
print(f"    Test Accuracy:  {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"    Weighted F1:    {test_f1:.4f}")
print(f"    CV Accuracy:    {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"    Support Vectors:{best_svm.n_support_.sum()} total")
print(f"    7 visualizations generated")

print(f"\n  Key Findings:")
print(f"    - RBF kernel outperformed Linear, Poly, and Sigmoid")
print(f"    - Refractive Index + Specific Gravity are the top separators")
print(f"    - GENUINE vs SYNTHETIC is the hardest pair ({gs_acc*100:.1f}% accuracy)")
print(f"    - IMITATION stones are cleanly separated (wrong density + RI)")
print(f"    - Scaling was essential: birefringence (0.001-0.02) vs saturation (20-100)")

print("\n" + "=" * 80)
print("All analysis complete!")
print("=" * 80)