"""
KNN CLASSIFICATION - WINE QUALITY PREDICTION
=============================================
Classifying wine quality (Low / Medium / High) from chemical lab measurements

Perfect Scenario for KNN:
- Wine samples with SIMILAR chemical profiles taste similar (locality matters)
- No need to assume linear boundaries (wine quality is complex)
- Multi-class problem (3 quality tiers)
- Feature similarity IS the prediction logic (wines close in chemical space = same quality)
- Moderate dataset size - KNN is efficient here
- Interpretable: "This wine is similar to these K known wines"

Dataset: Red Wine Chemical Properties (Generated)
Features:
- Fixed Acidity       (tartaric acid, g/L)
- Volatile Acidity    (acetic acid, g/L) — too high = vinegar taste
- Citric Acid         (g/L) — adds freshness
- Residual Sugar      (g/L) — sweetness
- Chlorides           (salt content, g/L)
- Free Sulfur Dioxide (mg/L) — preservative
- Total Sulfur Dioxide(mg/L)
- Density             (g/cm³)
- pH                  (acidity level)
- Sulphates           (g/L) — antimicrobial
- Alcohol             (% volume)

Target: Quality Tier
  - LOW    (score 3-4)
  - MEDIUM (score 5-6)
  - HIGH   (score 7-8)

Why KNN for Wine Quality?
- Similarity-based: wines with similar chemistry taste similar
- No training phase needed - stores all data
- Natural for multi-class problems
- Works great when decision boundary is irregular
- K controls smoothness: small K = complex, large K = smoother

Approach:
1. Generate realistic wine chemical data
2. Exploratory Data Analysis
3. Feature Scaling (CRITICAL for KNN)
4. Find optimal K (Elbow method)
5. Build and evaluate KNN classifier
6. Compare different distance metrics
7. Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
print("KNN CLASSIFICATION - WINE QUALITY PREDICTION")
print("=" * 80)


# ============================================================================
# STEP 1: GENERATE REALISTIC WINE CHEMICAL DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC WINE CHEMICAL DATA")
print("=" * 80)

np.random.seed(42)
n_samples = 1500

print(f"\nGenerating {n_samples} wine samples with realistic chemistry...")

# --- LOW QUALITY wines (score 3-4) ---
# High volatile acidity, low alcohol, high chlorides
n_low = 300
low = {
    'FixedAcidity':       np.random.normal(7.5, 1.2, n_low).clip(4, 12),
    'VolatileAcidity':    np.random.normal(0.75, 0.15, n_low).clip(0.3, 1.2),  # Too high = vinegar
    'CitricAcid':         np.random.normal(0.18, 0.08, n_low).clip(0, 0.5),
    'ResidualSugar':      np.random.normal(2.5, 1.0, n_low).clip(1, 8),
    'Chlorides':          np.random.normal(0.11, 0.03, n_low).clip(0.05, 0.2),  # Salty
    'FreeSulfurDioxide':  np.random.normal(11, 5, n_low).clip(1, 30),
    'TotalSulfurDioxide': np.random.normal(35, 12, n_low).clip(6, 80),
    'Density':            np.random.normal(0.9975, 0.003, n_low).clip(0.990, 1.005),
    'pH':                 np.random.normal(3.45, 0.18, n_low).clip(2.9, 4.0),
    'Sulphates':          np.random.normal(0.52, 0.10, n_low).clip(0.3, 0.9),
    'Alcohol':            np.random.normal(9.8, 0.7, n_low).clip(8, 11.5),  # Low alcohol
    'QualityTier':        ['LOW'] * n_low
}

# --- MEDIUM QUALITY wines (score 5-6) ---
# Balanced chemistry
n_med = 900
med = {
    'FixedAcidity':       np.random.normal(8.2, 1.0, n_med).clip(5, 13),
    'VolatileAcidity':    np.random.normal(0.50, 0.12, n_med).clip(0.2, 0.85),
    'CitricAcid':         np.random.normal(0.30, 0.09, n_med).clip(0, 0.6),
    'ResidualSugar':      np.random.normal(2.6, 0.9, n_med).clip(1, 9),
    'Chlorides':          np.random.normal(0.085, 0.022, n_med).clip(0.04, 0.16),
    'FreeSulfurDioxide':  np.random.normal(16, 7, n_med).clip(1, 40),
    'TotalSulfurDioxide': np.random.normal(46, 15, n_med).clip(10, 100),
    'Density':            np.random.normal(0.9965, 0.002, n_med).clip(0.990, 1.003),
    'pH':                 np.random.normal(3.32, 0.15, n_med).clip(2.9, 3.9),
    'Sulphates':          np.random.normal(0.65, 0.12, n_med).clip(0.3, 1.1),
    'Alcohol':            np.random.normal(10.6, 0.9, n_med).clip(8.5, 13),
    'QualityTier':        ['MEDIUM'] * n_med
}

# --- HIGH QUALITY wines (score 7-8) ---
# Low volatile acidity, high alcohol, high sulphates, good citric acid
n_high = 300
high = {
    'FixedAcidity':       np.random.normal(8.8, 0.9, n_high).clip(6, 14),
    'VolatileAcidity':    np.random.normal(0.33, 0.10, n_high).clip(0.12, 0.65),  # Low = good
    'CitricAcid':         np.random.normal(0.42, 0.09, n_high).clip(0.1, 0.75),   # High = fresh
    'ResidualSugar':      np.random.normal(2.7, 0.8, n_high).clip(1, 7),
    'Chlorides':          np.random.normal(0.072, 0.018, n_high).clip(0.03, 0.13),
    'FreeSulfurDioxide':  np.random.normal(13, 6, n_high).clip(1, 35),
    'TotalSulfurDioxide': np.random.normal(34, 12, n_high).clip(6, 75),
    'Density':            np.random.normal(0.9950, 0.002, n_high).clip(0.988, 1.000),
    'pH':                 np.random.normal(3.25, 0.14, n_high).clip(2.9, 3.7),
    'Sulphates':          np.random.normal(0.78, 0.12, n_high).clip(0.4, 1.3),    # High = good
    'Alcohol':            np.random.normal(11.8, 0.8, n_high).clip(9.5, 14),      # High = good
    'QualityTier':        ['HIGH'] * n_high
}

# Combine all
df = pd.concat([
    pd.DataFrame(low),
    pd.DataFrame(med),
    pd.DataFrame(high)
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

feature_columns = ['FixedAcidity', 'VolatileAcidity', 'CitricAcid', 'ResidualSugar',
                   'Chlorides', 'FreeSulfurDioxide', 'TotalSulfurDioxide',
                   'Density', 'pH', 'Sulphates', 'Alcohol']

print(f"\n  Generation logic:")
print(f"  LOW    wines: High volatile acidity, low alcohol, high chlorides")
print(f"  MEDIUM wines: Balanced chemistry across all features")
print(f"  HIGH   wines: Low volatile acidity, high alcohol, high sulphates")

print(f"\n  Dataset shape:    {df.shape}")
print(f"  LOW quality:      {n_low}  ({n_low/n_samples*100:.1f}%)")
print(f"  MEDIUM quality:   {n_med}  ({n_med/n_samples*100:.1f}%)")
print(f"  HIGH quality:     {n_high}  ({n_high/n_samples*100:.1f}%)")

print("\n--- First 10 Wine Samples ---")
print(df.head(10).to_string(index=False))

df.to_csv('wine_quality_data.csv', index=False, encoding='utf-8')
print(f"\n  Dataset saved to: wine_quality_data.csv")


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\n--- Class Distribution ---")
class_counts = df['QualityTier'].value_counts()
for tier, count in class_counts.items():
    print(f"  {tier:<8}: {count} wines ({count/n_samples*100:.1f}%)")

print(f"\n--- Feature Statistics ---")
print(df[feature_columns].describe().round(3))

print(f"\n--- Mean Feature Values by Quality Tier ---")
print(df.groupby('QualityTier')[feature_columns].mean().round(3))

print(f"\n--- Key Chemical Differences (tells KNN what to look for) ---")
group_means = df.groupby('QualityTier')[feature_columns].mean()
for feat in feature_columns:
    low_m   = group_means.loc['LOW', feat]
    med_m   = group_means.loc['MEDIUM', feat]
    high_m  = group_means.loc['HIGH', feat]
    trend   = "LOW->HIGH" if high_m > low_m else "HIGH->LOW"
    print(f"  {feat:<22}: LOW={low_m:.3f}  MED={med_m:.3f}  HIGH={high_m:.3f}  Trend: {trend}")


# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

print("""
  WHY SCALING IS ABSOLUTELY CRITICAL FOR KNN:
  ============================================================
  KNN calculates DISTANCE between data points.

  Without scaling:
    Alcohol range: 8 - 14    (range ~6)
    FreeSO2 range: 1 - 40    (range ~39)

  FreeSO2 would DOMINATE the distance calculation!
  A wine with very different SO2 looks "far away" even if
  all other features are nearly identical.

  With StandardScaler:
    All features: mean=0, std=1
    Every feature contributes EQUALLY to distance
""")

X = df[feature_columns]
y = df['QualityTier']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Train set: {X_train.shape[0]} wines  | Test set: {X_test.shape[0]} wines")
print(f"\n  Before scaling (first wine, first 4 features):")
print(f"  {X_train.iloc[0, :4].round(4).values}")
print(f"\n  After scaling (same wine):")
print(f"  {X_train_scaled[0, :4].round(4)}")
print(f"\n  Scaled mean ~0:  {X_train_scaled.mean(axis=0)[:4].round(3)}")
print(f"  Scaled std  ~1:  {X_train_scaled.std(axis=0)[:4].round(3)}")


# ============================================================================
# STEP 4: HOW KNN WORKS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HOW KNN WORKS")
print("=" * 80)

print("""
  K-Nearest Neighbors (KNN) Algorithm:
  ============================================================

  TRAINING PHASE:
    Simply stores all training data. Nothing else.
    (KNN is a "lazy learner" - no model is built)

  PREDICTION PHASE (for a new wine sample):
    1. Calculate distance from new wine to ALL stored wines
       Euclidean: d = sqrt(sum((x1-x2)^2 for each feature))

    2. Find the K closest wines (nearest neighbors)

    3. Take a VOTE among those K neighbors
       Most common quality tier wins -> prediction

  EXAMPLE (K=5):
    New wine's 5 nearest neighbors:
      Wine A -> HIGH
      Wine B -> HIGH
      Wine C -> MEDIUM
      Wine D -> HIGH
      Wine E -> MEDIUM

    Vote: HIGH=3, MEDIUM=2, LOW=0
    Prediction: HIGH quality

  CHOOSING K:
    K too small (K=1): Very sensitive to noise, overfitting
    K too large (K=100): Too many neighbors, underfitting
    Sweet spot: Usually sqrt(n_training_samples)
    sqrt(1200) = ~35, but we test a range to find the best

  DISTANCE METRICS:
    Euclidean (default): straight-line distance
    Manhattan: sum of absolute differences
    Minkowski: generalization of both
""")


# ============================================================================
# STEP 5: FIND OPTIMAL K
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: FIND OPTIMAL K (ELBOW METHOD)")
print("=" * 80)

print(f"\n  Testing K from 1 to 50...")
print(f"  Using 5-fold cross-validation for each K\n")

k_range     = range(1, 51)
cv_scores   = []
train_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    cv  = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(cv.mean())

    knn.fit(X_train_scaled, y_train)
    train_scores.append(knn.score(X_train_scaled, y_train))

best_k      = k_range[np.argmax(cv_scores)]
best_cv_acc = max(cv_scores)

print(f"  {'K':<6} {'Train Acc':>10} {'CV Acc':>10}")
print(f"  {'-'*30}")
for k, tr, cv in zip(k_range, train_scores, cv_scores):
    marker = " <-- BEST" if k == best_k else ""
    if k <= 15 or k == best_k:
        print(f"  {k:<6} {tr:>10.4f} {cv:>10.4f}{marker}")
    elif k == 16:
        print(f"  ...")

print(f"\n  Best K = {best_k}  (CV Accuracy = {best_cv_acc:.4f})")
print(f"  sqrt(n_train) = {int(np.sqrt(len(y_train)))}  (rule-of-thumb)")


# ============================================================================
# STEP 6: BUILD KNN MODEL WITH BEST K
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: BUILD KNN MODEL WITH OPTIMAL K")
print("=" * 80)

knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn_best.fit(X_train_scaled, y_train)

y_pred      = knn_best.predict(X_test_scaled)
y_pred_prob = knn_best.predict_proba(X_test_scaled)

test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average='weighted')

print(f"\n  KNN Model (K={best_k}, Euclidean Distance)")
print(f"  Test Accuracy:    {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Weighted F1:      {test_f1:.4f}")

print(f"\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['HIGH', 'LOW', 'MEDIUM']))

cm = confusion_matrix(y_test, y_pred, labels=['LOW', 'MEDIUM', 'HIGH'])
print(f"\n--- Confusion Matrix (rows=Actual, cols=Predicted) ---")
print(f"          {'LOW':>8} {'MEDIUM':>8} {'HIGH':>8}")
for i, label in enumerate(['LOW', 'MEDIUM', 'HIGH']):
    print(f"  {label:<8} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8}")


# ============================================================================
# STEP 7: COMPARE DISTANCE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: COMPARE DISTANCE METRICS")
print("=" * 80)

print(f"\n  Testing different distance metrics (K={best_k}):")

metrics_to_test = {
    'Euclidean':  {'metric': 'euclidean'},
    'Manhattan':  {'metric': 'manhattan'},
    'Minkowski':  {'metric': 'minkowski', 'p': 3},
    'Chebyshev':  {'metric': 'chebyshev'},
}

metric_results = {}
print(f"\n  {'Metric':<15} {'Test Accuracy':>14} {'F1-Score':>10} {'CV Mean':>10}")
print(f"  {'-'*52}")

for name, params in metrics_to_test.items():
    knn_m = KNeighborsClassifier(n_neighbors=best_k, **params)
    knn_m.fit(X_train_scaled, y_train)
    y_m   = knn_m.predict(X_test_scaled)
    acc_m = accuracy_score(y_test, y_m)
    f1_m  = f1_score(y_test, y_m, average='weighted')
    cv_m  = cross_val_score(knn_m, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    metric_results[name] = {'acc': acc_m, 'f1': f1_m, 'cv': cv_m}
    print(f"  {name:<15} {acc_m:>14.4f} {f1_m:>10.4f} {cv_m:>10.4f}")

best_metric = max(metric_results, key=lambda k: metric_results[k]['acc'])
print(f"\n  Best Metric: {best_metric}  (Acc={metric_results[best_metric]['acc']:.4f})")


# ============================================================================
# STEP 8: COMPARE K=1 vs BEST K vs LARGE K
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: EFFECT OF K - UNDERFITTING vs OVERFITTING")
print("=" * 80)

k_compare = {
    f'K=1  (Overfit)':   1,
    f'K={best_k}  (Optimal)': best_k,
    'K=50 (Underfit)':   50,
}

print(f"\n  {'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}")
print(f"  {'-'*52}")

for name, k in k_compare.items():
    knn_k = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_k.fit(X_train_scaled, y_train)
    tr_acc = knn_k.score(X_train_scaled, y_train)
    te_acc = knn_k.score(X_test_scaled, y_test)
    print(f"  {name:<22} {tr_acc:>10.4f} {te_acc:>10.4f} {tr_acc-te_acc:>8.4f}")

print(f"""
  Observations:
    K=1:         Memorizes training data (Train=1.0), overfits on test
    K={best_k}:        Generalizes well, small gap between train/test
    K=50:        Too smooth, underfits, misses patterns
""")


# ============================================================================
# STEP 9: CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: CROSS-VALIDATION")
print("=" * 80)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(knn_best, X_train_scaled, y_train, cv=skf, scoring='accuracy')
cv_f1  = cross_val_score(knn_best, X_train_scaled, y_train, cv=skf, scoring='f1_weighted')

print(f"\n  5-Fold Stratified Cross-Validation (K={best_k}):")
print(f"  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}  | Folds: {cv_acc.round(4)}")
print(f"  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}  | Folds: {cv_f1.round(4)}")
print(f"  Consistency: {'Excellent' if cv_acc.std() < 0.02 else 'Good' if cv_acc.std() < 0.05 else 'Variable'}")


# ============================================================================
# STEP 10: SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAMPLE WINE PREDICTIONS")
print("=" * 80)

print(f"\n  {'Wine':<7} {'Actual':<10} {'Predicted':<12} {'Conf%':>7} {'Correct?':>9}")
print(f"  {'-'*48}")

classes = knn_best.classes_
for i in range(20):
    actual    = y_test.iloc[i]
    predicted = y_pred[i]
    probs     = y_pred_prob[i]
    conf      = probs.max() * 100
    correct   = "Correct" if predicted == actual else "WRONG"
    print(f"  Wine {i+1:<3} {actual:<10} {predicted:<12} {conf:>6.1f}%  {correct:>9}")


# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

tier_colors  = {'LOW': '#E53935', 'MEDIUM': '#FB8C00', 'HIGH': '#43A047'}
tier_order   = ['LOW', 'MEDIUM', 'HIGH']

# --- Viz 1: Class Distribution + Key Features ---
print("\n  Creating class distribution and feature plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

counts = [class_counts.get(t, 0) for t in tier_order]
colors = [tier_colors[t] for t in tier_order]
axes[0].bar(tier_order, counts, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Wine Quality Tier Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Wines', fontsize=11, fontweight='bold')
for bar, count in zip(axes[0].patches, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 f'{count}\n({count/n_samples*100:.0f}%)', ha='center', fontsize=11, fontweight='bold')

for tier in tier_order:
    subset = df[df['QualityTier'] == tier]
    axes[1].hist(subset['Alcohol'], bins=20, alpha=0.65,
                 color=tier_colors[tier], label=tier, edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('Alcohol (% vol)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].set_title('Alcohol by Quality Tier', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

for tier in tier_order:
    subset = df[df['QualityTier'] == tier]
    axes[2].hist(subset['VolatileAcidity'], bins=20, alpha=0.65,
                 color=tier_colors[tier], label=tier, edgecolor='black', linewidth=0.5)
axes[2].set_xlabel('Volatile Acidity (g/L)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[2].set_title('Volatile Acidity by Quality Tier', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('knn_viz_1_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: knn_viz_1_distribution.png")

# --- Viz 2: Elbow Curve ---
print("  Creating elbow curve (optimal K)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), cv_scores, 'b-o', markersize=5, linewidth=2, label='CV Accuracy')
axes[0].plot(list(k_range), train_scores, 'r--s', markersize=4, linewidth=1.5, alpha=0.6, label='Train Accuracy')
axes[0].axvline(x=best_k, color='green', linestyle='--', linewidth=2.5, label=f'Optimal K={best_k}')
axes[0].set_xlabel('Number of Neighbors (K)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Elbow Curve — Finding Optimal K', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

k_names = list(k_compare.keys())
k_vals  = list(k_compare.values())
tr_accs, te_accs = [], []
for k in k_vals:
    knn_tmp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_tmp.fit(X_train_scaled, y_train)
    tr_accs.append(knn_tmp.score(X_train_scaled, y_train))
    te_accs.append(knn_tmp.score(X_test_scaled, y_test))

x_pos = np.arange(len(k_names))
width = 0.35
axes[1].bar(x_pos - width/2, tr_accs, width, label='Train', color='#42A5F5', edgecolor='black', alpha=0.85)
axes[1].bar(x_pos + width/2, te_accs, width, label='Test',  color='#66BB6A', edgecolor='black', alpha=0.85)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(k_names, fontsize=9)
axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('K Comparison: Overfit vs Optimal vs Underfit', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 1.1)
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('knn_viz_2_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: knn_viz_2_elbow.png")

# --- Viz 3: Confusion Matrix ---
print("  Creating confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=['LOW', 'MEDIUM', 'HIGH'],
            yticklabels=['LOW', 'MEDIUM', 'HIGH'],
            annot_kws={"size": 14, "weight": "bold"})
ax.set_xlabel('Predicted Quality', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Quality', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix (K={best_k}, Euclidean)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('knn_viz_3_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: knn_viz_3_confusion.png")

# --- Viz 4: PCA 2D Visualization ---
print("  Creating PCA 2D decision boundary plot...")
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn_pca.fit(X_train_pca, y_train)

h = 0.08
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])

label_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
Z_num     = np.array([label_map[z] for z in Z]).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].contourf(xx, yy, Z_num, alpha=0.25,
                 cmap=plt.cm.RdYlGn, levels=[-0.5, 0.5, 1.5, 2.5])
for tier in tier_order:
    mask  = y_train == tier
    pc_pts = X_train_pca[mask]
    axes[0].scatter(pc_pts[:, 0], pc_pts[:, 1],
                    c=tier_colors[tier], label=tier, s=20, alpha=0.6, edgecolors='none')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
axes[0].set_title(f'KNN Decision Boundary (K={best_k}) — PCA Space', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)

for tier in tier_order:
    mask = np.array(y_test) == tier
    axes[1].scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                    c=tier_colors[tier], label=f'Actual {tier}', s=40, alpha=0.7,
                    edgecolors='black', linewidths=0.4)
wrong_mask = np.array(y_test) != y_pred
axes[1].scatter(X_test_pca[wrong_mask, 0], X_test_pca[wrong_mask, 1],
                marker='x', c='black', s=80, label='Misclassified', linewidths=2, zorder=5)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
axes[1].set_title('Test Set: Actual vs Misclassified', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle('KNN in PCA-Reduced Feature Space', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('knn_viz_4_pca_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: knn_viz_4_pca_boundary.png")

# --- Viz 5: Feature Boxplots ---
print("  Creating feature boxplots by tier...")
key_features = ['Alcohol', 'VolatileAcidity', 'Sulphates', 'CitricAcid', 'pH', 'Chlorides']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    data_by_tier = [df[df['QualityTier'] == t][feat].values for t in tier_order]
    bp = axes[i].boxplot(data_by_tier, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, tier in zip(bp['boxes'], tier_order):
        patch.set_facecolor(tier_colors[tier])
        patch.set_alpha(0.8)
    axes[i].set_xticklabels(tier_order, fontsize=10)
    axes[i].set_title(f'{feat}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value', fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Chemical Feature Distribution by Quality Tier', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('knn_viz_5_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: knn_viz_5_boxplots.png")

# --- Viz 6: Distance Metric Comparison ---
print("  Creating distance metric comparison...")
m_names = list(metric_results.keys())
m_acc   = [metric_results[m]['acc'] for m in m_names]
m_f1    = [metric_results[m]['f1']  for m in m_names]
m_cv    = [metric_results[m]['cv']  for m in m_names]

x_pos = np.arange(len(m_names))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x_pos - width, m_acc, width, label='Test Accuracy', color='#42A5F5', edgecolor='black', alpha=0.85)
ax.bar(x_pos,         m_f1,  width, label='F1-Score',      color='#66BB6A', edgecolor='black', alpha=0.85)
ax.bar(x_pos + width, m_cv,  width, label='CV Accuracy',   color='#FFA726', edgecolor='black', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(m_names, fontsize=11)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Distance Metric Comparison (K={best_k})', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('knn_viz_6_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: knn_viz_6_metrics.png")

# --- Viz 7: Prediction Confidence Distribution ---
print("  Creating prediction confidence plot...")
max_probs = y_pred_prob.max(axis=1)
correct   = (y_pred == np.array(y_test))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(max_probs[correct],  bins=20, alpha=0.7, color='#43A047',
             label='Correct', edgecolor='black')
axes[0].hist(max_probs[~correct], bins=20, alpha=0.7, color='#E53935',
             label='Wrong',   edgecolor='black')
axes[0].set_xlabel('Prediction Confidence (max probability)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Wines', fontsize=12, fontweight='bold')
axes[0].set_title('Confidence Distribution: Correct vs Wrong', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

prob_df = pd.DataFrame(y_pred_prob, columns=knn_best.classes_)
prob_df['Actual'] = np.array(y_test)
prob_df['Correct'] = correct

for tier in tier_order:
    subset = prob_df[prob_df['Actual'] == tier]
    axes[1].scatter(range(len(subset)), subset[tier].values,
                    c=tier_colors[tier], label=f'{tier} (P own class)', s=15, alpha=0.7)
axes[1].set_xlabel('Test Wine Index', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Probability for True Class', fontsize=11, fontweight='bold')
axes[1].set_title('Confidence for True Class by Tier', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='50% line')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_viz_7_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: knn_viz_7_confidence.png")


# ============================================================================
# STEP 12: COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'='*80}
KNN CLASSIFICATION - WINE QUALITY PREDICTION
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
Classify red wine quality (LOW / MEDIUM / HIGH) from chemical lab measurements,
enabling wineries to predict quality before expensive human tasting panels,
flag defective batches early, and optimize production chemistry.

WHY KNN FOR WINE QUALITY?
  - Wines with similar chemistry taste similar (locality = prediction)
  - Multi-class problem handled naturally (majority vote among K neighbors)
  - No assumption about data distribution
  - Interpretable: prediction explained by citing similar wines
  - Chemical similarity IS the domain logic

DATASET SUMMARY
{'='*80}
  Total wine samples: {n_samples}
  Features:           {len(feature_columns)}
  LOW quality:        {n_low} wines ({n_low/n_samples*100:.1f}%)
  MEDIUM quality:     {n_med} wines ({n_med/n_samples*100:.1f}%)
  HIGH quality:       {n_high} wines ({n_high/n_samples*100:.1f}%)

  Train / Test split: 80% / 20% (stratified)

KEY CHEMICAL DIFFERENCES BY TIER
{'='*80}
{df.groupby('QualityTier')[feature_columns].mean().round(3).to_string()}

OPTIMAL K SELECTION
{'='*80}
  Method:     5-fold cross-validation across K=1 to K=50
  Best K:     {best_k}
  CV Accuracy at best K: {best_cv_acc:.4f}
  Rule-of-thumb sqrt(n_train) = {int(np.sqrt(len(y_train)))}

K SENSITIVITY ANALYSIS
{'='*80}
  K=1   (Overfit)  -> Memorizes training, poor generalization
  K={best_k:<3}  (Optimal) -> Best balance, highest CV accuracy
  K=50  (Underfit) -> Over-smoothed, misses patterns

DISTANCE METRIC COMPARISON (K={best_k})
{'='*80}
  {'Metric':<15} {'Test Acc':>10} {'F1-Score':>10} {'CV Acc':>10}
  {'-'*47}
{chr(10).join([f"  {m:<15} {metric_results[m]['acc']:>10.4f} {metric_results[m]['f1']:>10.4f} {metric_results[m]['cv']:>10.4f}"
               for m in metric_results])}
  Best Metric: {best_metric}

FINAL MODEL PERFORMANCE (K={best_k}, Euclidean)
{'='*80}
  Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)
  Weighted F1:     {test_f1:.4f}

  Classification Report:
{classification_report(y_test, y_pred, target_names=['HIGH', 'LOW', 'MEDIUM'])}

  Confusion Matrix (rows=Actual, cols=Predicted):
  {'':<8} {'LOW':>8} {'MEDIUM':>8} {'HIGH':>8}
  LOW     {cm[0,0]:>8} {cm[0,1]:>8} {cm[0,2]:>8}
  MEDIUM  {cm[1,0]:>8} {cm[1,1]:>8} {cm[1,2]:>8}
  HIGH    {cm[2,0]:>8} {cm[2,1]:>8} {cm[2,2]:>8}

CROSS-VALIDATION (5-Fold Stratified)
{'='*80}
  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}
  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}

KNN ADVANTAGES IN THIS SCENARIO
{'='*80}
  - No model training: stores all samples, predicts by similarity
  - Naturally multi-class: voting among K neighbors
  - Non-parametric: no assumption about decision boundary shape
  - Local patterns captured: each region of chemical space independent
  - Probability output: confidence score per prediction

KNN LIMITATIONS
{'='*80}
  - Slow at prediction time: computes distance to ALL training wines
  - Memory-heavy: must store entire dataset
  - Sensitive to irrelevant features (all features used in distance)
  - Needs feature scaling (critical!)
  - Performance degrades in very high dimensions

BUSINESS RECOMMENDATIONS
{'='*80}
  1. Deploy at bottling stage: run chemical analysis, get quality prediction
  2. Flag LOW quality batches for blending or re-processing
  3. Prioritize HIGH quality batches for premium labeling
  4. Monitor volatile acidity closely (strongest LOW quality signal)
  5. Optimize alcohol levels (strongest HIGH quality signal)
  6. Retrain model quarterly as new vintage data arrives

FILES GENERATED
{'='*80}
  Data:
    wine_quality_data.csv

  Visualizations:
    knn_viz_1_distribution.png   - Class distribution + key features
    knn_viz_2_elbow.png          - Optimal K elbow curve
    knn_viz_3_confusion.png      - Confusion matrix
    knn_viz_4_pca_boundary.png   - Decision boundary in PCA space
    knn_viz_5_boxplots.png       - Chemical features by tier
    knn_viz_6_metrics.png        - Distance metric comparison
    knn_viz_7_confidence.png     - Prediction confidence analysis

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)
with open('knn_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("  Report saved to: knn_analysis_report.txt")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KNN WINE QUALITY CLASSIFICATION COMPLETE!")
print("=" * 80)
print(f"\n  Summary:")
print(f"    Generated {n_samples} wine samples (LOW / MEDIUM / HIGH)")
print(f"    Optimal K = {best_k}  (found via elbow method)")
print(f"    Test Accuracy  = {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"    Weighted F1    = {test_f1:.4f}")
print(f"    Best metric    = {best_metric}")
print(f"    CV Accuracy    = {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"    7 visualizations generated")

print(f"\n  Key Findings:")
print(f"    - Alcohol is the #1 separator (LOW ~9.8% vs HIGH ~11.8% vol)")
print(f"    - Volatile acidity: LOW wines average {df[df['QualityTier']=='LOW']['VolatileAcidity'].mean():.3f} g/L vs HIGH {df[df['QualityTier']=='HIGH']['VolatileAcidity'].mean():.3f} g/L")
print(f"    - KNN works because similar chemistry = similar quality")
print(f"    - Scaling was critical: without it features dominate by range")
print(f"    - K=1 overfits badly, K=50 underfits, K={best_k} is the sweet spot")
print("\n" + "=" * 80)
print("All analysis complete!")
print("=" * 80)