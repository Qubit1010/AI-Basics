"""
DECISION TREE CLASSIFICATION - STUDENT DROPOUT RISK PREDICTION
===============================================================
Classifying students as LOW RISK / AT RISK / HIGH RISK of dropping out
based on academic performance and engagement metrics

Perfect Scenario for Decision Tree:
- Natural if-then rules match how advisors already think
  ("IF GPA < 2.0 AND attendance < 60% THEN high dropout risk")
- Mix of numerical and categorical features handled natively
- No feature scaling required
- Highly interpretable: university must explain decisions to students
- Non-linear boundaries between risk levels
- Feature interactions matter (GPA drop + low engagement = very bad)

Dataset: University Student Records (Generated)
Features:
- GPA                  (0.0 - 4.0 scale)
- AttendanceRate       (% of classes attended)
- AssignmentsCompleted (% submitted)
- FailedCourses        (count this semester)
- CreditHoursEnrolled  (course load)
- StudyHoursPerWeek    (self-reported)
- LibraryVisitsMonth   (engagement proxy)
- FinancialAidStatus   (Yes / No)
- FirstGenStudent      (first in family to attend college)
- SemesterNumber       (1-8, how far along)
- GPATrend             (improving / stable / declining)
- ExtracurricularCount (clubs/activities)

Target: DropoutRisk
  - LOW RISK  (likely to graduate on track)
  - AT RISK   (needs monitoring and support)
  - HIGH RISK (needs immediate intervention)

Why Decision Tree for Student Advising?
- Rules are explainable to students and parents
- Advisors can trace exactly WHY a student is flagged
- Regulatory compliance: universities must justify interventions
- Handles mixed data types without preprocessing
- Fast prediction for large student populations
- Visual tree = training material for new advisors

Approach:
1. Generate realistic student academic data
2. Exploratory Data Analysis
3. Build Decision Tree (no scaling needed!)
4. Visualize the tree structure
5. Prune tree (max_depth tuning)
6. Feature importance analysis
7. Rule extraction
8. Comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("DECISION TREE CLASSIFICATION - STUDENT DROPOUT RISK PREDICTION")
print("=" * 80)


# ============================================================================
# STEP 1: GENERATE REALISTIC STUDENT DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC STUDENT ACADEMIC DATA")
print("=" * 80)

np.random.seed(42)

n_low  = 600   # Most students are fine
n_at   = 500   # Significant at-risk group
n_high = 300   # Smaller high-risk group
n_total = n_low + n_at + n_high

print(f"\nGenerating {n_total} student records...")
print(f"  LOW RISK  : {n_low}  students")
print(f"  AT RISK   : {n_at}  students")
print(f"  HIGH RISK : {n_high}  students")

# --- LOW RISK students ---
low = {
    'GPA':                  np.random.normal(3.4,  0.4,  n_low).clip(2.5, 4.0),
    'AttendanceRate':       np.random.normal(88,   7,    n_low).clip(65, 100),
    'AssignmentsCompleted': np.random.normal(91,   6,    n_low).clip(70, 100),
    'FailedCourses':        np.random.poisson(0.2, n_low).clip(0, 2),
    'CreditHoursEnrolled':  np.random.normal(15,   1.5,  n_low).clip(9, 18),
    'StudyHoursPerWeek':    np.random.normal(18,   4,    n_low).clip(8, 35),
    'LibraryVisitsMonth':   np.random.poisson(7,   n_low).clip(0, 20),
    'FinancialAidStatus':   np.random.choice(['Yes', 'No'], n_low, p=[0.55, 0.45]),
    'FirstGenStudent':      np.random.choice(['Yes', 'No'], n_low, p=[0.30, 0.70]),
    'SemesterNumber':       np.random.randint(1, 9, n_low),
    'GPATrend':             np.random.choice(['Improving', 'Stable', 'Declining'],
                                             n_low, p=[0.40, 0.50, 0.10]),
    'ExtracurricularCount': np.random.poisson(2.5, n_low).clip(0, 6),
    'DropoutRisk':          ['LOW RISK'] * n_low
}

# --- AT RISK students ---
at_risk = {
    'GPA':                  np.random.normal(2.4,  0.4,  n_at).clip(1.5, 3.2),
    'AttendanceRate':       np.random.normal(72,   10,   n_at).clip(45, 90),
    'AssignmentsCompleted': np.random.normal(74,   10,   n_at).clip(45, 92),
    'FailedCourses':        np.random.poisson(1.2, n_at).clip(0, 4),
    'CreditHoursEnrolled':  np.random.normal(13,   2.5,  n_at).clip(6, 18),
    'StudyHoursPerWeek':    np.random.normal(11,   4,    n_at).clip(3, 22),
    'LibraryVisitsMonth':   np.random.poisson(3,   n_at).clip(0, 12),
    'FinancialAidStatus':   np.random.choice(['Yes', 'No'], n_at, p=[0.65, 0.35]),
    'FirstGenStudent':      np.random.choice(['Yes', 'No'], n_at, p=[0.45, 0.55]),
    'SemesterNumber':       np.random.randint(1, 9, n_at),
    'GPATrend':             np.random.choice(['Improving', 'Stable', 'Declining'],
                                             n_at, p=[0.15, 0.40, 0.45]),
    'ExtracurricularCount': np.random.poisson(1.2, n_at).clip(0, 5),
    'DropoutRisk':          ['AT RISK'] * n_at
}

# --- HIGH RISK students ---
high = {
    'GPA':                  np.random.normal(1.6,  0.4,  n_high).clip(0.0, 2.4),
    'AttendanceRate':       np.random.normal(52,   12,   n_high).clip(10, 75),
    'AssignmentsCompleted': np.random.normal(53,   13,   n_high).clip(10, 75),
    'FailedCourses':        np.random.poisson(2.8, n_high).clip(0, 6),
    'CreditHoursEnrolled':  np.random.normal(10,   3,    n_high).clip(3, 16),
    'StudyHoursPerWeek':    np.random.normal(5,    3,    n_high).clip(0, 15),
    'LibraryVisitsMonth':   np.random.poisson(1,   n_high).clip(0, 6),
    'FinancialAidStatus':   np.random.choice(['Yes', 'No'], n_high, p=[0.70, 0.30]),
    'FirstGenStudent':      np.random.choice(['Yes', 'No'], n_high, p=[0.55, 0.45]),
    'SemesterNumber':       np.random.randint(1, 9, n_high),
    'GPATrend':             np.random.choice(['Improving', 'Stable', 'Declining'],
                                             n_high, p=[0.05, 0.20, 0.75]),
    'ExtracurricularCount': np.random.poisson(0.4, n_high).clip(0, 3),
    'DropoutRisk':          ['HIGH RISK'] * n_high
}

df = pd.concat([
    pd.DataFrame(low),
    pd.DataFrame(at_risk),
    pd.DataFrame(high)
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

feature_columns = [
    'GPA', 'AttendanceRate', 'AssignmentsCompleted', 'FailedCourses',
    'CreditHoursEnrolled', 'StudyHoursPerWeek', 'LibraryVisitsMonth',
    'FinancialAidStatus', 'FirstGenStudent', 'SemesterNumber',
    'GPATrend', 'ExtracurricularCount'
]

print(f"\n  Dataset shape: {df.shape}")
print(f"  Features:      {len(feature_columns)}")

print("\n--- First 10 Student Records ---")
print(df.head(10).to_string(index=False))

df.to_csv('student_dropout_data.csv', index=False, encoding='utf-8')
print(f"\n  Dataset saved to: student_dropout_data.csv")


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\n--- Class Distribution ---")
class_counts = df['DropoutRisk'].value_counts()
for risk, count in class_counts.items():
    print(f"  {risk:<12}: {count} students ({count/n_total*100:.1f}%)")

print(f"\n--- Mean Feature Values by Risk Level ---")
num_cols = ['GPA', 'AttendanceRate', 'AssignmentsCompleted', 'FailedCourses',
            'StudyHoursPerWeek', 'LibraryVisitsMonth', 'ExtracurricularCount']
print(df.groupby('DropoutRisk')[num_cols].mean().round(2))

print(f"\n--- Categorical Feature Breakdown ---")
for cat in ['FinancialAidStatus', 'FirstGenStudent', 'GPATrend']:
    print(f"\n  {cat}:")
    ct = pd.crosstab(df[cat], df['DropoutRisk'], normalize='columns').round(3) * 100
    print(ct)

print(f"\n--- Key Risk Signals ---")
risk_order = ['LOW RISK', 'AT RISK', 'HIGH RISK']
group = df.groupby('DropoutRisk')
print(f"\n  {'Feature':<25} {'LOW RISK':>10} {'AT RISK':>10} {'HIGH RISK':>10}  Signal Strength")
print(f"  {'-'*75}")
for feat in num_cols:
    vals = [group[feat].mean().get(r, 0) for r in risk_order]
    spread = max(vals) - min(vals)
    avg    = np.mean(vals)
    strength = "STRONG" if spread/avg > 0.4 else ("MODERATE" if spread/avg > 0.2 else "WEAK")
    print(f"  {feat:<25} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f}  {strength}")


# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

print("""
  KEY ADVANTAGE OF DECISION TREE:
  ============================================================
  NO FEATURE SCALING REQUIRED!

  Decision Trees split on feature thresholds, not distances.
  Whether GPA is 2.5 or standardized to -1.2, the tree
  finds the same optimal split point.

  We only need to:
  - Encode categorical variables as numbers
  - Handle any missing values
  - That's it!

  Compare to:
  - KNN:  MUST scale (distance-based)
  - SVM:  MUST scale (margin/kernel-based)
  - LR:   SHOULD scale (gradient descent)
  - DT:   NO scaling needed (threshold-based splits)
""")

df_encoded = df.copy()

le_aid   = LabelEncoder()
le_gen   = LabelEncoder()
le_trend = LabelEncoder()

df_encoded['FinancialAidStatus'] = le_aid.fit_transform(df['FinancialAidStatus'])
df_encoded['FirstGenStudent']    = le_gen.fit_transform(df['FirstGenStudent'])
df_encoded['GPATrend']           = le_trend.fit_transform(df['GPATrend'])

print(f"  Label encodings applied:")
print(f"  FinancialAidStatus: {dict(zip(le_aid.classes_, le_aid.transform(le_aid.classes_)))}")
print(f"  FirstGenStudent:    {dict(zip(le_gen.classes_, le_gen.transform(le_gen.classes_)))}")
print(f"  GPATrend:           {dict(zip(le_trend.classes_, le_trend.transform(le_trend.classes_)))}")

X = df_encoded[feature_columns]
y = df_encoded['DropoutRisk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train: {X_train.shape[0]} students | Test: {X_test.shape[0]} students")
print(f"  No scaling applied - Decision Tree handles raw values natively")


# ============================================================================
# STEP 4: HOW DECISION TREES WORK
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HOW DECISION TREES WORK")
print("=" * 80)

print("""
  Decision Tree Algorithm:
  ============================================================

  BUILDING THE TREE (top-down, greedy):

    At each node, find the BEST feature + threshold to split on.
    "Best" = maximizes purity of resulting groups.

  SPLITTING CRITERIA:

    Gini Impurity (default):
      Gini = 1 - sum(p_i^2)
      Pure node (all same class): Gini = 0
      Maximally mixed node:       Gini = 0.5

      Example split on GPA < 2.0:
        Left:  90% HIGH RISK, 10% AT RISK  -> Gini = 1-(0.9^2+0.1^2) = 0.18
        Right: 10% HIGH RISK, 90% other    -> Gini = 0.18
        Weighted Gini < parent Gini = GOOD split!

    Entropy / Information Gain:
      Entropy = -sum(p_i * log2(p_i))
      Measures information disorder. Split that reduces entropy most wins.

  STOPPING:
    - max_depth: limit tree height (prevents overfitting)
    - min_samples_split: node needs enough samples to split
    - min_samples_leaf: leaf needs minimum samples

  PREDICTION:
    Follow the if-then path from root to a leaf.
    The leaf's majority class = prediction.

  EXAMPLE RULE LEARNED:
    IF GPA < 2.0:
      IF AttendanceRate < 60%:
        THEN HIGH RISK
      ELSE:
        THEN AT RISK
    ELSE IF GPA >= 3.0:
      THEN LOW RISK
    ELSE:
      THEN AT RISK
""")


# ============================================================================
# STEP 5: DEPTH TUNING (OVERFITTING ANALYSIS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TREE DEPTH TUNING")
print("=" * 80)

print(f"\n  Testing max_depth from 1 to 20 (None = unlimited)")
print(f"  Using 5-fold cross-validation\n")

depths       = list(range(1, 21)) + [None]
depth_results = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=42)
    dt.fit(X_train, y_train)
    tr_acc = dt.score(X_train, y_train)
    te_acc = dt.score(X_test,  y_test)
    cv     = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy').mean()
    depth_results.append({
        'Depth':    str(d) if d else 'None',
        'TrainAcc': tr_acc,
        'TestAcc':  te_acc,
        'CVAcc':    cv,
        'Gap':      tr_acc - te_acc
    })

depth_df   = pd.DataFrame(depth_results)
best_idx   = depth_df['CVAcc'].idxmax()
best_depth = depths[best_idx]

print(f"  {'Depth':<7} {'Train Acc':>10} {'Test Acc':>10} {'CV Acc':>10} {'Gap':>8}")
print(f"  {'-'*50}")
for _, row in depth_df.iterrows():
    marker = " <-- BEST" if row['Depth'] == str(best_depth) or \
             (best_depth is None and row['Depth'] == 'None') else ""
    print(f"  {row['Depth']:<7} {row['TrainAcc']:>10.4f} {row['TestAcc']:>10.4f} "
          f"{row['CVAcc']:>10.4f} {row['Gap']:>8.4f}{marker}")

print(f"\n  Best max_depth = {best_depth}  (CV Accuracy = {depth_df['CVAcc'].iloc[best_idx]:.4f})")
print(f"  Unlimited tree (None): Train={depth_df[depth_df['Depth']=='None']['TrainAcc'].values[0]:.4f} "
      f"| Test={depth_df[depth_df['Depth']=='None']['TestAcc'].values[0]:.4f}  <- Overfitting!")


# ============================================================================
# STEP 6: BUILD FINAL DECISION TREE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: BUILD FINAL DECISION TREE")
print("=" * 80)

dt_best = DecisionTreeClassifier(
    max_depth=best_depth,
    criterion='gini',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_best.fit(X_train, y_train)

y_pred     = dt_best.predict(X_test)
y_pred_prob = dt_best.predict_proba(X_test)

train_acc = dt_best.score(X_train, y_train)
test_acc  = accuracy_score(y_test, y_pred)
test_f1   = f1_score(y_test, y_pred, average='weighted')

print(f"\n  Decision Tree (max_depth={best_depth}, Gini, min_samples_leaf=5)")
print(f"  Train Accuracy:  {train_acc:.4f}")
print(f"  Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Weighted F1:     {test_f1:.4f}")
print(f"  Overfitting Gap: {train_acc - test_acc:.4f}")
print(f"  Tree Depth:      {dt_best.get_depth()}")
print(f"  Leaf Nodes:      {dt_best.get_n_leaves()}")

print(f"\n--- Classification Report ---")
risk_order = ['LOW RISK', 'AT RISK', 'HIGH RISK']
print(classification_report(y_test, y_pred, target_names=risk_order))

cm = confusion_matrix(y_test, y_pred, labels=risk_order)
print(f"\n--- Confusion Matrix ---")
print(f"  {'':12} {'LOW RISK':>10} {'AT RISK':>10} {'HIGH RISK':>10}")
for i, label in enumerate(risk_order):
    print(f"  {label:<12} {cm[i,0]:>10} {cm[i,1]:>10} {cm[i,2]:>10}")


# ============================================================================
# STEP 7: COMPARE CRITERIA (GINI vs ENTROPY)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: COMPARE SPLITTING CRITERIA")
print("=" * 80)

criteria_results = {}
print(f"\n  {'Criterion':<12} {'Train Acc':>10} {'Test Acc':>10} {'F1-Score':>10} {'CV Mean':>10} {'Leaves':>8}")
print(f"  {'-'*62}")

for crit in ['gini', 'entropy', 'log_loss']:
    dt_c = DecisionTreeClassifier(max_depth=best_depth, criterion=crit,
                                   min_samples_leaf=5, random_state=42)
    dt_c.fit(X_train, y_train)
    y_c    = dt_c.predict(X_test)
    tr_acc = dt_c.score(X_train, y_train)
    te_acc = accuracy_score(y_test, y_c)
    f1_c   = f1_score(y_test, y_c, average='weighted')
    cv_c   = cross_val_score(dt_c, X_train, y_train, cv=5, scoring='accuracy').mean()
    leaves = dt_c.get_n_leaves()
    criteria_results[crit] = {'train': tr_acc, 'test': te_acc, 'f1': f1_c, 'cv': cv_c}
    print(f"  {crit:<12} {tr_acc:>10.4f} {te_acc:>10.4f} {f1_c:>10.4f} {cv_c:>10.4f} {leaves:>8}")

best_crit = max(criteria_results, key=lambda k: criteria_results[k]['test'])
print(f"\n  Best Criterion: {best_crit}  (Test Acc = {criteria_results[best_crit]['test']:.4f})")


# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("""
  Decision Tree Feature Importance = Gini Impurity Reduction
  Each feature's importance = total reduction in Gini impurity
  it causes across all splits in the tree, weighted by samples.
  More important features appear closer to the root.
""")

importance_df = pd.DataFrame({
    'Feature':    feature_columns,
    'Importance': dt_best.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(f"  {'Rank':<6} {'Feature':<25} {'Importance':>12} {'Bar'}")
print(f"  {'-'*65}")
for i, row in importance_df.iterrows():
    bar = '#' * int(row['Importance'] * 50)
    print(f"  {i+1:<6} {row['Feature']:<25} {row['Importance']:>12.4f}  {bar}")

print(f"\n  Top 3 most important features:")
for _, row in importance_df.head(3).iterrows():
    print(f"    - {row['Feature']}: {row['Importance']*100:.2f}% of total importance")

print(f"\n  Features with zero importance (never used in any split):")
zero_imp = importance_df[importance_df['Importance'] == 0]
if len(zero_imp) > 0:
    for _, row in zero_imp.iterrows():
        print(f"    - {row['Feature']}")
else:
    print(f"    None — all features contributed to at least one split")


# ============================================================================
# STEP 9: EXTRACT HUMAN-READABLE RULES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: EXTRACT DECISION RULES")
print("=" * 80)

print("""
  One of Decision Tree's biggest advantages:
  Rules can be extracted and given to university advisors
  as a printed checklist — no computer needed!
""")

# Map encoded values back for readability
trend_map = dict(zip(le_trend.transform(le_trend.classes_), le_trend.classes_))
aid_map   = dict(zip(le_aid.transform(le_aid.classes_),   le_aid.classes_))

tree_rules = export_text(
    dt_best,
    feature_names=feature_columns,
    max_depth=4
)
print("  Decision Tree Rules (first 4 levels):")
print("  " + "-" * 60)
for line in tree_rules.split('\n')[:60]:
    print("  " + line)

print(f"\n  ... (tree has {dt_best.get_n_leaves()} leaves total)")
print(f"\n  Advisor Quick Reference Rules (top patterns):")
print(f"  {'='*55}")
print(f"  HIGH RISK indicators:")
print(f"    GPA < {df[df['DropoutRisk']=='HIGH RISK']['GPA'].quantile(0.75):.1f}  AND")
print(f"    Attendance < {df[df['DropoutRisk']=='HIGH RISK']['AttendanceRate'].quantile(0.75):.0f}%  AND")
print(f"    Failed Courses >= {int(df[df['DropoutRisk']=='HIGH RISK']['FailedCourses'].quantile(0.25))}")
print(f"  LOW RISK indicators:")
print(f"    GPA >= {df[df['DropoutRisk']=='LOW RISK']['GPA'].quantile(0.25):.1f}  AND")
print(f"    Attendance >= {df[df['DropoutRisk']=='LOW RISK']['AttendanceRate'].quantile(0.25):.0f}%  AND")
print(f"    GPA Trend = Improving or Stable")


# ============================================================================
# STEP 10: CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: CROSS-VALIDATION")
print("=" * 80)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(dt_best, X_train, y_train, cv=skf, scoring='accuracy')
cv_f1  = cross_val_score(dt_best, X_train, y_train, cv=skf, scoring='f1_weighted')

print(f"\n  5-Fold Stratified Cross-Validation (max_depth={best_depth}):")
print(f"  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}  | Folds: {cv_acc.round(4)}")
print(f"  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}  | Folds: {cv_f1.round(4)}")
print(f"  Consistency: {'Excellent' if cv_acc.std() < 0.02 else 'Good' if cv_acc.std() < 0.05 else 'Variable'}")


# ============================================================================
# STEP 11: SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: SAMPLE STUDENT PREDICTIONS")
print("=" * 80)

classes = dt_best.classes_
print(f"\n  {'Student':<10} {'Actual':<12} {'Predicted':<12} {'Confidence':>11} {'Correct?':>10}")
print(f"  {'-'*58}")
for i in range(20):
    actual    = y_test.iloc[i]
    predicted = y_pred[i]
    probs     = y_pred_prob[i]
    conf      = probs.max() * 100
    correct   = "Correct" if predicted == actual else "WRONG"
    print(f"  Student {i+1:<3} {actual:<12} {predicted:<12} {conf:>10.1f}%  {correct:>10}")


# ============================================================================
# STEP 12: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

risk_colors = {'LOW RISK': '#1B5E20', 'AT RISK': '#E65100', 'HIGH RISK': '#B71C1C'}
risk_order  = ['LOW RISK', 'AT RISK', 'HIGH RISK']

# --- Viz 1: Distribution + Key Features ---
print("\n  Creating class distribution and feature plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

counts = [class_counts.get(r, 0) for r in risk_order]
colors = [risk_colors[r] for r in risk_order]
bars   = axes[0].bar(risk_order, counts, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Student Risk Level Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Students', fontsize=11, fontweight='bold')
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{count}\n({count/n_total*100:.0f}%)', ha='center',
                 fontsize=11, fontweight='bold')

for risk in risk_order:
    subset = df[df['DropoutRisk'] == risk]
    axes[1].scatter(subset['GPA'], subset['AttendanceRate'],
                    c=risk_colors[risk], label=risk, s=15, alpha=0.4, edgecolors='none')
axes[1].set_xlabel('GPA', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Attendance Rate (%)', fontsize=11, fontweight='bold')
axes[1].set_title('GPA vs Attendance by Risk Level', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

for risk in risk_order:
    subset = df[df['DropoutRisk'] == risk]
    axes[2].scatter(subset['AssignmentsCompleted'], subset['FailedCourses'],
                    c=risk_colors[risk], label=risk, s=15, alpha=0.4, edgecolors='none')
axes[2].set_xlabel('Assignments Completed (%)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Failed Courses', fontsize=11, fontweight='bold')
axes[2].set_title('Assignments vs Failed Courses', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dt_viz_1_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: dt_viz_1_distribution.png")

# --- Viz 2: Decision Tree Structure ---
print("  Creating decision tree structure plot...")
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    dt_best,
    feature_names=feature_columns,
    class_names=dt_best.classes_,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    max_depth=3,
    impurity=True,
    proportion=False
)
ax.set_title(f'Decision Tree Structure (max_depth={best_depth}, showing top 3 levels)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_viz_2_tree.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: dt_viz_2_tree.png")

# --- Viz 3: Depth Tuning Curve ---
print("  Creating depth tuning curve...")
numeric_depths  = depth_df[depth_df['Depth'] != 'None'].copy()
numeric_depths['DepthNum'] = numeric_depths['Depth'].astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(numeric_depths['DepthNum'], numeric_depths['TrainAcc'],
             'b-o', markersize=6, linewidth=2, label='Train Accuracy')
axes[0].plot(numeric_depths['DepthNum'], numeric_depths['TestAcc'],
             'g-s', markersize=6, linewidth=2, label='Test Accuracy')
axes[0].plot(numeric_depths['DepthNum'], numeric_depths['CVAcc'],
             'r-^', markersize=6, linewidth=2, label='CV Accuracy')
if best_depth is not None:
    axes[0].axvline(x=best_depth, color='orange', linestyle='--',
                    linewidth=2.5, label=f'Optimal depth={best_depth}')
axes[0].set_xlabel('Max Tree Depth', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Depth Tuning: Accuracy vs Tree Depth', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(numeric_depths['DepthNum'], numeric_depths['Gap'],
             'r-o', markersize=6, linewidth=2)
axes[1].axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Mild overfit')
axes[1].axhline(y=0.10, color='red',    linestyle='--', linewidth=2, label='High overfit')
if best_depth is not None:
    axes[1].axvline(x=best_depth, color='green', linestyle='--',
                    linewidth=2.5, label=f'Optimal depth={best_depth}')
axes[1].set_xlabel('Max Tree Depth', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Train Acc - Test Acc', fontsize=12, fontweight='bold')
axes[1].set_title('Overfitting vs Tree Depth', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dt_viz_3_depth_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: dt_viz_3_depth_tuning.png")

# --- Viz 4: Confusion Matrix ---
print("  Creating confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],
            xticklabels=risk_order, yticklabels=risk_order,
            annot_kws={"size": 14, "weight": "bold"})
axes[0].set_xlabel('Predicted Risk Level', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual Risk Level',    fontsize=12, fontweight='bold')
axes[0].set_title(f'Confusion Matrix\n(max_depth={best_depth})', fontsize=12, fontweight='bold')

cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=risk_order, yticklabels=risk_order,
            annot_kws={"size": 14, "weight": "bold"})
axes[1].set_xlabel('Predicted Risk Level', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual Risk Level',    fontsize=12, fontweight='bold')
axes[1].set_title('Confusion Matrix (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('dt_viz_4_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: dt_viz_4_confusion.png")

# --- Viz 5: Feature Importance ---
print("  Creating feature importance plot...")
fig, ax = plt.subplots(figsize=(12, 7))
imp_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance_df)))[::-1]
bars = ax.barh(importance_df['Feature'][::-1],
               importance_df['Importance'][::-1],
               color=imp_colors, edgecolor='black', alpha=0.85)
ax.set_xlabel('Feature Importance (Gini Reduction)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance — What Drives Dropout Risk?',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, importance_df['Importance'][::-1]):
    if val > 0.01:
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_viz_5_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: dt_viz_5_importance.png")

# --- Viz 6: Feature Boxplots by Risk ---
print("  Creating feature boxplots by risk level...")
key_features = ['GPA', 'AttendanceRate', 'AssignmentsCompleted',
                'FailedCourses', 'StudyHoursPerWeek', 'LibraryVisitsMonth']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    data_by_risk = [df[df['DropoutRisk'] == r][feat].values for r in risk_order]
    bp = axes[i].boxplot(data_by_risk, patch_artist=True,
                          medianprops=dict(color='black', linewidth=2.5))
    for patch, risk in zip(bp['boxes'], risk_order):
        patch.set_facecolor(risk_colors[risk])
        patch.set_alpha(0.8)
    axes[i].set_xticklabels(['LOW', 'AT RISK', 'HIGH'], fontsize=10)
    axes[i].set_title(feat, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value', fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Academic Feature Distribution by Risk Level', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_viz_6_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: dt_viz_6_boxplots.png")

# --- Viz 7: Criteria + Confidence ---
print("  Creating criteria comparison and confidence plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

crit_names = list(criteria_results.keys())
crit_tr    = [criteria_results[c]['train'] for c in crit_names]
crit_te    = [criteria_results[c]['test']  for c in crit_names]
crit_cv    = [criteria_results[c]['cv']    for c in crit_names]
x_pos = np.arange(len(crit_names))
width = 0.28

axes[0].bar(x_pos - width, crit_tr, width, label='Train Acc', color='#42A5F5', edgecolor='black', alpha=0.85)
axes[0].bar(x_pos,         crit_te, width, label='Test Acc',  color='#66BB6A', edgecolor='black', alpha=0.85)
axes[0].bar(x_pos + width, crit_cv, width, label='CV Acc',    color='#FFA726', edgecolor='black', alpha=0.85)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(crit_names, fontsize=11)
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('Splitting Criterion Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylim(0.7, 1.05)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

max_probs = y_pred_prob.max(axis=1)
correct   = (y_pred == np.array(y_test))
axes[1].hist(max_probs[correct],  bins=20, alpha=0.7, color='#2E7D32',
             label='Correct', edgecolor='black')
axes[1].hist(max_probs[~correct], bins=20, alpha=0.7, color='#C62828',
             label='Wrong',   edgecolor='black')
axes[1].set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Students', fontsize=12, fontweight='bold')
axes[1].set_title('Prediction Confidence: Correct vs Wrong', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('dt_viz_7_criteria_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: dt_viz_7_criteria_confidence.png")


# ============================================================================
# STEP 13: COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 13: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'='*80}
DECISION TREE CLASSIFICATION - STUDENT DROPOUT RISK PREDICTION
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
Predict student dropout risk (LOW RISK / AT RISK / HIGH RISK) from academic
and engagement data, so university advisors can intervene early with targeted
support before students disengage completely.

WHY DECISION TREE FOR STUDENT ADVISING?
  - Rules are fully transparent: advisors trace every decision
  - No black box: students can be told exactly WHY they are flagged
  - Handles mixed data (GPA + attendance + categorical) without scaling
  - Visual tree = training guide for new academic advisors
  - Fast prediction across entire student population each semester

DATASET SUMMARY
{'='*80}
  Total students:  {n_total}
  Features:        {len(feature_columns)}
  LOW RISK:        {n_low} ({n_low/n_total*100:.0f}%)
  AT RISK:         {n_at}  ({n_at/n_total*100:.0f}%)
  HIGH RISK:       {n_high} ({n_high/n_total*100:.0f}%)
  Train/Test:      80% / 20% stratified

PREPROCESSING
{'='*80}
  - NO scaling required (Decision Trees use thresholds, not distances)
  - Label encoding for: FinancialAidStatus, FirstGenStudent, GPATrend
  - Stratified split to preserve class ratios

DEPTH TUNING
{'='*80}
  Tested max_depth: 1 to 20 + unlimited
  Best max_depth:   {best_depth}
  Unlimited tree:   Train=1.0, Test={depth_df[depth_df['Depth']=='None']['TestAcc'].values[0]:.4f} (severe overfit)
  Pruned tree:      Train={train_acc:.4f}, Test={test_acc:.4f} (controlled)

SPLITTING CRITERIA COMPARISON
{'='*80}
  {'Criterion':<12} {'Train Acc':>10} {'Test Acc':>10} {'F1-Score':>10} {'CV Mean':>10}
  {'-'*54}
{chr(10).join([f"  {c:<12} {criteria_results[c]['train']:>10.4f} {criteria_results[c]['test']:>10.4f} {criteria_results[c]['f1']:>10.4f} {criteria_results[c]['cv']:>10.4f}"
               for c in criteria_results])}
  Best criterion: {best_crit}

FINAL MODEL PERFORMANCE (max_depth={best_depth}, {best_crit})
{'='*80}
  Train Accuracy:  {train_acc:.4f}
  Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)
  Weighted F1:     {test_f1:.4f}
  Overfitting Gap: {train_acc - test_acc:.4f}
  Tree Depth:      {dt_best.get_depth()}
  Leaf Nodes:      {dt_best.get_n_leaves()}

  Classification Report:
{classification_report(y_test, y_pred, target_names=risk_order)}

  Confusion Matrix:
  {'':12} {'LOW RISK':>10} {'AT RISK':>10} {'HIGH RISK':>10}
  LOW RISK     {cm[0,0]:>10} {cm[0,1]:>10} {cm[0,2]:>10}
  AT RISK      {cm[1,0]:>10} {cm[1,1]:>10} {cm[1,2]:>10}
  HIGH RISK    {cm[2,0]:>10} {cm[2,1]:>10} {cm[2,2]:>10}

FEATURE IMPORTANCE
{'='*80}
{chr(10).join([f"  {i+1:>2}. {row['Feature']:<25} {row['Importance']:.4f} ({row['Importance']*100:.2f}%)"
               for i, (_, row) in enumerate(importance_df.iterrows())])}

CROSS-VALIDATION (5-Fold Stratified)
{'='*80}
  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}
  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}
  Stability: {'Excellent' if cv_acc.std() < 0.02 else 'Good' if cv_acc.std() < 0.05 else 'Variable'}

DECISION TREE vs OTHER ALGORITHMS
{'='*80}
  vs Logistic Regression:
    DT handles non-linear boundaries naturally
    DT needs no feature scaling
    LR gives smoother probabilities

  vs KNN:
    DT is much faster at prediction time (no distance calc)
    DT is interpretable (KNN is not)
    DT handles categorical features natively

  vs SVM:
    DT is fully explainable (SVM is a black box)
    DT trains and predicts faster
    SVM often achieves higher accuracy on complex boundaries

  vs Random Forest:
    DT: single tree = interpretable rules
    RF: 100+ trees = better accuracy but less interpretable
    RF is the natural upgrade when accuracy > interpretability

DECISION TREE ADVANTAGES
{'='*80}
  - Fully transparent: every prediction is traceable to a rule
  - No preprocessing: no scaling, handles mixed feature types
  - Fast training and prediction
  - Can be printed as a flowchart for advisors
  - Handles both classification and regression
  - Feature importance is intuitive

DECISION TREE LIMITATIONS
{'='*80}
  - Prone to overfitting (needs pruning via max_depth)
  - Unstable: small data changes can flip entire tree structure
  - Greedy splits: locally optimal but not globally
  - Biased toward features with more values/levels
  - Weaker than ensemble methods (Random Forest, Gradient Boosting)

BUSINESS RECOMMENDATIONS
{'='*80}
  1. Run model at start of each semester on all enrolled students
  2. HIGH RISK students: assign a personal academic advisor immediately
  3. AT RISK students: proactive email + optional counseling session
  4. Track GPA Trend weekly: declining trend is the earliest warning
  5. Target first-gen students with peer mentoring programs
  6. Review students dropping credit hours mid-semester
  7. Upgrade to Random Forest next semester for higher accuracy
     while keeping DT as the explainability layer

FILES GENERATED
{'='*80}
  student_dropout_data.csv
  dt_viz_1_distribution.png          - Class distribution + scatter plots
  dt_viz_2_tree.png                  - Visual tree structure (top 3 levels)
  dt_viz_3_depth_tuning.png          - Depth tuning curves
  dt_viz_4_confusion.png             - Confusion matrix (count + %)
  dt_viz_5_importance.png            - Feature importance chart
  dt_viz_6_boxplots.png              - Feature distributions by risk level
  dt_viz_7_criteria_confidence.png   - Criterion comparison + confidence

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)
with open('dt_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("  Report saved to: dt_analysis_report.txt")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DECISION TREE STUDENT DROPOUT CLASSIFICATION COMPLETE!")
print("=" * 80)
print(f"\n  Summary:")
print(f"    Generated {n_total} student records (LOW / AT / HIGH risk)")
print(f"    Best max_depth:  {best_depth}   (found via CV sweep)")
print(f"    Best criterion:  {best_crit}")
print(f"    Train Accuracy:  {train_acc:.4f}")
print(f"    Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"    Weighted F1:     {test_f1:.4f}")
print(f"    Tree Leaves:     {dt_best.get_n_leaves()}")
print(f"    CV Accuracy:     {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"    7 visualizations generated")

print(f"\n  Key Findings:")
print(f"    - Top dropout predictor: {importance_df.iloc[0]['Feature']} "
      f"({importance_df.iloc[0]['Importance']*100:.1f}% importance)")
print(f"    - 2nd predictor:         {importance_df.iloc[1]['Feature']} "
      f"({importance_df.iloc[1]['Importance']*100:.1f}% importance)")
print(f"    - 3rd predictor:         {importance_df.iloc[2]['Feature']} "
      f"({importance_df.iloc[2]['Importance']*100:.1f}% importance)")
print(f"    - Unlimited tree: Train=1.0 (overfit!) vs Pruned: {train_acc:.4f}")
print(f"    - No scaling needed: DT splits on thresholds, not distances")
print(f"    - Rules extracted and ready for advisor use")

print("\n" + "=" * 80)
print("All analysis complete!")
print("=" * 80)