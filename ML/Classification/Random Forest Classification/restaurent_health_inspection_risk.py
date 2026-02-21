"""
RANDOM FOREST CLASSIFICATION - RESTAURANT HEALTH INSPECTION RISK
=================================================================
Classifying restaurants as PASS / CONDITIONAL PASS / FAIL
based on complaint records, operational data, and historical violations

Perfect Scenario for Random Forest:
- Complex interactions between features (rodent complaints + old permit = very risky)
- Noisy real-world data (complaint counts vary, some false reports)
- Many features with varying importance (RF handles feature selection internally)
- Overfitting concern: single Decision Tree memorizes past violations
- Need stable, reliable predictions for public health decisions
- Ensemble averaging smooths out noise from false/duplicate complaints
- Feature importance reveals which factors drive violations most

Dataset: City Restaurant Inspection Records (Generated)
Features:
- DaysSinceLastInspection  (how long since last official visit)
- PreviousViolationCount   (total violations in past 2 years)
- ComplaintsLast6Months    (public complaints filed)
- CriticalViolationsBefore (dangerous violations in history)
- PermitAge                (years since health permit issued)
- SeatingCapacity          (restaurant size)
- EmployeeCount            (staff size)
- FoodHandlerCertified     (% of staff with food safety cert)
- KitchenInspectionScore   (last internal audit score 0-100)
- DaysSinceRenovation      (kitchen/facility age)
- RoachCockroachComplaints (specific pest complaints)
- TemperatureViolations     (cold/hot holding violations before)
- HighRiskFoodServed        (serves raw fish, meat etc: Yes/No)
- InspectorChangedRecently  (new inspector assigned: Yes/No)
- OwnershipChangedRecently  (new owner in past year: Yes/No)

Target: InspectionResult
  - PASS             (meets all health standards)
  - CONDITIONAL PASS (minor violations, must fix within 30 days)
  - FAIL             (serious violations, closed until resolved)

Why Random Forest for Health Inspections?
- Single Decision Tree overfits to past specific violations
- Ensemble of 100+ trees handles noisy complaint data robustly
- Feature importance tells health dept which risk factors matter most
- Handles class imbalance (FAIL is rare) better than single tree
- Bootstrap sampling ensures rare FAIL cases are well-represented
- More reliable predictions = better allocation of inspector resources

Approach:
1. Generate realistic restaurant inspection data
2. Exploratory Data Analysis
3. Preprocessing (minimal - RF handles most natively)
4. Compare Single Decision Tree vs Random Forest
5. Hyperparameter tuning (n_estimators, max_depth, max_features)
6. Feature importance analysis
7. Out-of-bag (OOB) error estimation
8. Comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
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
print("RANDOM FOREST CLASSIFICATION - RESTAURANT HEALTH INSPECTION RISK")
print("=" * 80)


# ============================================================================
# STEP 1: GENERATE REALISTIC RESTAURANT DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC RESTAURANT INSPECTION DATA")
print("=" * 80)

np.random.seed(42)

n_pass  = 700
n_cond  = 500
n_fail  = 300
n_total = n_pass + n_cond + n_fail

print(f"\nGenerating {n_total} restaurant records...")
print(f"  PASS             : {n_pass}   restaurants")
print(f"  CONDITIONAL PASS : {n_cond}   restaurants")
print(f"  FAIL             : {n_fail}   restaurants")

# --- PASS restaurants ---
# Well-maintained, certified staff, recent inspection, few complaints
passed = {
    'DaysSinceLastInspection':  np.random.normal(180, 60,  n_pass).clip(30,  365),
    'PreviousViolationCount':   np.random.poisson(1.0,     n_pass).clip(0,   8),
    'ComplaintsLast6Months':    np.random.poisson(0.5,     n_pass).clip(0,   5),
    'CriticalViolationsBefore': np.random.poisson(0.2,     n_pass).clip(0,   3),
    'PermitAge':                np.random.normal(4,   2.5, n_pass).clip(0.5, 15),
    'SeatingCapacity':          np.random.normal(65,  30,  n_pass).clip(10,  200),
    'EmployeeCount':            np.random.normal(12,  5,   n_pass).clip(2,   40),
    'FoodHandlerCertified':     np.random.normal(88,  8,   n_pass).clip(60,  100),
    'KitchenInspectionScore':   np.random.normal(87,  6,   n_pass).clip(65,  100),
    'DaysSinceRenovation':      np.random.normal(400, 200, n_pass).clip(30,  1200),
    'RoachCockroachComplaints': np.random.poisson(0.1,     n_pass).clip(0,   2),
    'TemperatureViolations':    np.random.poisson(0.3,     n_pass).clip(0,   3),
    'HighRiskFoodServed':       np.random.choice(['Yes','No'], n_pass, p=[0.40, 0.60]),
    'InspectorChangedRecently': np.random.choice(['Yes','No'], n_pass, p=[0.15, 0.85]),
    'OwnershipChangedRecently': np.random.choice(['Yes','No'], n_pass, p=[0.10, 0.90]),
    'InspectionResult':         ['PASS'] * n_pass
}

# --- CONDITIONAL PASS restaurants ---
# Some issues: moderate complaints, aging facility, incomplete certifications
cond = {
    'DaysSinceLastInspection':  np.random.normal(280, 70,  n_cond).clip(60,  500),
    'PreviousViolationCount':   np.random.poisson(3.5,     n_cond).clip(1,   12),
    'ComplaintsLast6Months':    np.random.poisson(2.5,     n_cond).clip(0,   10),
    'CriticalViolationsBefore': np.random.poisson(1.2,     n_cond).clip(0,   6),
    'PermitAge':                np.random.normal(7,   3,   n_cond).clip(1,   20),
    'SeatingCapacity':          np.random.normal(80,  35,  n_cond).clip(10,  250),
    'EmployeeCount':            np.random.normal(16,  7,   n_cond).clip(2,   50),
    'FoodHandlerCertified':     np.random.normal(68,  12,  n_cond).clip(30,  95),
    'KitchenInspectionScore':   np.random.normal(68,  8,   n_cond).clip(45,  85),
    'DaysSinceRenovation':      np.random.normal(900, 300, n_cond).clip(100, 2500),
    'RoachCockroachComplaints': np.random.poisson(0.8,     n_cond).clip(0,   5),
    'TemperatureViolations':    np.random.poisson(1.5,     n_cond).clip(0,   6),
    'HighRiskFoodServed':       np.random.choice(['Yes','No'], n_cond, p=[0.55, 0.45]),
    'InspectorChangedRecently': np.random.choice(['Yes','No'], n_cond, p=[0.25, 0.75]),
    'OwnershipChangedRecently': np.random.choice(['Yes','No'], n_cond, p=[0.25, 0.75]),
    'InspectionResult':         ['CONDITIONAL PASS'] * n_cond
}

# --- FAIL restaurants ---
# Pest complaints, critical violations, uncertified staff, overdue inspection
failed = {
    'DaysSinceLastInspection':  np.random.normal(420, 100, n_fail).clip(100, 730),
    'PreviousViolationCount':   np.random.poisson(7.0,     n_fail).clip(2,   20),
    'ComplaintsLast6Months':    np.random.poisson(5.5,     n_fail).clip(1,   20),
    'CriticalViolationsBefore': np.random.poisson(3.5,     n_fail).clip(1,   10),
    'PermitAge':                np.random.normal(11,  4,   n_fail).clip(2,   25),
    'SeatingCapacity':          np.random.normal(95,  40,  n_fail).clip(15,  300),
    'EmployeeCount':            np.random.normal(20,  8,   n_fail).clip(3,   60),
    'FoodHandlerCertified':     np.random.normal(45,  15,  n_fail).clip(5,   75),
    'KitchenInspectionScore':   np.random.normal(45,  10,  n_fail).clip(15,  65),
    'DaysSinceRenovation':      np.random.normal(1500,400, n_fail).clip(300, 3650),
    'RoachCockroachComplaints': np.random.poisson(3.0,     n_fail).clip(0,   12),
    'TemperatureViolations':    np.random.poisson(3.5,     n_fail).clip(0,   10),
    'HighRiskFoodServed':       np.random.choice(['Yes','No'], n_fail, p=[0.70, 0.30]),
    'InspectorChangedRecently': np.random.choice(['Yes','No'], n_fail, p=[0.35, 0.65]),
    'OwnershipChangedRecently': np.random.choice(['Yes','No'], n_fail, p=[0.40, 0.60]),
    'InspectionResult':         ['FAIL'] * n_fail
}

df = pd.concat([
    pd.DataFrame(passed),
    pd.DataFrame(cond),
    pd.DataFrame(failed)
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

feature_columns = [
    'DaysSinceLastInspection', 'PreviousViolationCount', 'ComplaintsLast6Months',
    'CriticalViolationsBefore', 'PermitAge', 'SeatingCapacity', 'EmployeeCount',
    'FoodHandlerCertified', 'KitchenInspectionScore', 'DaysSinceRenovation',
    'RoachCockroachComplaints', 'TemperatureViolations', 'HighRiskFoodServed',
    'InspectorChangedRecently', 'OwnershipChangedRecently'
]

print(f"\n  Dataset shape: {df.shape}")
print(f"  Features:      {len(feature_columns)}")

print("\n--- First 10 Restaurant Records ---")
print(df.head(10).to_string(index=False))

df.to_csv('restaurant_inspection_data.csv', index=False, encoding='utf-8')
print(f"\n  Dataset saved to: restaurant_inspection_data.csv")


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

result_order = ['PASS', 'CONDITIONAL PASS', 'FAIL']

print(f"\n--- Class Distribution ---")
class_counts = df['InspectionResult'].value_counts()
for res in result_order:
    count = class_counts.get(res, 0)
    print(f"  {res:<20}: {count} restaurants ({count/n_total*100:.1f}%)")

num_cols = [
    'DaysSinceLastInspection', 'PreviousViolationCount', 'ComplaintsLast6Months',
    'CriticalViolationsBefore', 'FoodHandlerCertified', 'KitchenInspectionScore',
    'RoachCockroachComplaints', 'TemperatureViolations'
]

print(f"\n--- Mean Feature Values by Inspection Result ---")
print(df.groupby('InspectionResult')[num_cols].mean().round(2).to_string())

print(f"\n--- Categorical Breakdown ---")
for cat in ['HighRiskFoodServed', 'OwnershipChangedRecently', 'InspectorChangedRecently']:
    print(f"\n  {cat}:")
    ct = pd.crosstab(df[cat], df['InspectionResult'],
                     normalize='columns').round(3) * 100
    print(ct[result_order])

print(f"\n--- Risk Signal Strength ---")
group = df.groupby('InspectionResult')
print(f"\n  {'Feature':<28} {'PASS':>8} {'COND':>8} {'FAIL':>8}  Signal")
print(f"  {'-'*65}")
for feat in num_cols:
    vals   = [group[feat].mean().get(r, 0) for r in result_order]
    spread = max(vals) - min(vals)
    avg    = np.mean(vals) + 1e-6
    s      = "STRONG" if spread/avg > 0.5 else ("MODERATE" if spread/avg > 0.2 else "WEAK")
    print(f"  {feat:<28} {vals[0]:>8.2f} {vals[1]:>8.2f} {vals[2]:>8.2f}  {s}")


# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

print("""
  Random Forest Preprocessing — Minimal Like Decision Tree:
  ============================================================
  - NO feature scaling needed (threshold-based splits)
  - Only encode categorical variables
  - Random Forest handles:
    * Different feature scales automatically
    * Missing values (with some implementations)
    * Irrelevant features (averaging dilutes their effect)
    * Class imbalance (class_weight parameter)

  Additional RF advantage over single DT:
  - Bootstrap sampling naturally oversamples rare FAIL cases
  - Each tree sees a different random subset = diverse perspectives
  - Averaging 100+ trees cancels out individual tree noise
""")

df_enc = df.copy()
le_risk     = LabelEncoder()
le_insp     = LabelEncoder()
le_own      = LabelEncoder()

df_enc['HighRiskFoodServed']       = le_risk.fit_transform(df['HighRiskFoodServed'])
df_enc['InspectorChangedRecently'] = le_insp.fit_transform(df['InspectorChangedRecently'])
df_enc['OwnershipChangedRecently'] = le_own.fit_transform(df['OwnershipChangedRecently'])

print(f"  Encoded: HighRiskFoodServed        "
      f"{dict(zip(le_risk.classes_, le_risk.transform(le_risk.classes_)))}")
print(f"  Encoded: InspectorChangedRecently  "
      f"{dict(zip(le_insp.classes_, le_insp.transform(le_insp.classes_)))}")
print(f"  Encoded: OwnershipChangedRecently  "
      f"{dict(zip(le_own.classes_, le_own.transform(le_own.classes_)))}")

X = df_enc[feature_columns]
y = df_enc['InspectionResult']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train: {X_train.shape[0]} restaurants | Test: {X_test.shape[0]} restaurants")
print(f"  Stratified class counts in test set:")
for res in result_order:
    count = (y_test == res).sum()
    print(f"    {res:<20}: {count}")


# ============================================================================
# STEP 4: HOW RANDOM FOREST WORKS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HOW RANDOM FOREST WORKS")
print("=" * 80)

print(f"""
  Random Forest = Ensemble of Decision Trees
  ============================================================

  TRAINING (for each of N trees):
    1. Bootstrap sample: randomly draw n samples WITH replacement
       (some restaurants appear multiple times, others not at all)
    2. At each split: consider only sqrt(n_features) random features
       (not all 15 features — random subset only)
    3. Grow the tree to full depth (or max_depth limit)
    4. Repeat for all N trees

  PREDICTION (for a new restaurant):
    1. Pass features through ALL N trees
    2. Each tree casts a vote: PASS / CONDITIONAL PASS / FAIL
    3. Majority vote wins

  EXAMPLE (N=5 trees):
    Restaurant with: 8 complaints, 4 critical violations, 420 days since inspection
    Tree 1: FAIL          Tree 4: FAIL
    Tree 2: CONDITIONAL   Tree 5: FAIL
    Tree 3: FAIL
    Vote: FAIL=4, COND=1, PASS=0  ->  Prediction: FAIL

  WHY BETTER THAN SINGLE DECISION TREE:
    Single DT: memorizes training data -> overfits
    Random Forest: 100 different DTs on different data subsets
                   -> variance cancelled by averaging
                   -> much better generalization

  OUT-OF-BAG (OOB) ESTIMATION:
    Each tree's bootstrap leaves out ~37% of data (out-of-bag samples)
    These OOB samples act as a built-in validation set
    OOB error = free estimate of generalization error, no CV needed!

  KEY PARAMETERS:
    n_estimators : number of trees (more = better, diminishing returns)
    max_depth    : depth per tree (None = full depth is fine in RF)
    max_features : features per split (sqrt = default, works well)
    class_weight : handle class imbalance (FAIL is rare)
""")


# ============================================================================
# STEP 5: SINGLE DECISION TREE vs RANDOM FOREST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SINGLE DECISION TREE vs RANDOM FOREST")
print("=" * 80)

print(f"\n  Training Single Decision Tree (max_depth=10)...")
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
y_dt_train = dt.predict(X_train)
y_dt_test  = dt.predict(X_test)
dt_train_acc = accuracy_score(y_train, y_dt_train)
dt_test_acc  = accuracy_score(y_test,  y_dt_test)
dt_f1        = f1_score(y_test, y_dt_test, average='weighted')
dt_cv        = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy').mean()

print(f"  Decision Tree  -> Train: {dt_train_acc:.4f} | Test: {dt_test_acc:.4f} | "
      f"F1: {dt_f1:.4f} | CV: {dt_cv:.4f} | Gap: {dt_train_acc-dt_test_acc:.4f}")

print(f"\n  Training Random Forest (100 trees, default params)...")
rf_default = RandomForestClassifier(n_estimators=100, random_state=42,
                                     n_jobs=-1, oob_score=True)
rf_default.fit(X_train, y_train)
y_rf_train = rf_default.predict(X_train)
y_rf_test  = rf_default.predict(X_test)
rf_train_acc = accuracy_score(y_train, y_rf_train)
rf_test_acc  = accuracy_score(y_test,  y_rf_test)
rf_f1        = f1_score(y_test, y_rf_test, average='weighted')
rf_cv        = cross_val_score(rf_default, X_train, y_train, cv=5, scoring='accuracy').mean()
rf_oob       = rf_default.oob_score_

print(f"  Random Forest  -> Train: {rf_train_acc:.4f} | Test: {rf_test_acc:.4f} | "
      f"F1: {rf_f1:.4f} | CV: {rf_cv:.4f} | Gap: {rf_train_acc-rf_test_acc:.4f}")
print(f"  OOB Score      -> {rf_oob:.4f}  (free accuracy estimate, no CV needed!)")

print(f"\n  Improvement Summary:")
print(f"    Accuracy gain:     {rf_test_acc - dt_test_acc:+.4f}")
print(f"    Overfitting reduction: {(dt_train_acc-dt_test_acc) - (rf_train_acc-rf_test_acc):+.4f}")
print(f"    F1 gain:           {rf_f1 - dt_f1:+.4f}")


# ============================================================================
# STEP 6: N_ESTIMATORS SWEEP (HOW MANY TREES?)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: N_ESTIMATORS SWEEP — HOW MANY TREES ARE ENOUGH?")
print("=" * 80)

print(f"\n  Testing tree counts: 1 to 300")
tree_counts  = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
estimator_results = []

print(f"\n  {'Trees':>6} {'Train Acc':>10} {'Test Acc':>10} {'OOB Score':>10}")
print(f"  {'-'*42}")

for n in tree_counts:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=42,
                                   n_jobs=-1, oob_score=True)
    rf_n.fit(X_train, y_train)
    tr = rf_n.score(X_train, y_train)
    te = rf_n.score(X_test,  y_test)
    ob = rf_n.oob_score_
    estimator_results.append({'Trees': n, 'Train': tr, 'Test': te, 'OOB': ob})
    print(f"  {n:>6} {tr:>10.4f} {te:>10.4f} {ob:>10.4f}")

est_df    = pd.DataFrame(estimator_results)
best_n    = est_df.loc[est_df['Test'].idxmax(), 'Trees']
print(f"\n  Best n_estimators = {best_n}  (Test Acc = {est_df['Test'].max():.4f})")
print(f"  OOB stabilizes around 50-100 trees — adding more gives diminishing returns")


# ============================================================================
# STEP 7: HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: HYPERPARAMETER TUNING")
print("=" * 80)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [10, 20, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 3, 5],
}

total = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
         len(param_grid['max_features']) * len(param_grid['min_samples_leaf']))

print(f"\n  Grid: {param_grid}")
print(f"  Total combinations: {total}  |  With 3-fold CV: {total*3} fits")
print(f"  Running grid search (this takes a moment)...")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=False),
    param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"\n  Best Parameters:")
for k, v in grid_search.best_params_.items():
    print(f"    {k}: {v}")
print(f"  Best CV F1: {grid_search.best_score_:.4f}")


# ============================================================================
# STEP 8: FINAL OPTIMIZED RANDOM FOREST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: FINAL OPTIMIZED RANDOM FOREST")
print("=" * 80)

best_rf    = grid_search.best_estimator_
y_pred     = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)

train_acc  = best_rf.score(X_train, y_train)
test_acc   = accuracy_score(y_test, y_pred)
test_f1    = f1_score(y_test, y_pred, average='weighted')

print(f"\n  Optimized RF: {grid_search.best_params_}")
print(f"  Train Accuracy:  {train_acc:.4f}")
print(f"  Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Weighted F1:     {test_f1:.4f}")
print(f"  Overfitting Gap: {train_acc - test_acc:.4f}")

print(f"\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=result_order))

cm = confusion_matrix(y_test, y_pred, labels=result_order)
print(f"\n--- Confusion Matrix ---")
header = f"  {'':22} {'PASS':>8} {'COND':>8} {'FAIL':>8}"
print(header)
for i, label in enumerate(result_order):
    short = label[:22]
    print(f"  {short:<22} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")


# ============================================================================
# STEP 9: OUT-OF-BAG SCORE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: OUT-OF-BAG (OOB) ERROR ANALYSIS")
print("=" * 80)

rf_oob_final = RandomForestClassifier(
    **grid_search.best_params_,
    random_state=42, n_jobs=-1, oob_score=True
)
rf_oob_final.fit(X_train, y_train)

print(f"""
  OOB Score Explained:
  ============================================================
  During bootstrap sampling, each tree leaves out ~37% of data.
  These left-out samples are the "out-of-bag" set for that tree.
  Predicting on OOB samples gives a free internal validation score.

  OOB Score:     {rf_oob_final.oob_score_:.4f}
  Test Accuracy: {test_acc:.4f}
  Difference:    {abs(rf_oob_final.oob_score_ - test_acc):.4f}
  Agreement:     {'Excellent' if abs(rf_oob_final.oob_score_ - test_acc) < 0.02 else 'Good'}

  OOB is especially useful when data is limited —
  no need to sacrifice a test set!
""")


# ============================================================================
# STEP 10: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print(f"""
  RF Feature Importance vs Single DT:
  ============================================================
  Single DT: Importance = Gini reduction in THAT tree's splits
             Unstable: different random seed = different ranking

  Random Forest: Importance = AVERAGE Gini reduction across ALL trees
                 Much more reliable and stable
                 Unimportant features consistently score near zero
""")

importance_df = pd.DataFrame({
    'Feature':    feature_columns,
    'RF_Importance': best_rf.feature_importances_,
    'DT_Importance': dt.feature_importances_
}).sort_values('RF_Importance', ascending=False).reset_index(drop=True)

print(f"  {'Rank':<5} {'Feature':<28} {'RF Imp':>9} {'DT Imp':>9}  {'Bar (RF)'}")
print(f"  {'-'*75}")
for i, row in importance_df.iterrows():
    bar = '#' * int(row['RF_Importance'] * 60)
    print(f"  {i+1:<5} {row['Feature']:<28} {row['RF_Importance']:>9.4f} "
          f"{row['DT_Importance']:>9.4f}  {bar}")

print(f"\n  Top 5 Risk Factors (RF):")
for _, row in importance_df.head(5).iterrows():
    print(f"    - {row['Feature']}: {row['RF_Importance']*100:.2f}%")


# ============================================================================
# STEP 11: CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: CROSS-VALIDATION")
print("=" * 80)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(best_rf, X_train, y_train, cv=skf, scoring='accuracy')
cv_f1  = cross_val_score(best_rf, X_train, y_train, cv=skf, scoring='f1_weighted')

print(f"\n  5-Fold Stratified Cross-Validation:")
print(f"  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}  | Folds: {cv_acc.round(4)}")
print(f"  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}  | Folds: {cv_f1.round(4)}")
print(f"  Stability: {'Excellent' if cv_acc.std() < 0.02 else 'Good' if cv_acc.std() < 0.05 else 'Variable'}")


# ============================================================================
# STEP 12: SAMPLE PREDICTIONS WITH RISK PRIORITY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: SAMPLE PREDICTIONS WITH INSPECTOR PRIORITY")
print("=" * 80)

classes = best_rf.classes_
print(f"\n  Inspector dispatch priority:")
print(f"  FAIL             -> Dispatch TODAY (close until resolved)")
print(f"  CONDITIONAL PASS -> Schedule within 2 weeks")
print(f"  PASS             -> Routine annual inspection")

print(f"\n  {'Restaurant':<13} {'Actual':<20} {'Predicted':<20} {'Conf%':>6} {'Priority':>15} {'OK?':>8}")
print(f"  {'-'*85}")
for i in range(20):
    actual    = y_test.iloc[i]
    predicted = y_pred[i]
    conf      = y_pred_prob[i].max() * 100
    priority  = "URGENT" if predicted == 'FAIL' else \
                ("SOON" if predicted == 'CONDITIONAL PASS' else "ROUTINE")
    correct   = "Correct" if predicted == actual else "WRONG"
    print(f"  Restaurant {i+1:<3} {actual:<20} {predicted:<20} {conf:>5.1f}%"
          f" {priority:>15} {correct:>8}")


# ============================================================================
# STEP 13: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 13: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

res_colors = {'PASS': '#1B5E20', 'CONDITIONAL PASS': '#E65100', 'FAIL': '#B71C1C'}

# --- Viz 1: Distribution + Key Scatter ---
print("\n  Creating distribution and scatter plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

counts = [class_counts.get(r, 0) for r in result_order]
colors = [res_colors[r] for r in result_order]
bars   = axes[0].bar(['PASS', 'COND PASS', 'FAIL'], counts,
                      color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Inspection Result Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Restaurants', fontsize=11, fontweight='bold')
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{count}\n({count/n_total*100:.0f}%)',
                 ha='center', fontsize=11, fontweight='bold')

for res in result_order:
    sub = df[df['InspectionResult'] == res]
    axes[1].scatter(sub['KitchenInspectionScore'], sub['PreviousViolationCount'],
                    c=res_colors[res], label=res, s=15, alpha=0.4, edgecolors='none')
axes[1].set_xlabel('Kitchen Inspection Score', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Previous Violation Count', fontsize=11, fontweight='bold')
axes[1].set_title('Kitchen Score vs Prior Violations', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

for res in result_order:
    sub = df[df['InspectionResult'] == res]
    axes[2].scatter(sub['FoodHandlerCertified'], sub['ComplaintsLast6Months'],
                    c=res_colors[res], label=res, s=15, alpha=0.4, edgecolors='none')
axes[2].set_xlabel('Food Handler Certified (%)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Complaints Last 6 Months', fontsize=11, fontweight='bold')
axes[2].set_title('Certification vs Complaints', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_cls_viz_1_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: rf_cls_viz_1_distribution.png")

# --- Viz 2: DT vs RF Comparison ---
print("  Creating DT vs RF comparison...")
models     = ['Decision Tree', 'RF Default', 'RF Optimized']
train_accs = [dt_train_acc, rf_train_acc, train_acc]
test_accs  = [dt_test_acc,  rf_test_acc,  test_acc]
f1_scores  = [dt_f1,        rf_f1,        test_f1]
gaps       = [dt_train_acc - dt_test_acc,
              rf_train_acc - rf_test_acc,
              train_acc    - test_acc]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x_pos = np.arange(len(models))
w = 0.28
axes[0].bar(x_pos - w, train_accs, w, label='Train Acc', color='#42A5F5',
            edgecolor='black', alpha=0.85)
axes[0].bar(x_pos,     test_accs,  w, label='Test Acc',  color='#66BB6A',
            edgecolor='black', alpha=0.85)
axes[0].bar(x_pos + w, f1_scores,  w, label='F1-Score',  color='#FFA726',
            edgecolor='black', alpha=0.85)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(models, fontsize=10)
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('Decision Tree vs Random Forest', fontsize=13, fontweight='bold')
axes[0].set_ylim(0.7, 1.08)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

gap_colors = ['#E53935' if g > 0.1 else '#FB8C00' if g > 0.05 else '#43A047'
              for g in gaps]
axes[1].bar(models, gaps, color=gap_colors, edgecolor='black', alpha=0.85)
axes[1].axhline(y=0.05, color='orange', linestyle='--', lw=2, label='Mild overfit')
axes[1].axhline(y=0.10, color='red',    linestyle='--', lw=2, label='High overfit')
axes[1].set_ylabel('Train Acc - Test Acc', fontsize=12, fontweight='bold')
axes[1].set_title('Overfitting Comparison', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('rf_cls_viz_2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: rf_cls_viz_2_comparison.png")

# --- Viz 3: N_Estimators Curve ---
print("  Creating n_estimators sweep plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(est_df['Trees'], est_df['Train'], 'b-o', markersize=6,
             lw=2, label='Train Accuracy')
axes[0].plot(est_df['Trees'], est_df['Test'],  'g-s', markersize=6,
             lw=2, label='Test Accuracy')
axes[0].plot(est_df['Trees'], est_df['OOB'],   'r-^', markersize=6,
             lw=2, label='OOB Score')
axes[0].axvline(x=best_n, color='orange', linestyle='--', lw=2.5,
                label=f'Best n={best_n}')
axes[0].set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Accuracy vs Number of Trees', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(est_df['Trees'], est_df['Train'] - est_df['Test'],
             'r-o', markersize=6, lw=2)
axes[1].axhline(y=0.05, color='orange', linestyle='--', lw=2, label='Mild overfit')
axes[1].axhline(y=0.10, color='red',    linestyle='--', lw=2, label='High overfit')
axes[1].set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Train - Test Gap', fontsize=12, fontweight='bold')
axes[1].set_title('Overfitting Gap vs Number of Trees', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_cls_viz_3_estimators.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: rf_cls_viz_3_estimators.png")

# --- Viz 4: Confusion Matrix ---
print("  Creating confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

labels_short = ['PASS', 'COND', 'FAIL']
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],
            xticklabels=labels_short, yticklabels=labels_short,
            annot_kws={"size": 14, "weight": "bold"})
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual',    fontsize=12, fontweight='bold')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')

cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=labels_short, yticklabels=labels_short,
            annot_kws={"size": 14, "weight": "bold"})
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual',    fontsize=12, fontweight='bold')
axes[1].set_title('Confusion Matrix (% of Actual)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('rf_cls_viz_4_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: rf_cls_viz_4_confusion.png")

# --- Viz 5: Feature Importance RF vs DT ---
print("  Creating feature importance comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

imp_colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(importance_df)))[::-1]
axes[0].barh(importance_df['Feature'][::-1],
             importance_df['RF_Importance'][::-1],
             color=imp_colors, edgecolor='black', alpha=0.85)
axes[0].set_xlabel('Importance (Avg Gini Reduction across all trees)',
                   fontsize=11, fontweight='bold')
axes[0].set_title('Random Forest Feature Importance\n(Stable — averaged across 100+ trees)',
                  fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
for i, (val, feat) in enumerate(zip(importance_df['RF_Importance'][::-1],
                                     importance_df['Feature'][::-1])):
    if val > 0.01:
        axes[0].text(val + 0.003, i, f'{val:.3f}', va='center',
                     fontsize=8, fontweight='bold')

plot_df = importance_df.sort_values('DT_Importance', ascending=False)
y_pos   = np.arange(len(plot_df))
axes[1].barh(y_pos - 0.2, plot_df['DT_Importance'], 0.4,
             label='Decision Tree', color='#EF5350', edgecolor='black', alpha=0.8)
axes[1].barh(y_pos + 0.2, plot_df['RF_Importance'], 0.4,
             label='Random Forest', color='#42A5F5', edgecolor='black', alpha=0.8)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(plot_df['Feature'], fontsize=9)
axes[1].set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
axes[1].set_title('DT vs RF Feature Importance\n(RF more stable — DT can be erratic)',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('rf_cls_viz_5_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: rf_cls_viz_5_importance.png")

# --- Viz 6: Feature Boxplots ---
print("  Creating feature boxplots...")
key_feats = ['KitchenInspectionScore', 'FoodHandlerCertified',
             'PreviousViolationCount', 'ComplaintsLast6Months',
             'RoachCockroachComplaints', 'DaysSinceLastInspection']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(key_feats):
    data_by_res = [df[df['InspectionResult'] == r][feat].values for r in result_order]
    bp = axes[i].boxplot(data_by_res, patch_artist=True,
                          medianprops=dict(color='black', linewidth=2.5))
    for patch, res in zip(bp['boxes'], result_order):
        patch.set_facecolor(res_colors[res])
        patch.set_alpha(0.8)
    axes[i].set_xticklabels(['PASS', 'COND', 'FAIL'], fontsize=10)
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Value', fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Feature Distributions by Inspection Result', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_cls_viz_6_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: rf_cls_viz_6_boxplots.png")

# --- Viz 7: Prediction Confidence ---
print("  Creating confidence and OOB plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

max_probs = y_pred_prob.max(axis=1)
correct   = (y_pred == np.array(y_test))
axes[0].hist(max_probs[correct],  bins=25, alpha=0.7, color='#2E7D32',
             label='Correct', edgecolor='black')
axes[0].hist(max_probs[~correct], bins=25, alpha=0.7, color='#C62828',
             label='Wrong',   edgecolor='black')
axes[0].set_xlabel('Prediction Confidence (max vote share)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Restaurants', fontsize=12, fontweight='bold')
axes[0].set_title('RF Confidence: Correct vs Wrong', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# OOB score across trees
oob_scores_n = []
for n in range(10, 210, 10):
    rf_tmp = RandomForestClassifier(n_estimators=n, random_state=42,
                                    n_jobs=-1, oob_score=True)
    rf_tmp.fit(X_train, y_train)
    oob_scores_n.append(rf_tmp.oob_score_)

axes[1].plot(range(10, 210, 10), oob_scores_n, 'b-o', markersize=5, lw=2)
axes[1].axhline(y=test_acc, color='green', linestyle='--', lw=2,
                label=f'Test Acc={test_acc:.4f}')
axes[1].set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
axes[1].set_ylabel('OOB Score', fontsize=12, fontweight='bold')
axes[1].set_title('OOB Score Convergence\n(No test set needed!)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_cls_viz_7_confidence_oob.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: rf_cls_viz_7_confidence_oob.png")


# ============================================================================
# STEP 14: COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 14: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'='*80}
RANDOM FOREST CLASSIFICATION - RESTAURANT HEALTH INSPECTION RISK
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
Classify restaurants as PASS / CONDITIONAL PASS / FAIL before dispatching
health inspectors, enabling the city health department to prioritize
urgent visits, optimize inspector workload, and protect public health.

WHY RANDOM FOREST FOR HEALTH INSPECTIONS?
  - Noisy complaint data (false reports, duplicates) — RF averages noise away
  - Complex interactions: old permit + pest complaints = very high risk
  - Rare FAIL cases handled better by ensemble bootstrap sampling
  - Feature importance reveals which factors truly drive violations
  - More stable predictions than single DT for public health decisions

DATASET SUMMARY
{'='*80}
  Total restaurants:  {n_total}
  Features:           {len(feature_columns)}
  PASS:               {n_pass} ({n_pass/n_total*100:.0f}%)
  CONDITIONAL PASS:   {n_cond} ({n_cond/n_total*100:.0f}%)
  FAIL:               {n_fail} ({n_fail/n_total*100:.0f}%)
  Train/Test:         80% / 20% stratified

DT vs RF COMPARISON
{'='*80}
  {'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'F1':>8} {'CV':>8} {'Gap':>8}
  {'-'*68}
  {'Decision Tree':<22} {dt_train_acc:>10.4f} {dt_test_acc:>10.4f} {dt_f1:>8.4f} {dt_cv:>8.4f} {dt_train_acc-dt_test_acc:>8.4f}
  {'RF Default':<22} {rf_train_acc:>10.4f} {rf_test_acc:>10.4f} {rf_f1:>8.4f} {rf_cv:>8.4f} {rf_train_acc-rf_test_acc:>8.4f}
  {'RF Optimized':<22} {train_acc:>10.4f} {test_acc:>10.4f} {test_f1:>8.4f} {cv_acc.mean():>8.4f} {train_acc-test_acc:>8.4f}

  Accuracy improvement over DT:  {test_acc - dt_test_acc:+.4f}
  Overfitting reduction:         {(dt_train_acc-dt_test_acc)-(train_acc-test_acc):+.4f}

BEST MODEL: RANDOM FOREST OPTIMIZED
{'='*80}
  Parameters: {grid_search.best_params_}
  OOB Score:       {rf_oob_final.oob_score_:.4f}
  Test Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)
  Weighted F1:     {test_f1:.4f}
  Overfitting Gap: {train_acc-test_acc:.4f}

  Classification Report:
{classification_report(y_test, y_pred, target_names=result_order)}

  Confusion Matrix:
  {'':22} {'PASS':>8} {'COND':>8} {'FAIL':>8}
  PASS          {cm[0,0]:>8} {cm[0,1]:>8} {cm[0,2]:>8}
  COND PASS     {cm[1,0]:>8} {cm[1,1]:>8} {cm[1,2]:>8}
  FAIL          {cm[2,0]:>8} {cm[2,1]:>8} {cm[2,2]:>8}

N_ESTIMATORS SWEEP
{'='*80}
  Best n_estimators: {best_n}
  OOB and test accuracy stabilize well before 100 trees
  Adding trees beyond 150 gives minimal improvement

CROSS-VALIDATION (5-Fold)
{'='*80}
  Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}
  F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}
  Stability: {'Excellent' if cv_acc.std() < 0.02 else 'Good'}

FEATURE IMPORTANCE (RF — Averaged across all trees)
{'='*80}
{chr(10).join([f"  {i+1:>2}. {row['Feature']:<30} RF: {row['RF_Importance']:.4f} ({row['RF_Importance']*100:.2f}%)  DT: {row['DT_Importance']:.4f}"
               for i, (_, row) in enumerate(importance_df.iterrows())])}

INSPECTOR DISPATCH PRIORITY SYSTEM
{'='*80}
  FAIL predicted:             Dispatch TODAY — close until resolved
  CONDITIONAL PASS predicted: Schedule inspection within 2 weeks
  PASS predicted:             Routine annual scheduling

  Using this model, the health dept can:
  - Prioritize the {(y_pred=='FAIL').sum()} restaurants predicted FAIL in test set
  - Pre-allocate inspector time before complaints escalate
  - Reduce unnecessary visits to low-risk establishments

RF vs DT KEY DIFFERENCES IN THIS SCENARIO
{'='*80}
  Decision Tree:
    - One tree memorizes specific past violations
    - Sensitive to which complaints happened to be in training data
    - Feature importance unstable across random seeds
    - Overfits on rare FAIL cases

  Random Forest:
    - 100+ trees each trained on different bootstrapped data
    - Complaint noise averaged out across trees
    - Feature importance stable and reliable
    - FAIL cases appear in multiple bootstrap samples
    - OOB score = free validation, no data sacrifice needed

BUSINESS RECOMMENDATIONS
{'='*80}
  1. Run model weekly against complaint database updates
  2. Auto-dispatch inspectors when FAIL probability > 70%
  3. Focus on top risk factors: {importance_df.iloc[0]['Feature']}, {importance_df.iloc[1]['Feature']}
  4. Track restaurants with rising complaint counts month-over-month
  5. Flag ownership changes immediately for re-inspection scheduling
  6. Use CONDITIONAL PASS predictions to plan proactive advisory visits
  7. Retrain model quarterly with new inspection outcomes

FILES GENERATED
{'='*80}
  restaurant_inspection_data.csv
  rf_cls_viz_1_distribution.png     - Class distribution + scatter plots
  rf_cls_viz_2_comparison.png       - DT vs RF comparison
  rf_cls_viz_3_estimators.png       - n_estimators sweep + OOB
  rf_cls_viz_4_confusion.png        - Confusion matrix (count + %)
  rf_cls_viz_5_importance.png       - Feature importance DT vs RF
  rf_cls_viz_6_boxplots.png         - Feature distributions by result
  rf_cls_viz_7_confidence_oob.png   - Confidence + OOB convergence

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)
with open('rf_cls_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("  Report saved to: rf_cls_report.txt")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("RANDOM FOREST RESTAURANT INSPECTION CLASSIFICATION COMPLETE!")
print("=" * 80)
print(f"\n  Summary:")
print(f"    Generated {n_total} restaurant records (PASS / COND / FAIL)")
print(f"    Best params: {grid_search.best_params_}")
print(f"    Train Accuracy:  {train_acc:.4f}")
print(f"    Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"    Weighted F1:     {test_f1:.4f}")
print(f"    OOB Score:       {rf_oob_final.oob_score_:.4f}")
print(f"    CV Accuracy:     {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"    DT Test Acc:     {dt_test_acc:.4f} (RF improved by {test_acc-dt_test_acc:+.4f})")
print(f"    7 visualizations generated")
print(f"\n  Key Findings:")
print(f"    - Top risk factor: {importance_df.iloc[0]['Feature']} "
      f"({importance_df.iloc[0]['RF_Importance']*100:.1f}%)")
print(f"    - 2nd risk factor: {importance_df.iloc[1]['Feature']} "
      f"({importance_df.iloc[1]['RF_Importance']*100:.1f}%)")
print(f"    - 3rd risk factor: {importance_df.iloc[2]['Feature']} "
      f"({importance_df.iloc[2]['RF_Importance']*100:.1f}%)")
print(f"    - RF reduced overfitting from {dt_train_acc-dt_test_acc:.4f} "
      f"to {train_acc-test_acc:.4f}")
print(f"    - OOB score {rf_oob_final.oob_score_:.4f} closely tracks "
      f"test acc {test_acc:.4f}")
print(f"    - RF importance stable; DT importance erratic across seeds")

print("\n" + "=" * 80)
print("All analysis complete!")
print("=" * 80)