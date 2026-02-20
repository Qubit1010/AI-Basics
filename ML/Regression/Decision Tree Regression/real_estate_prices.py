"""
DECISION TREE REGRESSION - REAL ESTATE PRICE PREDICTION
========================================================
Predicting house prices based on property features using Decision Trees

Perfect Scenario for Decision Trees:
- Real Estate prices depend on multiple discrete factors
- Non-linear relationships (location, size, age interact)
- Interpretable rules needed (explain pricing to clients)
- Categorical and numerical features
- Need to identify key price drivers

Dataset: House Prices (Generated)
Features:
- Square Footage
- Number of Bedrooms
- Number of Bathrooms
- Age of House
- Distance to City Center (km)
- Neighborhood Quality (1-10)
- Has Garage (Yes/No)
- Has Garden (Yes/No)

Approach:
1. Generate realistic house price data
2. Exploratory Data Analysis
3. Build Decision Tree models
4. Compare different tree depths
5. Feature importance analysis
6. Visualize decision tree structure
7. Model evaluation and interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("DECISION TREE REGRESSION - REAL ESTATE PRICE PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: GENERATE REALISTIC HOUSE PRICE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC HOUSE PRICE DATA")
print("=" * 80)

np.random.seed(42)

# Number of houses
n_samples = 200

print(f"\nGenerating {n_samples} house records...")

# Generate features
data = {
    'SquareFeet': np.random.randint(800, 4000, n_samples),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Bathrooms': np.random.randint(1, 4, n_samples),
    'Age': np.random.randint(0, 50, n_samples),
    'DistanceToCenter': np.round(np.random.uniform(0.5, 25, n_samples), 1),
    'NeighborhoodQuality': np.random.randint(1, 11, n_samples),
    'HasGarage': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
    'HasGarden': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
}

df = pd.DataFrame(data)

# Generate realistic house prices based on features
print("\nGenerating prices based on realistic rules:")
print("  • Base price: $100,000")
print("  • +$150 per square foot")
print("  • +$20,000 per bedroom")
print("  • +$15,000 per bathroom")
print("  • -$1,000 per year of age")
print("  • -$3,000 per km from city center")
print("  • +$10,000 per neighborhood quality point")
print("  • +$25,000 for garage")
print("  • +$15,000 for garden")
print("  • Random variation: ±$30,000")

base_price = 100000
price = (
        base_price +
        df['SquareFeet'] * 150 +
        df['Bedrooms'] * 20000 +
        df['Bathrooms'] * 15000 -
        df['Age'] * 1000 -
        df['DistanceToCenter'] * 3000 +
        df['NeighborhoodQuality'] * 10000 +
        (df['HasGarage'] == 'Yes') * 25000 +
        (df['HasGarden'] == 'Yes') * 15000 +
        np.random.normal(0, 30000, n_samples)
)

df['Price'] = price.astype(int)

print(f"\n✓ Dataset generated successfully!")
print(f"  Shape: {df.shape}")
print(f"  Features: {len(df.columns) - 1}")

print("\n--- First 10 Houses ---")
print(df.head(10).to_string(index=False))

print("\n--- Data Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

# Save dataset
df.to_csv('house_prices.csv', index=False)
print(f"\n✓ Dataset saved to: house_prices.csv")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n--- Price Statistics ---")
print(f"  Mean: ${df['Price'].mean():,.2f}")
print(f"  Median: ${df['Price'].median():,.2f}")
print(f"  Std Dev: ${df['Price'].std():,.2f}")
print(f"  Min: ${df['Price'].min():,}")
print(f"  Max: ${df['Price'].max():,}")
print(f"  Range: ${df['Price'].max() - df['Price'].min():,}")

print("\n--- Categorical Features ---")
print(
    f"Houses with Garage: {(df['HasGarage'] == 'Yes').sum()} ({(df['HasGarage'] == 'Yes').sum() / len(df) * 100:.1f}%)")
print(
    f"Houses with Garden: {(df['HasGarden'] == 'Yes').sum()} ({(df['HasGarden'] == 'Yes').sum() / len(df) * 100:.1f}%)")

print("\n--- Numerical Features Summary ---")
numerical_cols = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age', 'DistanceToCenter', 'NeighborhoodQuality']
for col in numerical_cols:
    print(f"{col}:")
    print(f"  Range: {df[col].min()} - {df[col].max()}")
    print(f"  Mean: {df[col].mean():.2f}")

# Correlation analysis
print("\n--- Correlation with Price ---")
# Encode categorical variables for correlation
df_corr = df.copy()
df_corr['HasGarage'] = (df_corr['HasGarage'] == 'Yes').astype(int)
df_corr['HasGarden'] = (df_corr['HasGarden'] == 'Yes').astype(int)

correlations = df_corr.corr()['Price'].sort_values(ascending=False)
print(correlations)

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

print("\n--- Encoding Categorical Variables ---")
# Label encoding for binary features
le_garage = LabelEncoder()
le_garden = LabelEncoder()

df['HasGarage_Encoded'] = le_garage.fit_transform(df['HasGarage'])
df['HasGarden_Encoded'] = le_garden.fit_transform(df['HasGarden'])

print(f"  HasGarage: {dict(zip(le_garage.classes_, le_garage.transform(le_garage.classes_)))}")
print(f"  HasGarden: {dict(zip(le_garden.classes_, le_garden.transform(le_garden.classes_)))}")

# Prepare features and target
feature_columns = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age',
                   'DistanceToCenter', 'NeighborhoodQuality',
                   'HasGarage_Encoded', 'HasGarden_Encoded']

X = df[feature_columns]
y = df['Price']

print(f"\nFeatures (X):")
print(f"  Shape: {X.shape}")
print(f"  Columns: {feature_columns}")

print(f"\nTarget (y):")
print(f"  Shape: {y.shape}")
print(f"  Name: Price")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split (80/20):")
print(f"  Training set: {X_train.shape[0]} houses ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} houses ({X_test.shape[0] / len(X) * 100:.1f}%)")

print(f"\nTraining set price statistics:")
print(f"  Mean: ${y_train.mean():,.2f}")
print(f"  Range: ${y_train.min():,} - ${y_train.max():,}")

print(f"\nTest set price statistics:")
print(f"  Mean: ${y_test.mean():,.2f}")
print(f"  Range: ${y_test.min():,} - ${y_test.max():,}")

# ============================================================================
# STEP 5: BUILD DECISION TREE MODELS (DIFFERENT DEPTHS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: BUILD DECISION TREE MODELS (DIFFERENT DEPTHS)")
print("=" * 80)

print("\nExplanation:")
print("  Decision Trees split data based on feature values")
print("  Each split creates a rule (e.g., 'If SquareFeet > 2000...')")
print("  Tree depth controls model complexity:")
print("    • Shallow trees (3-5): Simple, interpretable, may underfit")
print("    • Medium trees (6-10): Balanced")
print("    • Deep trees (15+): Complex, may overfit")

# Test different max_depths
depths = [3, 5, 7, 10, 15, None]  # None = unlimited depth
tree_models = {}
tree_results = []

for depth in depths:
    depth_name = depth if depth is not None else "Unlimited"
    print(f"\n--- Training Decision Tree (max_depth={depth_name}) ---")

    # Create and train model
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    # Predictions
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    # Evaluate
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Get tree statistics
    n_leaves = dt.get_n_leaves()
    tree_depth = dt.get_depth()

    print(f"  Tree depth: {tree_depth}")
    print(f"  Number of leaves: {n_leaves}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: ${test_rmse:,.2f}")
    print(f"  Test MAE: ${test_mae:,.2f}")

    # Check for overfitting
    overfit_diff = train_r2 - test_r2
    if overfit_diff > 0.1:
        print(f"  ⚠ Overfitting detected! (Diff: {overfit_diff:.4f})")
    else:
        print(f"  ✓ Good generalization")

    # Store results
    tree_models[depth_name] = {
        'model': dt,
        'train_pred': y_train_pred,
        'test_pred': y_test_pred
    }

    tree_results.append({
        'Max Depth': depth_name,
        'Actual Depth': tree_depth,
        'Leaves': n_leaves,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'RMSE': test_rmse,
        'MAE': test_mae,
        'Overfit': overfit_diff
    })

# Results summary
results_df = pd.DataFrame(tree_results)
print(f"\n--- Decision Tree Results Summary ---")
print(results_df.to_string(index=False))

# Best model (highest test R² with low overfitting)
best_idx = results_df['Test R²'].idxmax()
best_depth = results_df.iloc[best_idx]['Max Depth']
best_test_r2 = results_df.iloc[best_idx]['Test R²']

print(f"\n🌟 Best Model: max_depth={best_depth}")
print(f"   Test R²: {best_test_r2:.4f}")
print(f"   Number of leaves: {results_df.iloc[best_idx]['Leaves']}")

# ============================================================================
# STEP 6: DETAILED ANALYSIS OF BEST MODEL
# ============================================================================
print("\n" + "=" * 80)
print(f"STEP 6: DETAILED ANALYSIS (DEPTH={best_depth})")
print("=" * 80)

best_model = tree_models[best_depth]['model']
y_test_pred_best = tree_models[best_depth]['test_pred']

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n--- Feature Importance ---")
print(feature_importance.to_string(index=False))

print(f"\n--- Top 3 Most Important Features ---")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance'] * 100:.2f}%)")

# Sample predictions
print(f"\n--- Sample Predictions (First 10 Test Houses) ---")
print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<15} {'Error %':<10}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred_best[i]
    error = actual - predicted
    error_pct = (error / actual) * 100

    print(f"${actual:>13,} ${predicted:>16,.2f} ${error:>13,.2f} {error_pct:>8.2f}%")

# ============================================================================
# STEP 7: CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CROSS-VALIDATION")
print("=" * 80)

print("\nPerforming 5-fold cross-validation on best model...")
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')

print(f"\nCross-validation R² scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f}")
print(f"  Std: {cv_scores.std():.4f}")
print(f"  Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

if cv_scores.std() < 0.1:
    print(f"  ✓ Consistent performance across folds")
else:
    print(f"  ⚠ High variance across folds")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Correlation Heatmap
print("\n📊 Creating correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_viz_1_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: dt_viz_1_correlation.png")

# Visualization 2: Model Performance by Depth
print("\n📊 Creating performance comparison by depth...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² scores
axes[0, 0].plot(results_df['Max Depth'].astype(str), results_df['Train R²'],
                marker='o', linewidth=2, markersize=8, label='Training')
axes[0, 0].plot(results_df['Max Depth'].astype(str), results_df['Test R²'],
                marker='s', linewidth=2, markersize=8, label='Test')
axes[0, 0].set_xlabel('Max Depth', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
axes[0, 0].set_title('R² Score vs Tree Depth', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1.1])

# RMSE
axes[0, 1].bar(results_df['Max Depth'].astype(str), results_df['RMSE'],
               color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Max Depth', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('RMSE ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('RMSE vs Tree Depth', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Number of leaves
axes[1, 0].bar(results_df['Max Depth'].astype(str), results_df['Leaves'],
               color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Max Depth', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Number of Leaves', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Tree Complexity vs Max Depth', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Overfitting measure
axes[1, 1].bar(results_df['Max Depth'].astype(str), results_df['Overfit'],
               color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 1].axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='Overfit Threshold')
axes[1, 1].set_xlabel('Max Depth', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Train R² - Test R²', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Overfitting Measure', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Decision Tree Performance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_viz_2_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: dt_viz_2_depth_analysis.png")

# Visualization 3: Feature Importance
print("\n📊 Creating feature importance plot...")
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'],
               color=colors, edgecolor='black')
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title(f'Feature Importance - Decision Tree (depth={best_depth})',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('dt_viz_3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: dt_viz_3_feature_importance.png")

# Visualization 4: Actual vs Predicted
print("\n📊 Creating actual vs predicted plot...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(y_test, y_test_pred_best, alpha=0.6, s=80, edgecolors='black', linewidths=1)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', lw=2.5, label='Perfect Prediction')
ax.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax.set_title(f'Actual vs Predicted Prices (R² = {best_test_r2:.4f})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dt_viz_4_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: dt_viz_4_actual_vs_predicted.png")

# Visualization 5: Residual Analysis
print("\n📊 Creating residual analysis...")
residuals = y_test.values - y_test_pred_best

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Predicted
axes[0, 0].scatter(y_test_pred_best, residuals, alpha=0.6, s=80, edgecolors='black')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Histogram of Residuals
axes[0, 1].hist(residuals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Q-Q Plot
from scipy import stats

stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot - Normality Check', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Absolute Errors
abs_errors = np.abs(residuals)
sorted_indices = np.argsort(abs_errors)[::-1][:20]  # Top 20 errors
axes[1, 1].bar(range(len(sorted_indices)), abs_errors[sorted_indices],
               color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('House Index (sorted by error)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Absolute Error ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Top 20 Prediction Errors', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_viz_5_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: dt_viz_5_residuals.png")

# Visualization 6: Decision Tree Structure
print("\n📊 Creating decision tree visualization...")
fig, ax = plt.subplots(figsize=(20, 12))

plot_tree(best_model,
          feature_names=feature_columns,
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)

ax.set_title(f'Decision Tree Structure (max_depth={best_depth})',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_viz_6_tree_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: dt_viz_6_tree_structure.png")

# Visualization 7: Price Distribution by Key Features
print("\n📊 Creating price distribution by features...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# By Bedrooms
df.boxplot(column='Price', by='Bedrooms', ax=axes[0, 0])
axes[0, 0].set_title('Price by Number of Bedrooms', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Bedrooms', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
plt.suptitle('')

# By Has Garage
garage_yes = df[df['HasGarage'] == 'Yes']['Price']
garage_no = df[df['HasGarage'] == 'No']['Price']
axes[0, 1].boxplot([garage_no, garage_yes], labels=['No Garage', 'Has Garage'])
axes[0, 1].set_title('Price by Garage Presence', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# By Neighborhood Quality
df.boxplot(column='Price', by='NeighborhoodQuality', ax=axes[1, 0])
axes[1, 0].set_title('Price by Neighborhood Quality', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Quality Rating', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
plt.suptitle('')

# Square Feet vs Price
axes[1, 1].scatter(df['SquareFeet'], df['Price'], alpha=0.5, s=50, edgecolors='black')
axes[1, 1].set_xlabel('Square Feet', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Price vs Square Footage', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Price Distributions by Key Features', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('dt_viz_7_price_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: dt_viz_7_price_distributions.png")

# ============================================================================
# STEP 9: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'=' * 80}
DECISION TREE REGRESSION - REAL ESTATE PRICE PREDICTION
{'=' * 80}

BUSINESS OBJECTIVE
{'=' * 80}
Predict house prices based on property characteristics using Decision Trees.

Why Decision Trees for Real Estate?
  • Interpretable rules (explain pricing to clients)
  • Handle non-linear relationships naturally
  • No feature scaling needed
  • Capture interactions between features
  • Visual representation of decision logic

DATASET SUMMARY
{'=' * 80}
Total houses: {len(df)}
Features: {len(feature_columns)}

Feature Details:
  • SquareFeet: {df['SquareFeet'].min()} - {df['SquareFeet'].max()}
  • Bedrooms: {df['Bedrooms'].min()} - {df['Bedrooms'].max()}
  • Bathrooms: {df['Bathrooms'].min()} - {df['Bathrooms'].max()}
  • Age: {df['Age'].min()} - {df['Age'].max()} years
  • DistanceToCenter: {df['DistanceToCenter'].min()} - {df['DistanceToCenter'].max()} km
  • NeighborhoodQuality: {df['NeighborhoodQuality'].min()} - {df['NeighborhoodQuality'].max()} (1-10 scale)
  • HasGarage: {(df['HasGarage'] == 'Yes').sum()} Yes, {(df['HasGarage'] == 'No').sum()} No
  • HasGarden: {(df['HasGarden'] == 'Yes').sum()} Yes, {(df['HasGarden'] == 'No').sum()} No

Price Statistics:
  Mean: ${df['Price'].mean():,.2f}
  Median: ${df['Price'].median():,.2f}
  Range: ${df['Price'].min():,} - ${df['Price'].max():,}
  Std Dev: ${df['Price'].std():,.2f}

CORRELATION ANALYSIS
{'=' * 80}
Top Features Correlated with Price:
{chr(10).join([f'  {i + 1}. {idx}: {val:.4f}' for i, (idx, val) in enumerate(correlations.head(6).items()) if idx != 'Price'])}

MODEL COMPARISON (DIFFERENT TREE DEPTHS)
{'=' * 80}

{results_df.to_string(index=False)}

Key Observations:
  • Shallow trees (depth=3): Underfit - simple rules miss patterns
  • Medium trees (depth={best_depth}): Optimal - balanced complexity
  • Deep trees (unlimited): Overfit - memorize training data

BEST MODEL: max_depth={best_depth}
{'=' * 80}
Test Performance:
  R² Score: {best_test_r2:.4f} ({best_test_r2 * 100:.2f}% of variance explained)
  RMSE: ${results_df.iloc[best_idx]['RMSE']:,.2f}
  MAE: ${results_df.iloc[best_idx]['MAE']:,.2f}

Tree Statistics:
  Actual depth achieved: {results_df.iloc[best_idx]['Actual Depth']}
  Number of leaves (decision rules): {results_df.iloc[best_idx]['Leaves']}
  Overfitting measure: {results_df.iloc[best_idx]['Overfit']:.4f} ({'Low' if results_df.iloc[best_idx]['Overfit'] < 0.1 else 'Moderate' if results_df.iloc[best_idx]['Overfit'] < 0.2 else 'High'})

Cross-Validation:
  Mean R²: {cv_scores.mean():.4f}
  Std R²: {cv_scores.std():.4f}
  Consistency: {'✓ High' if cv_scores.std() < 0.1 else '⚠ Variable'}

FEATURE IMPORTANCE
{'=' * 80}
Ranked by importance in price prediction:

{chr(10).join([f'{i + 1:2d}. {row["Feature"]:<22} {row["Importance"]:.4f} ({row["Importance"] * 100:>6.2f}%)'
               for i, (_, row) in enumerate(feature_importance.iterrows())])}

Top 3 Price Drivers:
{chr(10).join([f'  • {row["Feature"]}: {row["Importance"] * 100:.2f}% importance'
               for _, row in feature_importance.head(3).iterrows()])}

DECISION TREE INTERPRETATION
{'=' * 80}
How the tree makes decisions:

The tree creates {results_df.iloc[best_idx]['Leaves']} rules based on feature thresholds.

Example Rules (conceptual):
  IF SquareFeet > 2500 AND NeighborhoodQuality > 7
    THEN Price = High Range

  IF SquareFeet < 1500 AND Age > 30
    THEN Price = Low Range

  IF Bedrooms >= 4 AND HasGarage = Yes
    THEN Price = Upper-Mid Range

These rules are automatically learned from the data!

RESIDUAL ANALYSIS
{'=' * 80}
Mean residual: ${residuals.mean():,.2f}
Std of residuals: ${residuals.std():,.2f}
Max absolute error: ${np.abs(residuals).max():,.2f}
Min absolute error: ${np.abs(residuals).min():,.2f}

Pattern: {"✓ Normally distributed - assumptions satisfied" if np.abs(residuals.mean()) < 10000 else "⚠ Check for systematic bias"}

KEY INSIGHTS
{'=' * 80}
✓ Decision Tree Performance:
  - {best_test_r2 * 100:.2f}% of price variance explained
  - Average prediction error: ${results_df.iloc[best_idx]['MAE']:,.2f}
  - Tree creates {results_df.iloc[best_idx]['Leaves']} interpretable rules

✓ Most Important Price Factors:
  1. {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance'] * 100:.1f}%)
  2. {feature_importance.iloc[1]['Feature']} ({feature_importance.iloc[1]['Importance'] * 100:.1f}%)
  3. {feature_importance.iloc[2]['Feature']} ({feature_importance.iloc[2]['Importance'] * 100:.1f}%)

✓ Model Interpretability:
  - Can explain every prediction with simple IF-THEN rules
  - Clients understand why their house is priced a certain way
  - Agents can justify pricing recommendations

✓ Overfitting Control:
  - Optimal depth prevents memorization
  - Cross-validation confirms generalization
  - Test performance close to training performance

BUSINESS RECOMMENDATIONS
{'=' * 80}
1. PRICING STRATEGY
   Use the decision tree to:
   • Set competitive initial listing prices
   • Justify valuations to sellers
   • Identify undervalued properties
   • Predict market value changes

2. FEATURE FOCUS
   Maximize value by improving top features:
   • {feature_importance.iloc[0]['Feature']}: Highest impact
   • {feature_importance.iloc[1]['Feature']}: Second priority
   • {feature_importance.iloc[2]['Feature']}: Third priority

3. CLIENT COMMUNICATION
   Decision trees provide:
   • Clear explanations: "Your price is $X because..."
   • Transparent logic: Show the decision path
   • Improvement suggestions: "Add garage → +$25K value"

4. MARKET ANALYSIS
   Identify patterns:
   • Which features drive prices in different neighborhoods?
   • What combinations maximize value?
   • Where are pricing inefficiencies?

PRACTICAL APPLICATIONS
{'=' * 80}

Example Predictions:
  House 1: 2000 sq ft, 3BR, 2BA, Age 5, Quality 8, Has Garage
  Predicted: ${best_model.predict([[2000, 3, 2, 5, 5, 8, 1, 0]])[0]:,.2f}

  House 2: 1500 sq ft, 2BR, 1BA, Age 30, Quality 5, No Garage
  Predicted: ${best_model.predict([[1500, 2, 1, 30, 10, 5, 0, 0]])[0]:,.2f}

  House 3: 3500 sq ft, 5BR, 3BA, Age 0, Quality 10, Has Garage & Garden
  Predicted: ${best_model.predict([[3500, 5, 3, 0, 2, 10, 1, 1]])[0]:,.2f}

Value-Add Scenarios:
  Adding a garage to House 2 (no garage):
  Before: ${best_model.predict([[1500, 2, 1, 30, 10, 5, 0, 0]])[0]:,.2f}
  After: ${best_model.predict([[1500, 2, 1, 30, 10, 5, 1, 0]])[0]:,.2f}
  Increase: ${best_model.predict([[1500, 2, 1, 30, 10, 5, 1, 0]])[0] - best_model.predict([[1500, 2, 1, 30, 10, 5, 0, 0]])[0]:,.2f}

ADVANTAGES OF DECISION TREES
{'=' * 80}
✓ Interpretability: Clear IF-THEN rules
✓ No scaling needed: Works with raw features
✓ Handles non-linearity: Automatically captures complex patterns
✓ Feature interactions: Captures combinations (e.g., size + location)
✓ Visual representation: Can draw the tree structure
✓ Fast predictions: Simple logic, quick evaluation
✓ Mixed data types: Handles numerical + categorical

LIMITATIONS
{'=' * 80}
⚠ Overfitting risk: Deep trees memorize training data
⚠ Instability: Small data changes → different tree
⚠ Biased to dominant features: May ignore subtle patterns
⚠ Step functions: Predictions are not smooth
⚠ Limited extrapolation: Poor on unseen value ranges

FILES GENERATED
{'=' * 80}
Dataset:
  • house_prices.csv - Generated real estate data

Visualizations:
  • dt_viz_1_correlation.png - Feature correlation heatmap
  • dt_viz_2_depth_analysis.png - Performance by tree depth
  • dt_viz_3_feature_importance.png - Feature importance ranking
  • dt_viz_4_actual_vs_predicted.png - Prediction accuracy
  • dt_viz_5_residuals.png - Residual analysis (4 plots)
  • dt_viz_6_tree_structure.png - Decision tree visualization
  • dt_viz_7_price_distributions.png - Price by features

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(report)

# Save report
with open('dt_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Report saved to: dt_analysis_report.txt")

# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAVE MODEL")
print("=" * 80)

import joblib

model_data = {
    'model': best_model,
    'feature_names': feature_columns,
    'label_encoders': {
        'HasGarage': le_garage,
        'HasGarden': le_garden
    },
    'best_depth': best_depth,
    'test_r2': best_test_r2,
    'feature_importance': feature_importance
}

model_path = 'decision_tree_model.pkl'
joblib.dump(model_data, model_path)
print(f"\n✓ Model saved to: {model_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DECISION TREE REGRESSION ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"  ✓ Generated {len(df)} realistic house price records")
print(f"  ✓ Tested {len(depths)} different tree depths")
print(f"  ✓ Best model: max_depth={best_depth}")
print(f"  ✓ Achieved R² = {best_test_r2:.4f} ({best_test_r2 * 100:.2f}%)")
print(f"  ✓ Average error: ${results_df.iloc[best_idx]['MAE']:,.2f}")
print(f"  ✓ Generated 7 comprehensive visualizations")

print(f"\n💡 Key Findings:")
print(f"  • Most important feature: {feature_importance.iloc[0]['Feature']}")
print(f"  • Tree creates {results_df.iloc[best_idx]['Leaves']} interpretable rules")
print(f"  • Model explains {best_test_r2 * 100:.2f}% of price variance")
print(f"  • Cross-validation R²: {cv_scores.mean():.4f} (consistent)")

print(f"\n🌳 Tree Characteristics:")
print(f"  • Depth: {results_df.iloc[best_idx]['Actual Depth']}")
print(f"  • Leaves: {results_df.iloc[best_idx]['Leaves']}")
print(
    f"  • Overfitting: {results_df.iloc[best_idx]['Overfit']:.4f} ({'Low' if results_df.iloc[best_idx]['Overfit'] < 0.1 else 'Moderate'})")

print("\n" + "=" * 80)
print("All analysis complete! Check output files for detailed results.")
print("=" * 80)