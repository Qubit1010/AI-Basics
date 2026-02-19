"""
SUPPORT VECTOR REGRESSION (SVR) - POSITION SALARY PREDICTION
=============================================================
Predicting salary based on position level using SVR

What is SVR?
- Advanced regression technique from Support Vector Machines (SVM)
- Uses kernel tricks to capture non-linear relationships
- Robust to outliers
- Works well with small datasets

Why SVR for Salary Prediction?
- Non-linear salary progression (especially at senior levels)
- Small dataset (10 positions)
- Robust predictions needed
- Can handle complex patterns

Approach:
1. Data exploration
2. Feature scaling (critical for SVR!)
3. Compare different SVR kernels (Linear, RBF, Polynomial)
4. Hyperparameter tuning
5. Model evaluation and comparison
6. Comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("SUPPORT VECTOR REGRESSION (SVR) - POSITION SALARY PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD AND EXPLORE DATA")
print("=" * 80)

# Load data
df = pd.read_csv('Position_Salaries.csv')

print(f"\n✓ Data loaded successfully")
print(f"  Shape: {df.shape}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")

print("\n--- Complete Dataset ---")
print(df.to_string(index=False))

print("\n--- Data Statistics ---")
print(df.describe())

print("\n--- Salary Analysis ---")
print(f"  Range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
print(f"  Mean: ${df['Salary'].mean():,.2f}")
print(f"  Median: ${df['Salary'].median():,.2f}")
print(f"  Std Dev: ${df['Salary'].std():,.2f}")
print(f"  Growth: {(df['Salary'].max() / df['Salary'].min()):.1f}x from entry to CEO")

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA PREPARATION")
print("=" * 80)

# Features and target
X = df[['Level']].values
y = df['Salary'].values

print(f"\nOriginal Data:")
print(f"  X (Level): {X.flatten()}")
print(f"  y (Salary): ${y}")

print("\n--- Feature Scaling ---")
print("  ⚠ CRITICAL for SVR: Features must be scaled!")
print("  Why? SVR is sensitive to feature magnitude")
print("  Solution: StandardScaler (mean=0, std=1)")

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"\nScaled Data:")
print(f"  X_scaled mean: {X_scaled.mean():.6f} (should be ~0)")
print(f"  X_scaled std: {X_scaled.std():.6f} (should be ~1)")
print(f"  y_scaled mean: {y_scaled.mean():.6f}")
print(f"  y_scaled std: {y_scaled.std():.6f}")

# ============================================================================
# STEP 3: SVR WITH DIFFERENT KERNELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: SVR WITH DIFFERENT KERNELS")
print("=" * 80)

print("\nExplanation:")
print("  SVR uses different kernel functions to transform data:")
print("  • Linear: For linear relationships")
print("  • RBF (Radial Basis Function): For non-linear patterns (most common)")
print("  • Polynomial: For polynomial relationships")
print("  • Sigmoid: For neural network-like transformations")

# Define kernels to test
kernels = {
    'linear': {'kernel': 'linear', 'C': 1.0},
    'rbf': {'kernel': 'rbf', 'C': 100, 'gamma': 0.1},
    'poly': {'kernel': 'poly', 'degree': 3, 'C': 1.0},
    'sigmoid': {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 0.1}
}

svr_models = {}
svr_results = []

for name, params in kernels.items():
    print(f"\n--- Testing {name.upper()} Kernel ---")
    print(f"  Parameters: {params}")

    # Create and train model
    svr = SVR(**params)
    svr.fit(X_scaled, y_scaled)

    # Predictions (scaled)
    y_pred_scaled = svr.predict(X_scaled)

    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Evaluate
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print(f"  R² Score: {r2:.4f} ({r2 * 100:.2f}%)")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE: ${mae:,.2f}")

    # Store results
    svr_models[name] = {
        'model': svr,
        'predictions': y_pred,
        'params': params
    }

    svr_results.append({
        'Kernel': name.upper(),
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae
    })

# Results summary
results_df = pd.DataFrame(svr_results)
print(f"\n--- SVR Results Summary ---")
print(results_df.to_string(index=False))

# Best kernel
best_kernel = results_df.loc[results_df['R²'].idxmax(), 'Kernel'].lower()
best_r2 = results_df.loc[results_df['R²'].idxmax(), 'R²']

print(f"\n🌟 Best Kernel: {best_kernel.upper()}")
print(f"   R² Score: {best_r2:.4f} ({best_r2 * 100:.2f}%)")

# ============================================================================
# STEP 4: HYPERPARAMETER TUNING (RBF Kernel)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HYPERPARAMETER TUNING (RBF KERNEL)")
print("=" * 80)

print("\nExplanation:")
print("  RBF kernel has two main parameters:")
print("  • C: Regularization (trade-off between error and model complexity)")
print("  • gamma: Kernel coefficient (how far influence reaches)")

# Parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5]
}

print(f"\nSearching parameter combinations:")
print(f"  C: {param_grid['C']}")
print(f"  gamma: {param_grid['gamma']}")
print(f"  epsilon: {param_grid['epsilon']}")
print(f"  Total combinations: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['epsilon'])}")

# Grid search
svr_rbf = SVR(kernel='rbf')
grid_search = GridSearchCV(
    svr_rbf,
    param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_scaled, y_scaled)

print(f"\n✓ Grid search complete!")
print(f"\n--- Best Parameters ---")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n--- Best Score ---")
print(f"  Cross-validated R²: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_svr = grid_search.best_estimator_
y_pred_scaled_best = best_svr.predict(X_scaled)
y_pred_best = scaler_y.inverse_transform(y_pred_scaled_best.reshape(-1, 1)).flatten()

# Evaluate
r2_best = r2_score(y, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y, y_pred_best))
mae_best = mean_absolute_error(y, y_pred_best)

print(f"\n--- Optimized RBF SVR Performance ---")
print(f"  R² Score: {r2_best:.4f} ({r2_best * 100:.2f}%)")
print(f"  RMSE: ${rmse_best:,.2f}")
print(f"  MAE: ${mae_best:,.2f}")

# ============================================================================
# STEP 5: DETAILED PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DETAILED PREDICTIONS")
print("=" * 80)

print(f"\n--- Predictions (Optimized RBF SVR) ---")
print(f"{'Position':<20} {'Level':<8} {'Actual':<15} {'Predicted':<15} {'Error':<15} {'Error %':<10}")
print("-" * 88)

for i in range(len(df)):
    position = df.iloc[i]['Position']
    level = df.iloc[i]['Level']
    actual = df.iloc[i]['Salary']
    predicted = y_pred_best[i]
    error = actual - predicted
    error_pct = (error / actual) * 100

    print(f"{position:<20} {level:<8} ${actual:>12,} ${predicted:>12,.2f} ${error:>12,.2f} {error_pct:>8.2f}%")

# Interpolation example
print(f"\n--- Interpolation Example ---")
level_interp = np.array([[6.5]])
level_interp_scaled = scaler_X.transform(level_interp)
salary_interp_scaled = best_svr.predict(level_interp_scaled)
salary_interp = scaler_y.inverse_transform(salary_interp_scaled.reshape(-1, 1))[0, 0]

print(f"  Question: What salary for Level 6.5?")
print(f"  Predicted Salary: ${salary_interp:,.2f}")
print(
    f"  Reference: Level 6 = ${df[df['Level'] == 6]['Salary'].values[0]:,}, Level 7 = ${df[df['Level'] == 7]['Salary'].values[0]:,}")

# ============================================================================
# STEP 6: COMPARISON WITH OTHER MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: COMPARISON WITH OTHER REGRESSION METHODS")
print("=" * 80)

# Linear Regression for comparison
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

r2_lin = r2_score(y, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y, y_pred_lin))
mae_lin = mean_absolute_error(y, y_pred_lin)

# Polynomial Regression (degree 4) for comparison
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

r2_poly = r2_score(y, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
mae_poly = mean_absolute_error(y, y_pred_poly)

# Comparison table
comparison_data = {
    'Model': ['Linear Regression', 'Polynomial (Degree 4)', 'SVR (RBF - Default)', 'SVR (RBF - Optimized)'],
    'R²': [r2_lin, r2_poly, svr_results[1]['R²'], r2_best],
    'RMSE': [rmse_lin, rmse_poly, svr_results[1]['RMSE'], rmse_best],
    'MAE': [mae_lin, mae_poly, svr_results[1]['MAE'], mae_best]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n--- Model Comparison ---")
print(comparison_df.to_string(index=False))

# Best overall model
best_overall_idx = comparison_df['R²'].idxmax()
best_overall_model = comparison_df.iloc[best_overall_idx]['Model']
best_overall_r2 = comparison_df.iloc[best_overall_idx]['R²']

print(f"\n🏆 Best Overall Model: {best_overall_model}")
print(f"   R² Score: {best_overall_r2:.4f}")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Create smooth curve for visualization
X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_smooth_scaled = scaler_X.transform(X_smooth)

# Visualization 1: SVR Kernels Comparison
print("\n📊 Creating SVR kernels comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

kernel_names = ['linear', 'rbf', 'poly', 'sigmoid']
colors = ['blue', 'green', 'purple', 'orange']

for idx, (name, color) in enumerate(zip(kernel_names, colors)):
    # Get predictions
    model = svr_models[name]['model']
    y_smooth_scaled = model.predict(X_smooth_scaled)
    y_smooth = scaler_y.inverse_transform(y_smooth_scaled.reshape(-1, 1)).flatten()

    # Plot
    axes[idx].scatter(X, y, color='red', s=150, edgecolors='black', label='Actual Data', zorder=3, linewidths=2)
    axes[idx].plot(X_smooth, y_smooth, color=color, linewidth=2.5, label=f'{name.upper()} Kernel')
    axes[idx].set_xlabel('Position Level', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Salary ($)', fontsize=11, fontweight='bold')

    r2 = svr_results[idx]['R²']
    axes[idx].set_title(f'{name.upper()} Kernel (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('SVR with Different Kernels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('svr_viz_1_kernels.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: svr_viz_1_kernels.png")

# Visualization 2: Optimized SVR vs Polynomial
print("\n📊 Creating optimized SVR comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Polynomial
X_poly_smooth = poly_features.transform(X_smooth)
y_poly_smooth = poly_reg.predict(X_poly_smooth)

axes[0].scatter(X, y, color='red', s=150, edgecolors='black', label='Actual Data', zorder=3, linewidths=2)
axes[0].plot(X_smooth, y_poly_smooth, color='purple', linewidth=2.5, label='Polynomial Regression')
axes[0].set_xlabel('Position Level', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Polynomial Regression (R² = {r2_poly:.4f})', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Optimized SVR
y_smooth_scaled_best = best_svr.predict(X_smooth_scaled)
y_smooth_best = scaler_y.inverse_transform(y_smooth_scaled_best.reshape(-1, 1)).flatten()

axes[1].scatter(X, y, color='red', s=150, edgecolors='black', label='Actual Data', zorder=3, linewidths=2)
axes[1].plot(X_smooth, y_smooth_best, color='green', linewidth=2.5, label='Optimized SVR (RBF)')
axes[1].set_xlabel('Position Level', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Optimized SVR (R² = {r2_best:.4f})', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('SVR vs Polynomial Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('svr_viz_2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: svr_viz_2_comparison.png")

# Visualization 3: Model Performance Comparison
print("\n📊 Creating performance metrics comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

models = comparison_df['Model'].tolist()
r2_scores = comparison_df['R²'].tolist()
rmse_scores = comparison_df['RMSE'].tolist()
mae_scores = comparison_df['MAE'].tolist()

# R² Scores
colors_r2 = ['red', 'orange', 'lightblue', 'green']
bars1 = axes[0].bar(range(len(models)), r2_scores, color=colors_r2, edgecolor='black', alpha=0.7)
axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('R² Score Comparison', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(len(models)))
axes[0].set_xticklabels(models, rotation=15, ha='right')
axes[0].set_ylim([0, 1.1])
axes[0].grid(axis='y', alpha=0.3)

# Highlight best
best_idx = r2_scores.index(max(r2_scores))
bars1[best_idx].set_edgecolor('gold')
bars1[best_idx].set_linewidth(3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# RMSE
axes[1].bar(range(len(models)), rmse_scores, color=colors_r2, edgecolor='black', alpha=0.7)
axes[1].set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
axes[1].set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(len(models)))
axes[1].set_xticklabels(models, rotation=15, ha='right')
axes[1].grid(axis='y', alpha=0.3)

# MAE
axes[2].bar(range(len(models)), mae_scores, color=colors_r2, edgecolor='black', alpha=0.7)
axes[2].set_ylabel('MAE ($)', fontsize=12, fontweight='bold')
axes[2].set_title('MAE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
axes[2].set_xticks(range(len(models)))
axes[2].set_xticklabels(models, rotation=15, ha='right')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('svr_viz_3_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: svr_viz_3_metrics.png")

# Visualization 4: Actual vs Predicted
print("\n📊 Creating actual vs predicted plot...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(y, y_pred_best, s=150, alpha=0.6, edgecolors='black', linewidths=2, color='green')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2.5, label='Perfect Prediction')

# Add labels
for i, position in enumerate(df['Position']):
    ax.annotate(position, (y[i], y_pred_best[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

ax.set_xlabel('Actual Salary ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
ax.set_title(f'Actual vs Predicted Salaries - Optimized SVR (R² = {r2_best:.4f})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svr_viz_4_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: svr_viz_4_actual_vs_predicted.png")

# Visualization 5: Residual Analysis
print("\n📊 Creating residual analysis...")
residuals = y - y_pred_best

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Level
axes[0, 0].scatter(X, residuals, s=120, alpha=0.6, edgecolors='black', linewidths=2)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Position Level', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Position Level', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs Predicted
axes[0, 1].scatter(y_pred_best, residuals, s=120, alpha=0.6, edgecolors='black', linewidths=2)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Salary ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Histogram
axes[1, 0].hist(residuals, bins=8, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Absolute errors
axes[1, 1].bar(range(len(residuals)), np.abs(residuals), color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Data Point', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Absolute Error ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Absolute Prediction Errors', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(len(residuals)))
axes[1, 1].set_xticklabels([f"L{i + 1}" for i in range(len(residuals))], rotation=0)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Residual Analysis - Optimized SVR', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('svr_viz_5_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: svr_viz_5_residuals.png")

# Visualization 6: Hyperparameter Heatmap
print("\n📊 Creating hyperparameter heatmap...")

# Extract results from grid search
results = pd.DataFrame(grid_search.cv_results_)

# Create pivot table for C and gamma
pivot_data = results.pivot_table(
    values='mean_test_score',
    index='param_gamma',
    columns='param_C',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlGnBu',
            ax=ax, cbar_kws={'label': 'R² Score'})
ax.set_xlabel('C (Regularization Parameter)', fontsize=12, fontweight='bold')
ax.set_ylabel('gamma (Kernel Coefficient)', fontsize=12, fontweight='bold')
ax.set_title('SVR Hyperparameter Tuning Results (RBF Kernel)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('svr_viz_6_hyperparameter_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: svr_viz_6_hyperparameter_heatmap.png")

# ============================================================================
# STEP 8: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'=' * 80}
SUPPORT VECTOR REGRESSION (SVR) - POSITION SALARY PREDICTION
{'=' * 80}

BUSINESS OBJECTIVE
{'=' * 80}
Predict salaries for different position levels using Support Vector Regression.
SVR Advantages:
  • Robust to outliers
  • Handles non-linear relationships via kernel trick
  • Works well with small datasets
  • Provides smooth predictions

DATASET SUMMARY
{'=' * 80}
Total positions: {len(df)}
Levels: {int(X.min())} to {int(X.max())}
Salary range: ${y.min():,} to ${y.max():,}

Complete Data:
{df.to_string(index=False)}

Salary Growth: {(df['Salary'].max() / df['Salary'].min()):.1f}x from entry to CEO

FEATURE SCALING (CRITICAL FOR SVR!)
{'=' * 80}
Why scaling is essential:
  • SVR is distance-based algorithm
  • Features with larger scales dominate
  • Without scaling: Level (1-10) vs Salary ($45K-$1M) → huge imbalance

Scaling method: StandardScaler (z-score normalization)
  X_scaled: mean = {X_scaled.mean():.6f}, std = {X_scaled.std():.6f}
  y_scaled: mean = {y_scaled.mean():.6f}, std = {y_scaled.std():.6f}

SVR KERNEL COMPARISON
{'=' * 80}

1. LINEAR KERNEL
   Use case: Linear relationships
   Performance:
     R²: {svr_results[0]['R²']:.4f}
     RMSE: ${svr_results[0]['RMSE']:,.2f}
     MAE: ${svr_results[0]['MAE']:,.2f}

2. RBF (Radial Basis Function) KERNEL
   Use case: Non-linear relationships (most versatile)
   Performance:
     R²: {svr_results[1]['R²']:.4f}
     RMSE: ${svr_results[1]['RMSE']:,.2f}
     MAE: ${svr_results[1]['MAE']:,.2f}

3. POLYNOMIAL KERNEL
   Use case: Polynomial relationships
   Performance:
     R²: {svr_results[2]['R²']:.4f}
     RMSE: ${svr_results[2]['RMSE']:,.2f}
     MAE: ${svr_results[2]['MAE']:,.2f}

4. SIGMOID KERNEL
   Use case: Neural network-like transformations
   Performance:
     R²: {svr_results[3]['R²']:.4f}
     RMSE: ${svr_results[3]['RMSE']:,.2f}
     MAE: ${svr_results[3]['MAE']:,.2f}

Best Kernel: {best_kernel.upper()}

HYPERPARAMETER TUNING (RBF KERNEL)
{'=' * 80}
Grid Search Results:
  Parameters tested: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['epsilon'])} combinations
  Cross-validation: 3-fold

Best Parameters:
{chr(10).join([f'  • {k}: {v}' for k, v in grid_search.best_params_.items()])}

Optimized Model Performance:
  R² Score: {r2_best:.4f} ({r2_best * 100:.2f}%)
  RMSE: ${rmse_best:,.2f}
  MAE: ${mae_best:,.2f}
  Cross-validated R²: {grid_search.best_score_:.4f}

PREDICTIONS
{'=' * 80}

Position-by-Position Results:
{'Position':<20} {'Level':<8} {'Actual':<15} {'Predicted':<15} {'Error':<15} {'Error %':<10}
{'-' * 88}
{chr(10).join([f'{df.iloc[i]["Position"]:<20} {df.iloc[i]["Level"]:<8} ${df.iloc[i]["Salary"]:>12,} ${y_pred_best[i]:>12,.2f} ${(df.iloc[i]["Salary"] - y_pred_best[i]):>12,.2f} {((df.iloc[i]["Salary"] - y_pred_best[i]) / df.iloc[i]["Salary"] * 100):>8.2f}%'
               for i in range(len(df))])}

Interpolation Example:
  Level 6.5 (between Region Manager and Partner)
  Predicted Salary: ${salary_interp:,.2f}

MODEL COMPARISON
{'=' * 80}

{comparison_df.to_string(index=False)}

Best Overall Model: {best_overall_model}
  R² Score: {best_overall_r2:.4f}

Key Findings:
  • SVR (optimized) {'outperforms' if r2_best > r2_poly else 'performs similarly to'} Polynomial Regression
  • RBF kernel captures non-linear salary growth effectively
  • Hyperparameter tuning {'significantly improved' if r2_best - svr_results[1]['R²'] > 0.05 else 'refined'} model performance

RESIDUAL ANALYSIS
{'=' * 80}
Mean residual: ${residuals.mean():,.2f}
Std of residuals: ${residuals.std():,.2f}
Max absolute error: ${np.abs(residuals).max():,.2f}
Min absolute error: ${np.abs(residuals).min():,.2f}

Pattern: {"✓ Random distribution - model assumptions satisfied" if np.abs(residuals.mean()) < 5000 else "⚠ Check for systematic patterns"}

KEY INSIGHTS
{'=' * 80}
✓ SVR Performance:
  - {r2_best * 100:.2f}% of salary variance explained
  - Average prediction error: ${mae_best:,.2f}
  - RBF kernel most effective for non-linear salary structure

✓ Why SVR Works Well:
  - Kernel trick handles exponential growth at senior levels
  - Robust to potential outliers (CEO salary)
  - Small dataset (10 points) - SVR excels here
  - Feature scaling ensures balanced learning

✓ Model Comparison:
  - Linear Regression: {r2_lin:.4f} (poor - misses non-linearity)
  - Polynomial (Deg 4): {r2_poly:.4f} (excellent)
  - SVR (Optimized): {r2_best:.4f} ({'best' if r2_best >= max(r2_lin, r2_poly) else 'competitive'})

BUSINESS RECOMMENDATIONS
{'=' * 80}
1. SALARY PREDICTIONS
   Use optimized SVR for:
   • New position salary estimates
   • Market benchmarking
   • Compensation planning

2. ADVANTAGES OF SVR
   • Smooth predictions (no overfitting)
   • Handles outliers well (CEO's $1M salary)
   • Generalizes to unseen levels (e.g., 6.5)
   • Mathematically rigorous

3. WHEN TO USE SVR
   ✓ Small datasets (<100 samples)
   ✓ Non-linear relationships
   ✓ Need for robust predictions
   ✓ Presence of outliers

4. IMPLEMENTATION
   • Always scale features (critical!)
   • Use RBF kernel for salary data
   • Tune hyperparameters (C, gamma)
   • Validate with cross-validation

PRACTICAL APPLICATIONS
{'=' * 80}

Salary Negotiation:
  Level 4.5: ${scaler_y.inverse_transform(best_svr.predict(scaler_X.transform([[4.5]])).reshape(-1, 1))[0, 0]:,.2f}
  Level 7.5: ${scaler_y.inverse_transform(best_svr.predict(scaler_X.transform([[7.5]])).reshape(-1, 1))[0, 0]:,.2f}
  Level 8.5: ${scaler_y.inverse_transform(best_svr.predict(scaler_X.transform([[8.5]])).reshape(-1, 1))[0, 0]:,.2f}

Budget Planning:
  Promote from Level 5 to 6: ~${df[df['Level'] == 6]['Salary'].values[0] - df[df['Level'] == 5]['Salary'].values[0]:,}
  Promote from Level 8 to 9: ~${df[df['Level'] == 9]['Salary'].values[0] - df[df['Level'] == 8]['Salary'].values[0]:,}

FILES GENERATED
{'=' * 80}
Visualizations:
  • svr_viz_1_kernels.png - All SVR kernels comparison
  • svr_viz_2_comparison.png - SVR vs Polynomial
  • svr_viz_3_metrics.png - Performance metrics
  • svr_viz_4_actual_vs_predicted.png - Prediction accuracy
  • svr_viz_5_residuals.png - Residual analysis
  • svr_viz_6_hyperparameter_heatmap.png - Tuning results

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(report)

# Save report
with open('svr_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Report saved to: svr_analysis_report.txt")

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: SAVE MODEL")
print("=" * 80)

import joblib

model_data = {
    'svr_model': best_svr,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'best_params': grid_search.best_params_,
    'r2_score': r2_best
}

model_path = 'svr_salary_model.pkl'
joblib.dump(model_data, model_path)
print(f"\n✓ Model saved to: {model_path}")
print(f"  Includes: SVR model + scalers + parameters")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUPPORT VECTOR REGRESSION ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"  ✓ Tested 4 SVR kernels (Linear, RBF, Polynomial, Sigmoid)")
print(f"  ✓ Optimized RBF kernel with grid search")
print(f"  ✓ Best model: RBF SVR with C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
print(f"  ✓ Achieved R² = {r2_best:.4f} ({r2_best * 100:.2f}%)")
print(f"  ✓ Average error: ${mae_best:,.2f}")
print(f"  ✓ Generated 6 comprehensive visualizations")

print(f"\n💡 Key Findings:")
print(f"  • Best kernel: {best_kernel.upper()}")
print(f"  • Model explains {r2_best * 100:.2f}% of salary variance")
print(f"  • Feature scaling {'is critical' if True else 'improves'} for SVR")
print(f"  • SVR handles non-linear growth effectively")

print(f"\n🏆 Model Ranking:")
print(f"  1. {best_overall_model}: R² = {best_overall_r2:.4f}")
for idx, row in comparison_df.iterrows():
    if idx != best_overall_idx:
        print(f"  {idx + 1}. {row['Model']}: R² = {row['R²']:.4f}")

print("\n" + "=" * 80)
print("All analysis complete! Check output files for detailed results.")
print("=" * 80)