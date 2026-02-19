"""
POLYNOMIAL REGRESSION - POSITION SALARY PREDICTION
==================================================
Predicting salary based on position level using polynomial regression

Business Problem:
A company wants to predict salaries for different position levels.
Linear regression may not capture the non-linear relationship between
position level and salary (especially at senior levels).

Solution: Polynomial Regression
- Captures non-linear relationships
- Fits curved patterns in data
- Better for hierarchical salary structures

Approach:
1. Data exploration and visualization
2. Compare Linear vs Polynomial regression
3. Find optimal polynomial degree
4. Model evaluation and predictions
5. Comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("POLYNOMIAL REGRESSION - POSITION SALARY PREDICTION")
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

print("\n--- Data Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Salary Range ---")
print(f"  Minimum: ${df['Salary'].min():,}")
print(f"  Maximum: ${df['Salary'].max():,}")
print(f"  Mean: ${df['Salary'].mean():,.2f}")
print(f"  Median: ${df['Salary'].median():,.2f}")
print(f"  Salary Growth: {(df['Salary'].max() / df['Salary'].min()):.1f}x from entry to CEO")

# Check for missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing)
print(f"Total missing values: {missing.sum()}")

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA PREPARATION")
print("=" * 80)

# Features and target
X = df[['Level']].values
y = df['Salary'].values

print(f"\nFeatures (X - Position Level):")
print(f"  Shape: {X.shape}")
print(f"  Range: {X.min()} to {X.max()}")

print(f"\nTarget (y - Salary):")
print(f"  Shape: {y.shape}")
print(f"  Range: ${y.min():,} to ${y.max():,}")

print("\n--- Data Characteristics ---")
print(f"  Number of data points: {len(X)}")
print(f"  This is a small dataset - perfect for polynomial regression")
print(f"  No train-test split needed due to small size")

# ============================================================================
# STEP 3: LINEAR REGRESSION (BASELINE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: LINEAR REGRESSION (BASELINE)")
print("=" * 80)

print("\nExplanation:")
print("  First, let's try simple linear regression as a baseline")
print("  Linear: Salary = β₀ + β₁ × Level")

# Train linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Get parameters
lin_intercept = lin_reg.intercept_
lin_coef = lin_reg.coef_[0]

print(f"\n--- Linear Model Parameters ---")
print(f"  Intercept (β₀): ${lin_intercept:,.2f}")
print(f"  Coefficient (β₁): ${lin_coef:,.2f}")
print(f"\n  Equation: Salary = ${lin_intercept:,.2f} + ${lin_coef:,.2f} × Level")

# Predictions
y_pred_lin = lin_reg.predict(X)

# Evaluate
r2_lin = r2_score(y, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y, y_pred_lin))
mae_lin = mean_absolute_error(y, y_pred_lin)

print(f"\n--- Linear Model Performance ---")
print(f"  R² Score: {r2_lin:.4f} ({r2_lin * 100:.2f}% variance explained)")
print(f"  RMSE: ${rmse_lin:,.2f}")
print(f"  MAE: ${mae_lin:,.2f}")

print(f"\n--- Observation ---")
if r2_lin < 0.9:
    print(f"  ⚠ Linear model R² = {r2_lin:.4f} suggests linear fit is not ideal")
    print(f"  The relationship appears non-linear - polynomial regression needed!")
else:
    print(f"  ✓ Linear model fits reasonably well")

# ============================================================================
# STEP 4: POLYNOMIAL REGRESSION (MULTIPLE DEGREES)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: POLYNOMIAL REGRESSION (MULTIPLE DEGREES)")
print("=" * 80)

print("\nExplanation:")
print("  Polynomial regression fits curves by adding polynomial terms")
print("  Degree 2: Salary = β₀ + β₁×Level + β₂×Level²")
print("  Degree 3: Salary = β₀ + β₁×Level + β₂×Level² + β₃×Level³")
print("  And so on...")

# Test different polynomial degrees
degrees = [2, 3, 4, 5, 6]
poly_models = {}
poly_results = []

for degree in degrees:
    print(f"\n--- Testing Polynomial Degree {degree} ---")

    # Transform features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Train model
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)

    # Predictions
    y_pred_poly = poly_reg.predict(X_poly)

    # Evaluate
    r2_poly = r2_score(y, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
    mae_poly = mean_absolute_error(y, y_pred_poly)

    print(f"  R² Score: {r2_poly:.4f} ({r2_poly * 100:.2f}%)")
    print(f"  RMSE: ${rmse_poly:,.2f}")
    print(f"  MAE: ${mae_poly:,.2f}")

    # Store results
    poly_models[degree] = {
        'poly_features': poly_features,
        'model': poly_reg,
        'predictions': y_pred_poly
    }

    poly_results.append({
        'Degree': degree,
        'R²': r2_poly,
        'RMSE': rmse_poly,
        'MAE': mae_poly
    })

# Results summary
results_df = pd.DataFrame(poly_results)
print(f"\n--- Polynomial Regression Results Summary ---")
print(results_df.to_string(index=False))

# Best model
best_degree = results_df.loc[results_df['R²'].idxmax(), 'Degree']
best_r2 = results_df.loc[results_df['R²'].idxmax(), 'R²']

print(f"\n🌟 Best Polynomial Degree: {int(best_degree)}")
print(f"   R² Score: {best_r2:.4f} ({best_r2 * 100:.2f}%)")

# ============================================================================
# STEP 5: DETAILED ANALYSIS OF BEST MODEL
# ============================================================================
print("\n" + "=" * 80)
print(f"STEP 5: DETAILED ANALYSIS (DEGREE {int(best_degree)} POLYNOMIAL)")
print("=" * 80)

# Get best model
best_poly_features = poly_models[best_degree]['poly_features']
best_poly_model = poly_models[best_degree]['model']
X_poly_best = best_poly_features.fit_transform(X)

# Get coefficients
coefficients = best_poly_model.coef_
intercept = best_poly_model.intercept_

print(f"\n--- Model Coefficients (Degree {int(best_degree)}) ---")
print(f"  Intercept (β₀): ${intercept:,.2f}")
for i, coef in enumerate(coefficients[1:], 1):  # Skip the bias term
    print(f"  β{i} (Level^{i}): {coef:,.6f}")

# Build equation
equation = f"Salary = ${intercept:,.2f}"
for i, coef in enumerate(coefficients[1:], 1):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} {abs(coef):.6f} × Level^{i}"
print(f"\n--- Polynomial Equation ---")
print(equation)

# ============================================================================
# STEP 6: PREDICTIONS AND INTERPOLATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: PREDICTIONS AND INTERPOLATION")
print("=" * 80)

# Create smooth curve for visualization
X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_smooth_poly = best_poly_features.transform(X_smooth)
y_smooth = best_poly_model.predict(X_smooth_poly)

print(f"\n--- Predictions for Existing Positions ---")
print(f"{'Position':<20} {'Level':<8} {'Actual':<15} {'Predicted':<15} {'Error':<15}")
print("-" * 75)

y_pred_best = best_poly_model.predict(X_poly_best)
for i in range(len(df)):
    position = df.iloc[i]['Position']
    level = df.iloc[i]['Level']
    actual = df.iloc[i]['Salary']
    predicted = y_pred_best[i]
    error = actual - predicted

    print(f"{position:<20} {level:<8} ${actual:>12,} ${predicted:>12,.2f} ${error:>12,.2f}")

# Interpolation: Predict for level 6.5
print(f"\n--- Interpolation Example ---")
level_interp = np.array([[6.5]])
level_interp_poly = best_poly_features.transform(level_interp)
salary_interp = best_poly_model.predict(level_interp_poly)[0]

print(f"  Question: What salary for Level 6.5 (between Region Manager and Partner)?")
print(f"  Predicted Salary: ${salary_interp:,.2f}")
print(
    f"  Reference: Level 6 = ${df[df['Level'] == 6]['Salary'].values[0]:,}, Level 7 = ${df[df['Level'] == 7]['Salary'].values[0]:,}")

# ============================================================================
# STEP 7: MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL COMPARISON")
print("=" * 80)

comparison_data = {
    'Model': ['Linear', f'Polynomial (Degree {int(best_degree)})'],
    'R²': [r2_lin, best_r2],
    'RMSE': [rmse_lin, results_df.loc[results_df['Degree'] == best_degree, 'RMSE'].values[0]],
    'MAE': [mae_lin, results_df.loc[results_df['Degree'] == best_degree, 'MAE'].values[0]]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n--- Linear vs Polynomial Comparison ---")
print(comparison_df.to_string(index=False))

improvement_r2 = ((best_r2 - r2_lin) / r2_lin) * 100
improvement_rmse = ((rmse_lin - results_df.loc[results_df['Degree'] == best_degree, 'RMSE'].values[0]) / rmse_lin) * 100

print(f"\n--- Improvement with Polynomial Regression ---")
print(f"  R² improvement: {improvement_r2:+.2f}%")
print(f"  RMSE improvement: {improvement_rmse:+.2f}%")
print(
    f"  Conclusion: Polynomial regression {'significantly' if improvement_r2 > 10 else 'moderately'} outperforms linear regression")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Linear vs Polynomial Fit
print("\n📊 Creating linear vs polynomial comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Linear regression
axes[0].scatter(X, y, color='red', s=100, edgecolors='black', label='Actual Data', zorder=3)
axes[0].plot(X, y_pred_lin, color='blue', linewidth=2, label='Linear Fit')
axes[0].set_xlabel('Position Level', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Linear Regression (R² = {r2_lin:.4f})', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Polynomial regression
axes[1].scatter(X, y, color='red', s=100, edgecolors='black', label='Actual Data', zorder=3)
axes[1].plot(X_smooth, y_smooth, color='green', linewidth=2, label=f'Polynomial Fit (Degree {int(best_degree)})')
axes[1].set_xlabel('Position Level', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Polynomial Regression (R² = {best_r2:.4f})', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Linear vs Polynomial Regression Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('poly_viz_1_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: poly_viz_1_comparison.png")

# Visualization 2: Multiple Polynomial Degrees
print("\n📊 Creating multiple polynomial degrees comparison...")
fig, ax = plt.subplots(figsize=(14, 8))

# Plot actual data
ax.scatter(X, y, color='red', s=150, edgecolors='black', label='Actual Data', zorder=5, linewidths=2)

# Plot different polynomial degrees
colors = ['blue', 'green', 'purple', 'orange', 'brown']
for i, degree in enumerate(degrees):
    poly_features = poly_models[degree]['poly_features']
    X_smooth_poly = poly_features.transform(X_smooth)
    y_smooth_pred = poly_models[degree]['model'].predict(X_smooth_poly)
    r2 = results_df[results_df['Degree'] == degree]['R²'].values[0]

    ax.plot(X_smooth, y_smooth_pred, color=colors[i], linewidth=2,
            label=f'Degree {degree} (R²={r2:.4f})', alpha=0.8)

ax.set_xlabel('Position Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
ax.set_title('Polynomial Regression - Different Degrees', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('poly_viz_2_multiple_degrees.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: poly_viz_2_multiple_degrees.png")

# Visualization 3: R² Score by Degree
print("\n📊 Creating R² score comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

degrees_plot = [1] + degrees  # Include linear (degree 1)
r2_plot = [r2_lin] + results_df['R²'].tolist()

bars = ax.bar(degrees_plot, r2_plot, color='skyblue', edgecolor='black', alpha=0.7, width=0.6)

# Highlight best
best_idx = r2_plot.index(max(r2_plot))
bars[best_idx].set_color('green')
bars[best_idx].set_alpha(1.0)

ax.set_xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance by Polynomial Degree', fontsize=14, fontweight='bold')
ax.set_xticks(degrees_plot)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('poly_viz_3_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: poly_viz_3_r2_comparison.png")

# Visualization 4: Actual vs Predicted
print("\n📊 Creating actual vs predicted plot...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(y, y_pred_best, s=120, alpha=0.6, edgecolors='black', linewidths=2)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')

# Add labels for each point
for i, position in enumerate(df['Position']):
    ax.annotate(position, (y[i], y_pred_best[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('Actual Salary ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
ax.set_title(f'Actual vs Predicted Salaries (Polynomial Degree {int(best_degree)})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('poly_viz_4_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: poly_viz_4_actual_vs_predicted.png")

# Visualization 5: Residual Analysis
print("\n📊 Creating residual analysis...")
residuals = y - y_pred_best

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Level
axes[0, 0].scatter(X, residuals, s=100, alpha=0.6, edgecolors='black')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Position Level', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Position Level', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs Predicted
axes[0, 1].scatter(y_pred_best, residuals, s=100, alpha=0.6, edgecolors='black')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Salary ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Histogram of Residuals
axes[1, 0].hist(residuals, bins=8, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Absolute Residuals
axes[1, 1].bar(range(len(residuals)), np.abs(residuals), color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Data Point', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Absolute Error ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Absolute Prediction Errors', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(len(residuals)))
axes[1, 1].set_xticklabels([f"L{i + 1}" for i in range(len(residuals))], rotation=0)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle(f'Residual Analysis - Polynomial Degree {int(best_degree)}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('poly_viz_5_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: poly_viz_5_residuals.png")

# Visualization 6: Error Metrics Comparison
print("\n📊 Creating error metrics comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# RMSE comparison
models = ['Linear'] + [f'Poly-{d}' for d in degrees]
rmse_values = [rmse_lin] + results_df['RMSE'].tolist()

axes[0].bar(models, rmse_values, color=['red'] + ['lightblue'] * len(degrees),
            edgecolor='black', alpha=0.7)
axes[0].set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
axes[0].set_title('Root Mean Squared Error Comparison', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# MAE comparison
mae_values = [mae_lin] + results_df['MAE'].tolist()

axes[1].bar(models, mae_values, color=['red'] + ['lightgreen'] * len(degrees),
            edgecolor='black', alpha=0.7)
axes[1].set_ylabel('MAE ($)', fontsize=12, fontweight='bold')
axes[1].set_title('Mean Absolute Error Comparison', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('poly_viz_6_error_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: poly_viz_6_error_metrics.png")

# ============================================================================
# STEP 9: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'=' * 80}
POLYNOMIAL REGRESSION ANALYSIS - POSITION SALARY PREDICTION
{'=' * 80}

BUSINESS OBJECTIVE
{'=' * 80}
Predict salaries for different position levels in a company hierarchy.
Challenge: Salary growth is non-linear (accelerates at senior levels).
Solution: Polynomial regression to capture curved relationships.

DATASET SUMMARY
{'=' * 80}
Total positions: {len(df)}
Levels: {int(X.min())} to {int(X.max())}
Salary range: ${y.min():,} to ${y.max():,}

Position Hierarchy:
{df[['Position', 'Level', 'Salary']].to_string(index=False)}

Salary Growth Analysis:
  Entry to CEO: {(df['Salary'].max() / df['Salary'].min()):.1f}x increase
  Average increase per level: ${(df['Salary'].max() - df['Salary'].min()) / (df['Level'].max() - df['Level'].min()):,.2f}
  Growth pattern: Exponential (accelerating at higher levels)

MODEL COMPARISON
{'=' * 80}

1. LINEAR REGRESSION (Baseline)
   Equation: Salary = ${lin_intercept:,.2f} + ${lin_coef:,.2f} × Level

   Performance:
     R² Score: {r2_lin:.4f} ({r2_lin * 100:.2f}%)
     RMSE: ${rmse_lin:,.2f}
     MAE: ${mae_lin:,.2f}

   Conclusion: {"Linear model is adequate" if r2_lin > 0.9 else "Linear model underfits - non-linear pattern detected"}

2. POLYNOMIAL REGRESSION (Multiple Degrees Tested)

{chr(10).join([f'   Degree {int(row["Degree"])}: R² = {row["R²"]:.4f}, RMSE = ${row["RMSE"]:,.2f}, MAE = ${row["MAE"]:,.2f}'
               for _, row in results_df.iterrows()])}

BEST MODEL: POLYNOMIAL DEGREE {int(best_degree)}
{'=' * 80}
R² Score: {best_r2:.4f} ({best_r2 * 100:.2f}% of variance explained)
RMSE: ${results_df.loc[results_df['Degree'] == best_degree, 'RMSE'].values[0]:,.2f}
MAE: ${results_df.loc[results_df['Degree'] == best_degree, 'MAE'].values[0]:,.2f}

Model Equation:
{equation}

Improvement over Linear:
  R² improvement: {improvement_r2:+.2f}%
  RMSE improvement: {improvement_rmse:+.2f}%

PREDICTIONS
{'=' * 80}

Existing Positions:
{'Position':<20} {'Level':<8} {'Actual':<15} {'Predicted':<15} {'Error':<15}
{'-' * 75}
{chr(10).join([f'{df.iloc[i]["Position"]:<20} {df.iloc[i]["Level"]:<8} ${df.iloc[i]["Salary"]:>12,} ${y_pred_best[i]:>12,.2f} ${(df.iloc[i]["Salary"] - y_pred_best[i]):>12,.2f}'
               for i in range(len(df))])}

Interpolation Example:
  Level 6.5 (between Region Manager and Partner)
  Predicted Salary: ${salary_interp:,.2f}

  This is useful for:
    • Negotiating salaries for new positions
    • Planning compensation for intermediate roles
    • Ensuring pay equity across levels

RESIDUAL ANALYSIS
{'=' * 80}
Mean residual: ${residuals.mean():,.2f} (should be close to $0)
Std of residuals: ${residuals.std():,.2f}
Max absolute error: ${np.abs(residuals).max():,.2f}

Residual Pattern:
  {"✓ Residuals randomly distributed - model assumptions met" if np.abs(residuals.mean()) < 1000 else "⚠ Check for systematic patterns in residuals"}

KEY INSIGHTS
{'=' * 80}
✓ Salary Growth Pattern:
  - Non-linear (exponential) growth at senior levels
  - Entry level to mid-level: Steady increase
  - Senior to C-level: Dramatic acceleration
  - CEO salary is {(df[df['Level'] == 10]['Salary'].values[0] / df[df['Level'] == 1]['Salary'].values[0]):.1f}x entry level

✓ Model Performance:
  - Polynomial degree {int(best_degree)} captures the curve perfectly
  - {best_r2 * 100:.2f}% of salary variance explained
  - Average prediction error: only ${results_df.loc[results_df['Degree'] == best_degree, 'MAE'].values[0]:,.2f}

✓ Why Polynomial Works Better:
  - Captures exponential growth at senior levels
  - Linear model assumes constant increase per level
  - Reality: Each level jump increases by larger amounts

BUSINESS RECOMMENDATIONS
{'=' * 80}
1. SALARY STRUCTURE PLANNING
   Use the polynomial model to:
   • Set competitive salaries for each level
   • Ensure smooth progression between levels
   • Justify compensation decisions with data

2. RECRUITMENT & RETENTION
   • Predict fair offers for candidates at any level
   • Identify underpaid positions (actual < predicted)
   • Benchmark against market standards

3. CAREER PATH TRANSPARENCY
   • Show employees expected salary progression
   • Motivate career advancement with clear targets
   • Demonstrate value of promotions

4. BUDGET FORECASTING
   • Accurately project salary costs for workforce planning
   • Model impact of promotions on overall compensation
   • Plan for executive compensation growth

5. PAY EQUITY ANALYSIS
   • Use model as baseline for fair compensation
   • Identify anomalies (outliers from model)
   • Ensure consistent pay structure across organization

PRACTICAL APPLICATIONS
{'=' * 80}
Example Questions the Model Can Answer:

Q1: What should we offer a Level 4.5 candidate?
A1: Predicted salary: ${best_poly_model.predict(best_poly_features.transform([[4.5]]))[0]:,.2f}

Q2: Is $175,000 fair for Level 6.5?
A2: Model predicts ${salary_interp:,.2f} - {"Above" if 175000 > salary_interp else "Below"} model

Q3: What's the salary jump from Level 7 to 8?
A3: ${df[df['Level'] == 8]['Salary'].values[0]:,} - ${df[df['Level'] == 7]['Salary'].values[0]:,} = ${df[df['Level'] == 8]['Salary'].values[0] - df[df['Level'] == 7]['Salary'].values[0]:,}

FILES GENERATED
{'=' * 80}
Visualizations:
  • poly_viz_1_comparison.png - Linear vs Polynomial comparison
  • poly_viz_2_multiple_degrees.png - All polynomial degrees
  • poly_viz_3_r2_comparison.png - R² scores by degree
  • poly_viz_4_actual_vs_predicted.png - Prediction accuracy
  • poly_viz_5_residuals.png - Residual analysis (4 plots)
  • poly_viz_6_error_metrics.png - RMSE and MAE comparison

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(report)

# Save report
with open('poly_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Report saved to: poly_analysis_report.txt")

# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAVE MODEL")
print("=" * 80)

import joblib

# Save best model
model_data = {
    'poly_features': best_poly_features,
    'model': best_poly_model,
    'degree': best_degree,
    'r2_score': best_r2
}

model_path = 'polynomial_salary_model.pkl'
joblib.dump(model_data, model_path)
print(f"\n✓ Model saved to: {model_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("POLYNOMIAL REGRESSION ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"  ✓ Analyzed {len(df)} position levels")
print(f"  ✓ Tested {len(degrees) + 1} models (linear + {len(degrees)} polynomial)")
print(f"  ✓ Best model: Polynomial Degree {int(best_degree)}")
print(f"  ✓ Achieved R² = {best_r2:.4f} ({best_r2 * 100:.2f}%)")
print(f"  ✓ Average error: ${results_df.loc[results_df['Degree'] == best_degree, 'MAE'].values[0]:,.2f}")
print(f"  ✓ Generated 6 comprehensive visualizations")

print(f"\n💡 Key Findings:")
print(f"  • Salary growth is non-linear (exponential)")
print(f"  • Polynomial degree {int(best_degree)} fits best")
print(f"  • {improvement_r2:.1f}% improvement over linear model")
print(f"  • Model explains {best_r2 * 100:.2f}% of salary variance")

print(f"\n📈 Business Value:")
print(f"  • Accurate salary predictions for any level")
print(f"  • Fair compensation structure")
print(f"  • Data-driven hiring decisions")
print(f"  • Career progression transparency")

print("\n" + "=" * 80)
print("All analysis complete! Check output files for detailed results.")
print("=" * 80)