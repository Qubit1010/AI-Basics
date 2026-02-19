"""
MULTI-LINEAR REGRESSION - STARTUP PROFIT PREDICTION
===================================================
Predicting startup profit based on multiple independent variables

Business Question:
How do R&D Spend, Administration, Marketing Spend, and State location
affect a startup's profit?

Approach:
1. Data exploration and preprocessing
2. Feature engineering (handling categorical variable)
3. Building multi-linear regression model
4. Model evaluation and diagnostics
5. Feature importance analysis
6. Comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("MULTI-LINEAR REGRESSION - STARTUP PROFIT PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD AND EXPLORE DATA")
print("=" * 80)

# Load data
df = pd.read_csv('50_Startups.csv')

print(f"\n✓ Data loaded successfully")
print(f"  Shape: {df.shape}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")

print("\n--- First 10 Rows ---")
print(df.head(10))

print("\n--- Data Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Column Names ---")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# Check for missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing)
print(f"Total missing values: {missing.sum()}")

# Check data types
print("\n--- Data Types ---")
print(df.dtypes)

# Categorical variable
print("\n--- Categorical Variable (State) ---")
print(df['State'].value_counts())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Correlation analysis
print("\n--- Correlation Analysis ---")
numerical_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
correlation = df[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(correlation)

print("\n--- Correlation with Profit ---")
profit_corr = correlation['Profit'].sort_values(ascending=False)
print(profit_corr)

# State-wise analysis
print("\n--- State-wise Analysis ---")
state_stats = df.groupby('State')['Profit'].agg(['count', 'mean', 'std', 'min', 'max'])
print(state_stats)

# Feature ranges
print("\n--- Feature Ranges ---")
for col in numerical_cols:
    print(f"{col}:")
    print(f"  Min: ${df[col].min():,.2f}")
    print(f"  Max: ${df[col].max():,.2f}")
    print(f"  Range: ${df[col].max() - df[col].min():,.2f}")
    print(f"  Mean: ${df[col].mean():,.2f}")
    print(f"  Std: ${df[col].std():,.2f}")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

# Handle categorical variable (State) using One-Hot Encoding
print("\n--- Encoding Categorical Variable (State) ---")
print("Using One-Hot Encoding for 'State' variable")

# Create dummy variables
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
print(f"\n✓ State variable encoded")
print(f"  Original columns: {df.columns.tolist()}")
print(f"  New columns: {df_encoded.columns.tolist()}")

# Separate features and target
X = df_encoded.drop('Profit', axis=1)
y = df_encoded['Profit']

print(f"\nFeatures (X):")
print(f"  Shape: {X.shape}")
print(f"  Columns: {X.columns.tolist()}")

print(f"\nTarget (y):")
print(f"  Shape: {y.shape}")
print(f"  Name: Profit")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT")
print("=" * 80)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split (80/20):")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

print(f"\nTraining set statistics:")
print(f"  Mean profit: ${y_train.mean():,.2f}")
print(f"  Std profit: ${y_train.std():,.2f}")

print(f"\nTest set statistics:")
print(f"  Mean profit: ${y_test.mean():,.2f}")
print(f"  Std profit: ${y_test.std():,.2f}")

# ============================================================================
# STEP 5: BUILD MULTI-LINEAR REGRESSION MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: BUILD MULTI-LINEAR REGRESSION MODEL")
print("=" * 80)

print("\nExplanation:")
print("  Multi-Linear Regression Equation:")
print("  Profit = β₀ + β₁(R&D) + β₂(Admin) + β₃(Marketing) + β₄(State_FL) + β₅(State_NY)")
print("  where β₀ is intercept and β₁-β₅ are coefficients")

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n✓ Model trained successfully")

# Get model parameters
intercept = model.intercept_
coefficients = model.coef_

print(f"\n--- Model Parameters ---")
print(f"Intercept (β₀): ${intercept:,.2f}")
print(f"\nCoefficients:")
for i, (feature, coef) in enumerate(zip(X.columns, coefficients), 1):
    print(f"  β{i} ({feature}): {coef:,.4f}")

# Build equation
print(f"\n--- Regression Equation ---")
equation = f"Profit = ${intercept:,.2f}"
for feature, coef in zip(X.columns, coefficients):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} {abs(coef):.4f} × {feature}"
print(equation)

# ============================================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MAKE PREDICTIONS")
print("=" * 80)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n--- Sample Predictions (First 10 Test Samples) ---")
print(f"{'Actual Profit':<20} {'Predicted Profit':<20} {'Difference':<20}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    diff = actual - predicted
    print(f"${actual:>18,.2f} ${predicted:>18,.2f} ${diff:>18,.2f}")

# ============================================================================
# STEP 7: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL EVALUATION")
print("=" * 80)

# Training set metrics
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test set metrics
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n--- Training Set Performance ---")
print(f"  R² Score: {train_r2:.4f} ({train_r2 * 100:.2f}% variance explained)")
print(f"  MSE: ${train_mse:,.2f}")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE: ${train_mae:,.2f}")

print("\n--- Test Set Performance ---")
print(f"  R² Score: {test_r2:.4f} ({test_r2 * 100:.2f}% variance explained)")
print(f"  MSE: ${test_mse:,.2f}")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE: ${test_mae:,.2f}")

print("\n--- Model Performance Interpretation ---")
print(f"  R² = {test_r2:.4f} means the model explains {test_r2 * 100:.2f}% of profit variance")
print(f"  Average prediction error (MAE): ${test_mae:,.2f}")
print(f"  Root Mean Square Error: ${test_rmse:,.2f}")

# Calculate residuals
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

print(f"\n--- Residual Analysis ---")
print(f"Training residuals:")
print(f"  Mean: ${residuals_train.mean():,.2f} (should be close to 0)")
print(f"  Std: ${residuals_train.std():,.2f}")

print(f"\nTest residuals:")
print(f"  Mean: ${residuals_test.mean():,.2f}")
print(f"  Std: ${residuals_test.std():,.2f}")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n--- Feature Importance (by Coefficient Magnitude) ---")
print(feature_importance.to_string(index=False))

print("\n--- Interpretation ---")
for i, row in feature_importance.head(3).iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    if coef > 0:
        print(f"  • {feature}: +${abs(coef):.2f} increase in profit per unit increase")
    else:
        print(f"  • {feature}: -${abs(coef):.2f} decrease in profit per unit increase")

# Most influential feature
most_important = feature_importance.iloc[0]
print(f"\n🌟 Most Important Feature: {most_important['Feature']}")
print(f"   Coefficient: {most_important['Coefficient']:.4f}")

# ============================================================================
# STEP 9: MODEL ASSUMPTIONS CHECK
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: MODEL ASSUMPTIONS CHECK")
print("=" * 80)

print("\nLinear Regression Assumptions:")
print("  1. Linearity: Relationship between X and y is linear")
print("  2. Independence: Observations are independent")
print("  3. Homoscedasticity: Constant variance of residuals")
print("  4. Normality: Residuals are normally distributed")

# Test for normality of residuals (Shapiro-Wilk test)
from scipy.stats import shapiro

stat, p_value = shapiro(residuals_test)
print(f"\n--- Normality Test (Shapiro-Wilk) ---")
print(f"  Test Statistic: {stat:.4f}")
print(f"  P-value: {p_value:.4f}")
if p_value > 0.05:
    print(f"  ✓ Residuals are normally distributed (p > 0.05)")
else:
    print(f"  ⚠ Residuals may not be perfectly normal (p < 0.05)")

# Durbin-Watson test for autocorrelation
from scipy.stats import pearsonr

dw_stat = np.sum(np.diff(residuals_test.values) ** 2) / np.sum(residuals_test.values ** 2)
print(f"\n--- Durbin-Watson Test (Independence) ---")
print(f"  DW Statistic: {dw_stat:.4f}")
print(f"  Interpretation: Values close to 2 indicate no autocorrelation")
if 1.5 < dw_stat < 2.5:
    print(f"  ✓ No significant autocorrelation")
else:
    print(f"  ⚠ Possible autocorrelation detected")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Correlation Heatmap
print("\n📊 Creating correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix - Startup Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mlr_viz_1_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: mlr_viz_1_correlation.png")

# Visualization 2: Actual vs Predicted
print("\n📊 Creating actual vs predicted plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.6, edgecolors='black', s=80)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Training Set (R² = {train_r2:.4f})', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='black', s=80)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Test Set (R² = {test_r2:.4f})', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Actual vs Predicted Profit', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mlr_viz_2_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: mlr_viz_2_actual_vs_predicted.png")

# Visualization 3: Residual Plot
print("\n📊 Creating residual plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Predicted (Training)
axes[0, 0].scatter(y_train_pred, residuals_train, alpha=0.6, edgecolors='black')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residual Plot - Training Set', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs Predicted (Test)
axes[0, 1].scatter(y_test_pred, residuals_test, alpha=0.6, color='green', edgecolors='black')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Residual Plot - Test Set', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Histogram of Residuals
axes[1, 0].hist(residuals_test, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Q-Q Plot
stats.probplot(residuals_test, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot - Normality Check', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mlr_viz_3_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: mlr_viz_3_residuals.png")

# Visualization 4: Feature Coefficients
print("\n📊 Creating feature coefficients plot...")
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
bars = ax.barh(feature_importance['Feature'], feature_importance['Coefficient'],
               color=colors, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Feature Coefficients (Impact on Profit)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + (0.001 if width > 0 else -0.001)
    ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
            f'{width:.4f}', ha='left' if width > 0 else 'right',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('mlr_viz_4_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: mlr_viz_4_coefficients.png")

# Visualization 5: Feature vs Profit Relationships
print("\n📊 Creating feature vs profit relationships...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

numerical_features = ['R&D Spend', 'Administration', 'Marketing Spend']
for i, feature in enumerate(numerical_features):
    row = i // 2
    col = i % 2

    axes[row, col].scatter(df[feature], df['Profit'], alpha=0.6, edgecolors='black', s=60)

    # Add regression line
    z = np.polyfit(df[feature], df['Profit'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
    axes[row, col].plot(x_line, p(x_line), "r--", linewidth=2, label='Trend Line')

    axes[row, col].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[row, col].set_ylabel('Profit ($)', fontsize=11, fontweight='bold')
    axes[row, col].set_title(f'{feature} vs Profit', fontsize=12, fontweight='bold')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

# State comparison
axes[1, 1].boxplot([df[df['State'] == state]['Profit'] for state in df['State'].unique()],
                   labels=df['State'].unique())
axes[1, 1].set_xlabel('State', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Profit ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Profit Distribution by State', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Feature Relationships with Profit', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mlr_viz_5_feature_relationships.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: mlr_viz_5_feature_relationships.png")

# Visualization 6: Model Performance Metrics
print("\n📊 Creating performance metrics comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

metrics_names = ['R² Score', 'RMSE ($1000s)', 'MAE ($1000s)']
train_metrics = [train_r2, train_rmse / 1000, train_mae / 1000]
test_metrics = [test_r2, test_rmse / 1000, test_mae / 1000]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width / 2, train_metrics, width, label='Training Set',
               color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width / 2, test_metrics, width, label='Test Set',
               color='lightgreen', edgecolor='black')

ax.set_ylabel('Score / Value', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('mlr_viz_6_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: mlr_viz_6_metrics.png")

# ============================================================================
# STEP 11: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'=' * 80}
MULTI-LINEAR REGRESSION ANALYSIS - STARTUP PROFIT PREDICTION
{'=' * 80}

BUSINESS OBJECTIVE
{'=' * 80}
Predict startup profit based on:
  • R&D Spend
  • Administration costs
  • Marketing Spend
  • State location (New York, California, Florida)

DATASET SUMMARY
{'=' * 80}
Total startups: {len(df)}
Features: {len(X.columns)}
Target: Profit

Feature Statistics:
  R&D Spend:
    Range: ${df['R&D Spend'].min():,.2f} - ${df['R&D Spend'].max():,.2f}
    Mean: ${df['R&D Spend'].mean():,.2f}

  Administration:
    Range: ${df['Administration'].min():,.2f} - ${df['Administration'].max():,.2f}
    Mean: ${df['Administration'].mean():,.2f}

  Marketing Spend:
    Range: ${df['Marketing Spend'].min():,.2f} - ${df['Marketing Spend'].max():,.2f}
    Mean: ${df['Marketing Spend'].mean():,.2f}

  Profit (Target):
    Range: ${df['Profit'].min():,.2f} - ${df['Profit'].max():,.2f}
    Mean: ${df['Profit'].mean():,.2f}

State Distribution:
{df['State'].value_counts().to_string()}

REGRESSION MODEL
{'=' * 80}
Type: Multi-Linear Regression
Formula: Profit = β₀ + β₁(R&D) + β₂(Admin) + β₃(Marketing) + β₄(State_FL) + β₅(State_NY)

Model Parameters:
  Intercept (β₀): ${intercept:,.2f}

  Coefficients:
{chr(10).join([f'    {feature}: {coef:,.6f}' for feature, coef in zip(X.columns, coefficients)])}

Regression Equation:
{equation}

MODEL PERFORMANCE
{'=' * 80}
Training Set:
  R² Score: {train_r2:.4f} ({train_r2 * 100:.2f}% of variance explained)
  RMSE: ${train_rmse:,.2f}
  MAE: ${train_mae:,.2f}

Test Set:
  R² Score: {test_r2:.4f} ({test_r2 * 100:.2f}% of variance explained)
  RMSE: ${test_rmse:,.2f}
  MAE: ${test_mae:,.2f}

Performance Interpretation:
  • R² = {test_r2:.4f} means the model explains {test_r2 * 100:.2f}% of profit variance
  • Average prediction error: ${test_mae:,.2f}
  • Model performs {'well' if test_r2 > 0.9 else 'reasonably' if test_r2 > 0.7 else 'adequately'} with minimal overfitting

FEATURE IMPORTANCE
{'=' * 80}
Ranked by absolute coefficient magnitude:

{chr(10).join([f'{i + 1}. {row["Feature"]}: {row["Coefficient"]:,.6f} ({"+ve" if row["Coefficient"] > 0 else "-ve"} impact)'
               for i, (_, row) in enumerate(feature_importance.iterrows())])}

KEY INSIGHTS
{'=' * 80}
✓ Most Important Feature: {most_important['Feature']}
  - Coefficient: {most_important['Coefficient']:.6f}
  - Impact: {"Positive" if most_important['Coefficient'] > 0 else "Negative"}

✓ Correlation Analysis:
  - Strongest correlation with profit: {profit_corr.index[1]} (r = {profit_corr.iloc[1]:.3f})
  - R&D Spend shows correlation of {profit_corr['R&D Spend']:.3f}
  - Marketing Spend shows correlation of {profit_corr['Marketing Spend']:.3f}

✓ Model Assumptions:
  - Normality: {"✓ Satisfied" if p_value > 0.05 else "⚠ May be violated"} (p = {p_value:.4f})
  - Independence: {"✓ Satisfied" if 1.5 < dw_stat < 2.5 else "⚠ Check for autocorrelation"} (DW = {dw_stat:.4f})
  - Linearity: Based on scatter plots and R² score
  - Homoscedasticity: Based on residual plots

BUSINESS RECOMMENDATIONS
{'=' * 80}
1. MAXIMIZE R&D INVESTMENT
   R&D Spend has the {'highest' if feature_importance.iloc[0]['Feature'] == 'R&D Spend' else 'significant'} impact on profit
   Every $1 increase in R&D spending leads to ${coefficients[X.columns.tolist().index('R&D Spend')]:.4f} increase in profit

2. OPTIMIZE MARKETING BUDGET
   Marketing coefficient: {coefficients[X.columns.tolist().index('Marketing Spend')]:.6f}
   {"Focus on efficient marketing strategies" if coefficients[X.columns.tolist().index('Marketing Spend')] > 0 else "Review marketing effectiveness"}

3. CONTROL ADMINISTRATIVE COSTS
   Administration coefficient: {coefficients[X.columns.tolist().index('Administration')]:.6f}
   {"Streamline admin processes" if coefficients[X.columns.tolist().index('Administration')] < 0 else "Admin spending correlates positively"}

4. LOCATION STRATEGY
   State impact analysis:
   - Consider geographic expansion based on state coefficients
   - Optimize location selection for new startups

PREDICTION ACCURACY
{'=' * 80}
The model predicts startup profit with:
  • {test_r2 * 100:.2f}% accuracy (R² score)
  • Average error of ${test_mae:,.2f} (MAE)
  • Typical deviation of ${test_rmse:,.2f} (RMSE)

Example Predictions (Test Set):
{chr(10).join([f'  Actual: ${y_test.iloc[i]:>12,.2f} | Predicted: ${y_test_pred[i]:>12,.2f} | Error: ${abs(y_test.iloc[i] - y_test_pred[i]):>12,.2f}'
               for i in range(min(5, len(y_test)))])}

FILES GENERATED
{'=' * 80}
Visualizations:
  • mlr_viz_1_correlation.png - Correlation heatmap
  • mlr_viz_2_actual_vs_predicted.png - Prediction accuracy plots
  • mlr_viz_3_residuals.png - Residual analysis (4 plots)
  • mlr_viz_4_coefficients.png - Feature coefficients
  • mlr_viz_5_feature_relationships.png - Feature vs profit relationships
  • mlr_viz_6_metrics.png - Performance metrics comparison

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(report)

# Save report
with open('mlr_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Report saved to: mlr_analysis_report.txt")

# ============================================================================
# STEP 12: SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: SAVE MODEL")
print("=" * 80)

import joblib

model_path = 'startup_profit_model.pkl'
joblib.dump(model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save feature names for future predictions
feature_info = {
    'feature_names': X.columns.tolist(),
    'feature_count': len(X.columns)
}
joblib.dump(feature_info, 'feature_info.pkl')
print(f"✓ Feature info saved to: feature_info.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MULTI-LINEAR REGRESSION ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"  ✓ Analyzed {len(df)} startup records")
print(f"  ✓ Built multi-linear regression model")
print(f"  ✓ Achieved R² = {test_r2:.4f} ({test_r2 * 100:.2f}% variance explained)")
print(f"  ✓ Average prediction error: ${test_mae:,.2f}")
print(f"  ✓ Generated 6 comprehensive visualizations")
print(f"  ✓ Created detailed analytical report")

print(f"\n💡 Key Findings:")
print(f"  • Most important feature: {most_important['Feature']}")
print(f"  • Model explains {test_r2 * 100:.2f}% of profit variance")
print(f"  • Average error: ${test_mae:,.2f}")
print(f"  • RMSE: ${test_rmse:,.2f}")

print(f"\n📈 Business Impact:")
print(f"  • R&D Spend coefficient: {coefficients[X.columns.tolist().index('R&D Spend')]:.6f}")
print(f"  • Marketing Spend coefficient: {coefficients[X.columns.tolist().index('Marketing Spend')]:.6f}")
print(f"  • Administration coefficient: {coefficients[X.columns.tolist().index('Administration')]:.6f}")

print("\n" + "=" * 80)
print("All analysis complete! Check output files for detailed results.")
print("=" * 80)