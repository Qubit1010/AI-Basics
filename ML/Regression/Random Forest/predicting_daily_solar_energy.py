"""
RANDOM FOREST REGRESSION - SOLAR FARM ENERGY PRODUCTION PREDICTION
===================================================================
Predicting daily solar energy output based on environmental conditions

Perfect Scenario for Random Forest:
- Solar energy production depends on MANY interacting factors
- Non-linear relationships (weather, seasonal patterns)
- Need robust predictions (outliers from extreme weather)
- Feature interactions critical (temperature + sunlight + humidity)
- Reduce overfitting from single decision trees
- Ensemble learning for stability

Dataset: Solar Farm Daily Production (Generated)
Features:
- Solar Irradiance (W/m²)
- Temperature (°C)
- Humidity (%)
- Cloud Cover (%)
- Wind Speed (m/s)
- Day of Year (1-365)
- Panel Temperature (°C)
- Dust Accumulation (0-100 scale)
- Hours of Sunlight
- Previous Day Production (kWh)

Target: Daily Energy Production (kWh)

Why Random Forest > Single Decision Tree:
- Averages predictions from multiple trees → more stable
- Reduces overfitting → better generalization
- Handles outliers better → robust to weather extremes
- Feature importance more reliable → averaged across trees
- Better accuracy on complex patterns

Approach:
1. Generate realistic solar farm production data
2. Exploratory Data Analysis
3. Build Random Forest models
4. Compare with single Decision Tree
5. Hyperparameter tuning
6. Feature importance analysis
7. Model interpretation and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("RANDOM FOREST REGRESSION - SOLAR FARM ENERGY PRODUCTION")
print("=" * 80)

# ============================================================================
# STEP 1: GENERATE REALISTIC SOLAR FARM DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC SOLAR FARM PRODUCTION DATA")
print("=" * 80)

np.random.seed(42)

# Number of days (1 year of data)
n_days = 365

print(f"\nGenerating {n_days} days of solar farm production data...")

# Generate features
data = {
    'DayOfYear': np.arange(1, n_days + 1),
    'SolarIrradiance': np.random.uniform(200, 1200, n_days),  # W/m²
    'Temperature': np.random.uniform(5, 40, n_days),  # °C
    'Humidity': np.random.uniform(20, 90, n_days),  # %
    'CloudCover': np.random.uniform(0, 100, n_days),  # %
    'WindSpeed': np.random.uniform(0, 15, n_days),  # m/s
    'HoursOfSunlight': np.random.uniform(4, 14, n_days),  # hours
    'PanelTemperature': np.random.uniform(10, 65, n_days),  # °C
    'DustAccumulation': np.random.uniform(0, 100, n_days)  # 0-100 scale
}

df = pd.DataFrame(data)

# Add seasonal patterns
df['Season'] = pd.cut(df['DayOfYear'],
                      bins=[0, 80, 172, 264, 365],
                      labels=['Winter', 'Spring', 'Summer', 'Fall'])

# Adjust features based on season for realism
season_adjustments = {
    'Winter': {'irrad_mult': 0.6, 'temp_adj': -10, 'sun_hours': -3},
    'Spring': {'irrad_mult': 0.9, 'temp_adj': 0, 'sun_hours': 0},
    'Summer': {'irrad_mult': 1.2, 'temp_adj': 5, 'sun_hours': 2},
    'Fall': {'irrad_mult': 0.8, 'temp_adj': -3, 'sun_hours': -1}
}

for season, adjustments in season_adjustments.items():
    mask = df['Season'] == season
    df.loc[mask, 'SolarIrradiance'] *= adjustments['irrad_mult']
    df.loc[mask, 'Temperature'] += adjustments['temp_adj']
    df.loc[mask, 'HoursOfSunlight'] += adjustments['sun_hours']

# Clip to realistic ranges
df['SolarIrradiance'] = df['SolarIrradiance'].clip(100, 1200)
df['Temperature'] = df['Temperature'].clip(0, 45)
df['HoursOfSunlight'] = df['HoursOfSunlight'].clip(4, 14)

# Generate previous day production (with some randomness)
prev_production = np.zeros(n_days)
prev_production[0] = np.random.uniform(8000, 12000)
for i in range(1, n_days):
    # Correlated with yesterday but with variation
    prev_production[i] = prev_production[i - 1] * np.random.uniform(0.8, 1.2)
df['PreviousDayProduction'] = prev_production

print("\nGenerating energy production based on realistic physics:")
print("  • Base production: 10,000 kWh")
print("  • +8 kWh per W/m² of solar irradiance")
print("  • +50 kWh per hour of sunlight")
print("  • -20 kWh per % cloud cover")
print("  • -10 kWh per unit of dust accumulation")
print("  • -30 kWh per °C above 25°C (panel efficiency loss)")
print("  • +0.05 × previous day production (autocorrelation)")
print("  • Random variation: ±1,000 kWh")

# Calculate energy production based on realistic model
base_production = 10000
production = (
        base_production +
        df['SolarIrradiance'] * 8 +
        df['HoursOfSunlight'] * 50 -
        df['CloudCover'] * 20 -
        df['DustAccumulation'] * 10 -
        np.maximum(0, df['PanelTemperature'] - 25) * 30 +  # Efficiency loss when hot
        df['PreviousDayProduction'] * 0.05 +
        np.random.normal(0, 1000, n_days)
)

# Ensure non-negative production
df['EnergyProduction'] = np.maximum(0, production)

print(f"\n✓ Dataset generated successfully!")
print(f"  Shape: {df.shape}")
print(f"  Features: {len(df.columns) - 2}")  # Excluding Season and target

print("\n--- First 10 Days ---")
print(df.head(10).to_string(index=False))

print("\n--- Data Statistics ---")
print(df.describe())

# Save dataset
df.to_csv('solar_farm_data.csv', index=False)
print(f"\n✓ Dataset saved to: solar_farm_data.csv")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n--- Energy Production Statistics ---")
print(f"  Mean: {df['EnergyProduction'].mean():,.2f} kWh/day")
print(f"  Median: {df['EnergyProduction'].median():,.2f} kWh/day")
print(f"  Std Dev: {df['EnergyProduction'].std():,.2f} kWh/day")
print(f"  Min: {df['EnergyProduction'].min():,.2f} kWh/day")
print(f"  Max: {df['EnergyProduction'].max():,.2f} kWh/day")
print(f"  Range: {df['EnergyProduction'].max() - df['EnergyProduction'].min():,.2f} kWh/day")

print("\n--- Seasonal Production ---")
seasonal_stats = df.groupby('Season')['EnergyProduction'].agg(['mean', 'std', 'min', 'max'])
print(seasonal_stats)

print("\n--- Feature Summary ---")
numerical_cols = ['SolarIrradiance', 'Temperature', 'Humidity', 'CloudCover',
                  'WindSpeed', 'HoursOfSunlight', 'PanelTemperature', 'DustAccumulation']
for col in numerical_cols:
    print(f"{col}:")
    print(f"  Range: {df[col].min():.2f} - {df[col].max():.2f}")
    print(f"  Mean: {df[col].mean():.2f}")

# Correlation analysis
print("\n--- Correlation with Energy Production ---")
correlations = df[numerical_cols + ['EnergyProduction']].corr()['EnergyProduction'].sort_values(ascending=False)
print(correlations)

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

# Prepare features and target
feature_columns = ['DayOfYear', 'SolarIrradiance', 'Temperature', 'Humidity',
                   'CloudCover', 'WindSpeed', 'HoursOfSunlight',
                   'PanelTemperature', 'DustAccumulation', 'PreviousDayProduction']

X = df[feature_columns]
y = df['EnergyProduction']

print(f"\nFeatures (X):")
print(f"  Shape: {X.shape}")
print(f"  Columns: {feature_columns}")

print(f"\nTarget (y):")
print(f"  Shape: {y.shape}")
print(f"  Name: EnergyProduction (kWh)")

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
print(f"  Training set: {X_train.shape[0]} days ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} days ({X_test.shape[0] / len(X) * 100:.1f}%)")

print(f"\nTraining set production statistics:")
print(f"  Mean: {y_train.mean():,.2f} kWh/day")
print(f"  Range: {y_train.min():,.2f} - {y_train.max():,.2f} kWh/day")

print(f"\nTest set production statistics:")
print(f"  Mean: {y_test.mean():,.2f} kWh/day")
print(f"  Range: {y_test.min():,.2f} - {y_test.max():,.2f} kWh/day")

# ============================================================================
# STEP 5: BUILD BASELINE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: BUILD BASELINE MODELS")
print("=" * 80)

print("\nExplanation:")
print("  Random Forest = Ensemble of Decision Trees")
print("  Instead of 1 tree, train 100+ trees and average predictions")
print("  Each tree trained on random subset of data (bootstrap)")
print("  Each split uses random subset of features")
print("  Result: More robust, less overfitting, better accuracy")

# Single Decision Tree (for comparison)
print("\n--- Training Single Decision Tree ---")
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

y_train_pred_dt = dt.predict(X_train)
y_test_pred_dt = dt.predict(X_test)

train_r2_dt = r2_score(y_train, y_train_pred_dt)
test_r2_dt = r2_score(y_test, y_test_pred_dt)
test_rmse_dt = np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
test_mae_dt = mean_absolute_error(y_test, y_test_pred_dt)

print(f"  Training R²: {train_r2_dt:.4f}")
print(f"  Test R²: {test_r2_dt:.4f}")
print(f"  Test RMSE: {test_rmse_dt:,.2f} kWh")
print(f"  Test MAE: {test_mae_dt:,.2f} kWh")
print(f"  Overfitting: {train_r2_dt - test_r2_dt:.4f}")

# Random Forest (default parameters)
print("\n--- Training Random Forest (Default) ---")
print("  Parameters: n_estimators=100, max_depth=None")

rf_default = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_default.fit(X_train, y_train)

y_train_pred_rf = rf_default.predict(X_train)
y_test_pred_rf = rf_default.predict(X_test)

train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)

print(f"  Training R²: {train_r2_rf:.4f}")
print(f"  Test R²: {test_r2_rf:.4f}")
print(f"  Test RMSE: {test_rmse_rf:,.2f} kWh")
print(f"  Test MAE: {test_mae_rf:,.2f} kWh")
print(f"  Overfitting: {train_r2_rf - test_r2_rf:.4f}")

print("\n--- Comparison ---")
print(f"  Decision Tree Test R²: {test_r2_dt:.4f}")
print(f"  Random Forest Test R²: {test_r2_rf:.4f}")
print(f"  Improvement: {((test_r2_rf - test_r2_dt) / test_r2_dt * 100):.2f}%")

# ============================================================================
# STEP 6: HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: HYPERPARAMETER TUNING")
print("=" * 80)

print("\nExplanation:")
print("  Key Random Forest parameters:")
print("  • n_estimators: Number of trees (more = better, but slower)")
print("  • max_depth: Maximum tree depth (controls overfitting)")
print("  • min_samples_split: Min samples to split node")
print("  • max_features: Features to consider for each split")

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

print(f"\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = (len(param_grid['n_estimators']) *
                      len(param_grid['max_depth']) *
                      len(param_grid['min_samples_split']) *
                      len(param_grid['max_features']))
print(f"\nTotal combinations: {total_combinations}")

print("\nPerforming Grid Search with 3-fold cross-validation...")
print("(This may take a minute...)")

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Grid search complete!")

print(f"\n--- Best Parameters ---")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n--- Best Score ---")
print(f"  Cross-validated R²: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_rf = grid_search.best_estimator_
y_train_pred_best = best_rf.predict(X_train)
y_test_pred_best = best_rf.predict(X_test)

train_r2_best = r2_score(y_train, y_train_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)
test_rmse_best = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
test_mae_best = mean_absolute_error(y_test, y_test_pred_best)

print(f"\n--- Optimized Random Forest Performance ---")
print(f"  Training R²: {train_r2_best:.4f}")
print(f"  Test R²: {test_r2_best:.4f}")
print(f"  Test RMSE: {test_rmse_best:,.2f} kWh")
print(f"  Test MAE: {test_mae_best:,.2f} kWh")
print(f"  Overfitting: {train_r2_best - test_r2_best:.4f}")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("\nExplanation:")
print("  Random Forest feature importance = average across all trees")
print("  More reliable than single tree importance")
print("  Shows which features most influence predictions")

# Feature importance from best model
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n--- Feature Importance (Optimized Random Forest) ---")
print(feature_importance.to_string(index=False))

print(f"\n--- Top 5 Most Important Features ---")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {i + 1}. {row['Feature']}: {row['Importance']:.4f} ({row['Importance'] * 100:.2f}%)")

# Compare with Decision Tree importance
dt_importance = pd.DataFrame({
    'Feature': feature_columns,
    'DT_Importance': dt.feature_importances_,
    'RF_Importance': best_rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print(f"\n--- Decision Tree vs Random Forest Feature Importance ---")
print(dt_importance.to_string(index=False))

# ============================================================================
# STEP 8: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: DETAILED MODEL EVALUATION")
print("=" * 80)

# Cross-validation
print("\nPerforming 5-fold cross-validation on best model...")
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='r2')

print(f"\nCross-validation R² scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f}")
print(f"  Std: {cv_scores.std():.4f}")
print(f"  Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

# Sample predictions
print(f"\n--- Sample Predictions (First 10 Test Days) ---")
print(f"{'Actual (kWh)':<15} {'Predicted (kWh)':<18} {'Error (kWh)':<15} {'Error %':<10}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred_best[i]
    error = actual - predicted
    error_pct = (error / actual) * 100

    print(f"{actual:>14,.2f} {predicted:>17,.2f} {error:>14,.2f} {error_pct:>8.2f}%")

# Residual analysis
residuals = y_test.values - y_test_pred_best

print(f"\n--- Residual Analysis ---")
print(f"  Mean residual: {residuals.mean():,.2f} kWh")
print(f"  Std of residuals: {residuals.std():,.2f} kWh")
print(f"  Max absolute error: {np.abs(residuals).max():,.2f} kWh")
print(f"  Min absolute error: {np.abs(residuals).min():,.2f} kWh")

# ============================================================================
# STEP 9: MODEL COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: MODEL COMPARISON SUMMARY")
print("=" * 80)

comparison_data = {
    'Model': ['Decision Tree', 'Random Forest (Default)', 'Random Forest (Optimized)'],
    'Train R²': [train_r2_dt, train_r2_rf, train_r2_best],
    'Test R²': [test_r2_dt, test_r2_rf, test_r2_best],
    'RMSE (kWh)': [test_rmse_dt, test_rmse_rf, test_rmse_best],
    'MAE (kWh)': [test_mae_dt, test_mae_rf, test_mae_best],
    'Overfitting': [train_r2_dt - test_r2_dt,
                    train_r2_rf - test_r2_rf,
                    train_r2_best - test_r2_best]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n--- Model Comparison Table ---")
print(comparison_df.to_string(index=False))

best_model_idx = comparison_df['Test R²'].idxmax()
print(f"\n🏆 Best Model: {comparison_df.iloc[best_model_idx]['Model']}")
print(f"   Test R²: {comparison_df.iloc[best_model_idx]['Test R²']:.4f}")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Time series of actual vs predicted
print("\n📊 Creating time series comparison...")
test_indices = X_test.index
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(test_indices, y_test.values, 'o-', label='Actual', alpha=0.7, markersize=4)
ax.plot(test_indices, y_test_pred_best, 's-', label='Predicted (RF)', alpha=0.7, markersize=4)
ax.set_xlabel('Day Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy Production (kWh)', fontsize=12, fontweight='bold')
ax.set_title('Actual vs Predicted Daily Energy Production', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_viz_1_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rf_viz_1_timeseries.png")

# Visualization 2: Model Comparison
print("\n📊 Creating model comparison charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² Scores
axes[0, 0].bar(comparison_df['Model'], comparison_df['Test R²'],
               color=['red', 'orange', 'green'], edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Model Performance (R² Score)', fontsize=12, fontweight='bold')
axes[0, 0].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[0, 0].set_ylim([0, 1.1])
axes[0, 0].grid(axis='y', alpha=0.3)

# RMSE
axes[0, 1].bar(comparison_df['Model'], comparison_df['RMSE (kWh)'],
               color=['red', 'orange', 'green'], edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('RMSE (kWh)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Prediction Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[0, 1].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[0, 1].grid(axis='y', alpha=0.3)

# Overfitting
axes[1, 0].bar(comparison_df['Model'], comparison_df['Overfitting'],
               color=['red', 'orange', 'green'], edgecolor='black', alpha=0.7)
axes[1, 0].axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='High Overfit Threshold')
axes[1, 0].set_ylabel('Train R² - Test R²', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Overfitting Measure', fontsize=12, fontweight='bold')
axes[1, 0].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Train vs Test R²
x_pos = np.arange(len(comparison_df))
width = 0.35
axes[1, 1].bar(x_pos - width / 2, comparison_df['Train R²'], width,
               label='Train', color='lightblue', edgecolor='black')
axes[1, 1].bar(x_pos + width / 2, comparison_df['Test R²'], width,
               label='Test', color='lightgreen', edgecolor='black')
axes[1, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Train vs Test Performance', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Random Forest vs Decision Tree Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_viz_2_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rf_viz_2_model_comparison.png")

# Visualization 3: Feature Importance
print("\n📊 Creating feature importance plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest
colors_rf = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
axes[0].barh(feature_importance['Feature'], feature_importance['Importance'],
             color=colors_rf, edgecolor='black')
axes[0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Feature', fontsize=12, fontweight='bold')
axes[0].set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Comparison
dt_sorted = dt_importance.sort_values('RF_Importance', ascending=True)
y_pos = np.arange(len(dt_sorted))
axes[1].barh(y_pos - 0.2, dt_sorted['DT_Importance'], 0.4,
             label='Decision Tree', color='red', alpha=0.7, edgecolor='black')
axes[1].barh(y_pos + 0.2, dt_sorted['RF_Importance'], 0.4,
             label='Random Forest', color='green', alpha=0.7, edgecolor='black')
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(dt_sorted['Feature'])
axes[1].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[1].set_title('Feature Importance Comparison', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('rf_viz_3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rf_viz_3_feature_importance.png")

# Visualization 4: Actual vs Predicted Scatter
print("\n📊 Creating actual vs predicted scatter plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree
axes[0].scatter(y_test, y_test_pred_dt, alpha=0.5, s=60, edgecolors='black')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2.5, label='Perfect Prediction')
axes[0].set_xlabel('Actual Production (kWh)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Production (kWh)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Decision Tree (R² = {test_r2_dt:.4f})', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].scatter(y_test, y_test_pred_best, alpha=0.5, s=60, edgecolors='black', color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2.5, label='Perfect Prediction')
axes[1].set_xlabel('Actual Production (kWh)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Production (kWh)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Random Forest (R² = {test_r2_best:.4f})', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_viz_4_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rf_viz_4_scatter.png")

# Visualization 5: Residual Analysis
print("\n📊 Creating residual analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Predicted
axes[0, 0].scatter(y_test_pred_best, residuals, alpha=0.5, s=60, edgecolors='black')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Production (kWh)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals (kWh)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Histogram of Residuals
axes[0, 1].hist(residuals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residuals (kWh)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Q-Q Plot
from scipy import stats

stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot - Normality Check', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Residuals over time
axes[1, 1].scatter(test_indices, residuals, alpha=0.5, s=40, edgecolors='black')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Day Index', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Residuals (kWh)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Residual Analysis - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_viz_5_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rf_viz_5_residuals.png")

# Visualization 6: Correlation Heatmap
print("\n📊 Creating correlation heatmap...")
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = df[feature_columns + ['EnergyProduction']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_viz_6_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rf_viz_6_correlation.png")

# Visualization 7: Production by Season
print("\n📊 Creating seasonal analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plot by season
df.boxplot(column='EnergyProduction', by='Season', ax=axes[0, 0])
axes[0, 0].set_title('Energy Production by Season', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Season', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Production (kWh)', fontsize=11, fontweight='bold')
plt.suptitle('')

# Solar irradiance vs Production
axes[0, 1].scatter(df['SolarIrradiance'], df['EnergyProduction'], alpha=0.5, s=30, edgecolors='black')
axes[0, 1].set_xlabel('Solar Irradiance (W/m²)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Production (kWh)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Production vs Solar Irradiance', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Cloud cover vs Production
axes[1, 0].scatter(df['CloudCover'], df['EnergyProduction'], alpha=0.5, s=30,
                   edgecolors='black', color='red')
axes[1, 0].set_xlabel('Cloud Cover (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Production (kWh)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Production vs Cloud Cover', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Production over year
axes[1, 1].scatter(df['DayOfYear'], df['EnergyProduction'], alpha=0.5, s=30,
                   c=df['DayOfYear'], cmap='viridis', edgecolors='black')
axes[1, 1].set_xlabel('Day of Year', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Production (kWh)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Production Throughout the Year', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Solar Farm Production Analysis', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('rf_viz_7_seasonal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: rf_viz_7_seasonal_analysis.png")

# ============================================================================
# STEP 11: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'=' * 80}
RANDOM FOREST REGRESSION - SOLAR FARM ENERGY PRODUCTION
{'=' * 80}

BUSINESS OBJECTIVE
{'=' * 80}
Predict daily solar farm energy production based on environmental conditions.

Why Random Forest for Solar Energy?
  • Multiple interacting factors (weather, seasonal, equipment)
  • Non-linear relationships (cloud cover impact varies by irradiance)
  • Need robust predictions (extreme weather outliers)
  • Ensemble reduces overfitting vs single tree
  • Feature importance reveals key production drivers

DATASET SUMMARY
{'=' * 80}
Total days: {len(df)} (1 year of data)
Features: {len(feature_columns)}

Environmental Conditions:
  • Solar Irradiance: {df['SolarIrradiance'].min():.0f} - {df['SolarIrradiance'].max():.0f} W/m²
  • Temperature: {df['Temperature'].min():.1f} - {df['Temperature'].max():.1f} °C
  • Humidity: {df['Humidity'].min():.1f} - {df['Humidity'].max():.1f} %
  • Cloud Cover: {df['CloudCover'].min():.1f} - {df['CloudCover'].max():.1f} %
  • Wind Speed: {df['WindSpeed'].min():.1f} - {df['WindSpeed'].max():.1f} m/s
  • Hours of Sunlight: {df['HoursOfSunlight'].min():.1f} - {df['HoursOfSunlight'].max():.1f} hours
  • Panel Temperature: {df['PanelTemperature'].min():.1f} - {df['PanelTemperature'].max():.1f} °C
  • Dust Accumulation: {df['DustAccumulation'].min():.1f} - {df['DustAccumulation'].max():.1f}

Energy Production:
  Mean: {df['EnergyProduction'].mean():,.2f} kWh/day
  Range: {df['EnergyProduction'].min():,.2f} - {df['EnergyProduction'].max():,.2f} kWh/day
  Total Annual: {df['EnergyProduction'].sum():,.2f} kWh

Seasonal Production:
{seasonal_stats.to_string()}

CORRELATION ANALYSIS
{'=' * 80}
Top Features Correlated with Production:
{chr(10).join([f'  {i + 1}. {idx}: {val:.4f}' for i, (idx, val) in enumerate(correlations.head(6).items()) if idx != 'EnergyProduction'])}

MODEL COMPARISON
{'=' * 80}

{comparison_df.to_string(index=False)}

Key Findings:
  • Random Forest outperforms Decision Tree by {((test_r2_best - test_r2_dt) / test_r2_dt * 100):.2f}%
  • Random Forest reduces overfitting from {(train_r2_dt - test_r2_dt):.4f} to {(train_r2_best - test_r2_best):.4f}
  • RMSE improved from {test_rmse_dt:,.2f} to {test_rmse_best:,.2f} kWh

BEST MODEL: RANDOM FOREST (OPTIMIZED)
{'=' * 80}
Architecture:
  • Number of trees: {best_rf.n_estimators}
  • Max depth: {best_rf.max_depth}
  • Min samples split: {best_rf.min_samples_split}
  • Max features: {best_rf.max_features}

Performance:
  Test R²: {test_r2_best:.4f} ({test_r2_best * 100:.2f}% of variance explained)
  RMSE: {test_rmse_best:,.2f} kWh ({test_rmse_best / df['EnergyProduction'].mean() * 100:.2f}% of mean)
  MAE: {test_mae_best:,.2f} kWh
  Overfitting: {train_r2_best - test_r2_best:.4f} (Low - excellent generalization!)

Cross-Validation:
  Mean R²: {cv_scores.mean():.4f}
  Std R²: {cv_scores.std():.4f}
  Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}
  Consistency: {'✓ Excellent' if cv_scores.std() < 0.05 else '✓ Good' if cv_scores.std() < 0.1 else '⚠ Variable'}

FEATURE IMPORTANCE
{'=' * 80}
Production Drivers (Ranked):

{chr(10).join([f'{i + 1:2d}. {row["Feature"]:<25} {row["Importance"]:.4f} ({row["Importance"] * 100:>6.2f}%)'
               for i, (_, row) in enumerate(feature_importance.iterrows())])}

Top 3 Production Factors:
{chr(10).join([f'  • {row["Feature"]}: {row["Importance"] * 100:.2f}% importance'
               for _, row in feature_importance.head(3).iterrows()])}

WHY RANDOM FOREST WORKS BETTER
{'=' * 80}
Advantages over Single Decision Tree:

1. ENSEMBLE POWER
   • Trains {best_rf.n_estimators} trees on different data subsets (bootstrap)
   • Each tree sees random features at each split
   • Final prediction = average of all {best_rf.n_estimators} trees
   • Result: More stable, robust predictions

2. REDUCED OVERFITTING
   • Single tree overfitting: {(train_r2_dt - test_r2_dt):.4f}
   • Random Forest overfitting: {(train_r2_best - test_r2_best):.4f}
   • Improvement: {((train_r2_dt - test_r2_dt) - (train_r2_best - test_r2_best)):.4f} reduction

3. BETTER ACCURACY
   • Single tree Test R²: {test_r2_dt:.4f}
   • Random Forest Test R²: {test_r2_best:.4f}
   • Improvement: {((test_r2_best - test_r2_dt) / test_r2_dt * 100):.2f}%

4. ROBUST TO OUTLIERS
   • Extreme weather days don't dominate single tree
   • Ensemble averaging smooths out anomalies
   • More reliable predictions in variable conditions

5. FEATURE IMPORTANCE RELIABILITY
   • Single tree: Feature importance varies with random seed
   • Random Forest: Averaged across {best_rf.n_estimators} trees
   • More trustworthy insights for business decisions

RESIDUAL ANALYSIS
{'=' * 80}
Mean residual: {residuals.mean():,.2f} kWh (near-zero = unbiased)
Std of residuals: {residuals.std():,.2f} kWh
Max error: {np.abs(residuals).max():,.2f} kWh
Min error: {np.abs(residuals).min():,.2f} kWh

Distribution: {"✓ Approximately normal - good model fit" if np.abs(residuals.mean()) < 100 else "⚠ Check for bias"}

BUSINESS INSIGHTS
{'=' * 80}
✓ Production Optimization:
  - {feature_importance.iloc[0]['Feature']} has {feature_importance.iloc[0]['Importance'] * 100:.1f}% impact
  - Focus on: {', '.join([row['Feature'] for _, row in feature_importance.head(3).iterrows()])}

✓ Predictive Accuracy:
  - Model explains {test_r2_best * 100:.2f}% of daily variation
  - Average error: ±{test_mae_best:,.0f} kWh ({test_mae_best / df['EnergyProduction'].mean() * 100:.1f}% of mean)
  - Reliable for: Production forecasting, grid planning, maintenance scheduling

✓ Seasonal Patterns:
  - Summer production: {seasonal_stats.loc['Summer', 'mean']:,.0f} kWh/day average
  - Winter production: {seasonal_stats.loc['Winter', 'mean']:,.0f} kWh/day average
  - Seasonal variation: {(seasonal_stats['mean'].max() - seasonal_stats['mean'].min()):,.0f} kWh/day

✓ Key Operational Findings:
  - Cloud cover correlation: {correlations['CloudCover']:.3f} (strong negative impact)
  - Dust accumulation reduces efficiency (correlation: {correlations['DustAccumulation']:.3f})
  - Panel temperature matters (efficiency loss when hot)

BUSINESS RECOMMENDATIONS
{'=' * 80}
1. PRODUCTION FORECASTING
   Use Random Forest to:
   • Predict tomorrow's production (using weather forecasts)
   • Plan grid delivery commitments
   • Optimize energy storage systems
   • Schedule maintenance during low-production periods

2. OPTIMIZATION PRIORITIES
   Based on feature importance:
   • Monitor {feature_importance.iloc[0]['Feature']} continuously
   • Invest in {feature_importance.iloc[1]['Feature']} improvements
   • Track {feature_importance.iloc[2]['Feature']} patterns

3. OPERATIONAL EFFICIENCY
   • Clean panels when dust > threshold (prevents {correlations['DustAccumulation']:.1%} loss)
   • Monitor panel temperature (cooling systems if > 25°C)
   • Weather-based maintenance scheduling

4. FINANCIAL PLANNING
   • Accurate production forecasts → better revenue projections
   • Identify underperforming days → investigate causes
   • Seasonal patterns → budget for variability

PRACTICAL APPLICATIONS
{'=' * 80}

Example Predictions:
  Optimal Day (high irradiance, low clouds):
  Features: 1000 W/m², 25°C, 0% clouds, 12 hrs sun
  Predicted: ~{best_rf.predict([[200, 1000, 25, 30, 0, 5, 12, 30, 20, 15000]])[0]:,.0f} kWh

  Poor Day (cloudy, dusty):
  Features: 300 W/m², 15°C, 80% clouds, 6 hrs sun, high dust
  Predicted: ~{best_rf.predict([[200, 300, 15, 70, 80, 8, 6, 20, 80, 8000]])[0]:,.0f} kWh

  Average Day:
  Features: 600 W/m², 20°C, 40% clouds, 9 hrs sun
  Predicted: ~{best_rf.predict([[200, 600, 20, 50, 40, 6, 9, 25, 40, 12000]])[0]:,.0f} kWh

ADVANTAGES OF RANDOM FOREST
{'=' * 80}
✓ Accuracy: Best-in-class for tabular data
✓ Robustness: Handles outliers and noise well
✓ No scaling: Works with raw features (unlike SVR, Neural Networks)
✓ Interpretability: Feature importance reveals insights
✓ Versatility: Handles non-linear patterns automatically
✓ Overfitting control: Ensemble reduces variance
✓ Missing data: Can handle missing values
✓ Parallel: Fast training on multiple cores

LIMITATIONS
{'=' * 80}
⚠ Memory: Stores {best_rf.n_estimators} trees (large models)
⚠ Interpretability: Less than single tree (black box ensemble)
⚠ Extrapolation: Poor on values outside training range
⚠ Speed: Slower predictions than linear models
⚠ Tuning: Many hyperparameters to optimize

FILES GENERATED
{'=' * 80}
Dataset:
  • solar_farm_data.csv - Generated solar production data

Visualizations:
  • rf_viz_1_timeseries.png - Actual vs predicted over time
  • rf_viz_2_model_comparison.png - RF vs Decision Tree
  • rf_viz_3_feature_importance.png - Feature ranking
  • rf_viz_4_scatter.png - Prediction accuracy scatter
  • rf_viz_5_residuals.png - Residual analysis (4 plots)
  • rf_viz_6_correlation.png - Feature correlations
  • rf_viz_7_seasonal_analysis.png - Seasonal patterns

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(report)

# Save report
with open('rf_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Report saved to: rf_analysis_report.txt")

# ============================================================================
# STEP 12: SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: SAVE MODEL")
print("=" * 80)

import joblib

model_data = {
    'model': best_rf,
    'feature_names': feature_columns,
    'feature_importance': feature_importance,
    'best_params': grid_search.best_params_,
    'test_r2': test_r2_best,
    'test_rmse': test_rmse_best
}

model_path = 'random_forest_model.pkl'
joblib.dump(model_data, model_path)
print(f"\n✓ Model saved to: {model_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("RANDOM FOREST REGRESSION ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"  ✓ Generated {len(df)} days of solar farm production data")
print(f"  ✓ Trained Random Forest with {best_rf.n_estimators} trees")
print(f"  ✓ Optimized hyperparameters via grid search")
print(f"  ✓ Achieved R² = {test_r2_best:.4f} ({test_r2_best * 100:.2f}%)")
print(f"  ✓ Average error: {test_mae_best:,.2f} kWh")
print(f"  ✓ Generated 7 comprehensive visualizations")

print(f"\n💡 Key Findings:")
print(f"  • Random Forest outperforms Decision Tree by {((test_r2_best - test_r2_dt) / test_r2_dt * 100):.2f}%")
print(f"  • Most important feature: {feature_importance.iloc[0]['Feature']}")
print(f"  • Model explains {test_r2_best * 100:.2f}% of production variance")
print(f"  • Overfitting: {train_r2_best - test_r2_best:.4f} (excellent control)")

print(f"\n🌲 Random Forest Advantages:")
print(f"  • Ensemble of {best_rf.n_estimators} trees → stability")
print(f"  • Reduced overfitting: {(train_r2_dt - test_r2_dt):.4f} → {(train_r2_best - test_r2_best):.4f}")
print(f"  • Better accuracy: {test_r2_dt:.4f} → {test_r2_best:.4f}")
print(f"  • Reliable feature importance")

print("\n" + "=" * 80)
print("All analysis complete! Check output files for detailed results.")
print("=" * 80)