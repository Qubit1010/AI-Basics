"""
SIMPLE LINEAR REGRESSION - HOUSE PRICE PREDICTION
==================================================
A beginner-friendly project using Linear Regression

Goal: Predict house prices based on house size
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

matplotlib.use('Agg')

print("=" * 70)
print("SIMPLE LINEAR REGRESSION - HOUSE PRICE PREDICTION")
print("=" * 70)

# ============================================================================
# 1. CREATE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: CREATE SAMPLE DATA")
print("=" * 70)

# Set random seed for reproducibility
np.random.seed(42)

# Generate house sizes (in square feet) - 100 houses
house_sizes = np.random.randint(500, 3500, 100)

# Generate prices based on size with some random variation
# Formula: Price = 100 * size + random noise
base_price = house_sizes * 100
noise = np.random.normal(0, 20000, 100)  # Random variation
house_prices = base_price + noise

# Create DataFrame
df = pd.DataFrame({
    'Size_SqFt': house_sizes,
    'Price_USD': house_prices
})

print(f"\n✓ Generated data for {len(df)} houses")
print(f"\nFirst 10 houses:")
print(df.head(10))

print(f"\nData Statistics:")
print(f"  Average house size: {df['Size_SqFt'].mean():.0f} sq ft")
print(f"  Average price: ${df['Price_USD'].mean():,.0f}")
print(f"  Cheapest house: ${df['Price_USD'].min():,.0f}")
print(f"  Most expensive: ${df['Price_USD'].max():,.0f}")

# Save data
df.to_csv('house_data.csv', index=False)
print(f"\n✓ Data saved to: house_data.csv")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: PREPARE DATA FOR TRAINING")
print("=" * 70)

# X = Features (house size), y = Target (price)
X = df[['Size_SqFt']].values  # Need 2D array for sklearn
y = df['Price_USD'].values

print(f"\nX (Features) shape: {X.shape}")
print(f"y (Target) shape: {y.shape}")

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} houses (80%)")
print(f"Test set: {len(X_test)} houses (20%)")

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: TRAIN LINEAR REGRESSION MODEL")
print("=" * 70)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n✓ Model trained successfully!")
print(f"\nModel Parameters:")
print(f"  Slope (coefficient): ${model.coef_[0]:.2f} per sq ft")
print(f"  Intercept: ${model.intercept_:,.2f}")
print(f"\nEquation: Price = {model.coef_[0]:.2f} × Size + {model.intercept_:,.2f}")

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: MAKE PREDICTIONS")
print("=" * 70)

# Predict on test set
y_pred = model.predict(X_test)

print(f"\nPredictions on test set:")
print(f"{'Actual Size':<15} {'Actual Price':<20} {'Predicted Price':<20} {'Difference':<15}")
print("-" * 70)

for i in range(min(10, len(X_test))):
    actual_size = X_test[i][0]
    actual_price = y_test[i]
    predicted_price = y_pred[i]
    difference = actual_price - predicted_price

    print(f"{actual_size:<15.0f} ${actual_price:<19,.0f} ${predicted_price:<19,.0f} ${difference:<14,.0f}")

# ============================================================================
# 5. EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: EVALUATE MODEL PERFORMANCE")
print("=" * 70)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance Metrics:")
print(f"  Mean Squared Error (MSE): ${mse:,.0f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse:,.0f}")
print(f"  R² Score: {r2:.4f} ({r2 * 100:.2f}%)")

print(f"\nInterpretation:")
print(f"  • The model explains {r2 * 100:.1f}% of the variance in house prices")
print(f"  • Average prediction error: ${rmse:,.0f}")

if r2 > 0.9:
    print(f"  • Excellent model! R² > 0.9")
elif r2 > 0.7:
    print(f"  • Good model! R² > 0.7")
else:
    print(f"  • Model needs improvement")

# ============================================================================
# 6. PREDICT NEW HOUSES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: PREDICT PRICES FOR NEW HOUSES")
print("=" * 70)

# New houses to predict
new_houses = np.array([[1000], [1500], [2000], [2500], [3000]])

predictions = model.predict(new_houses)

print(f"\n🏠 Price Predictions for New Houses:")
print(f"{'House Size (sq ft)':<25} {'Predicted Price':<20}")
print("-" * 45)
for size, price in zip(new_houses, predictions):
    print(f"{size[0]:<25.0f} ${price:>18,.0f}")

# ============================================================================
# 7. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: CREATE VISUALIZATIONS")
print("=" * 70)

# Visualization 1: Scatter plot with regression line
print("\n📊 Creating scatter plot with regression line...")

plt.figure(figsize=(12, 6))

# Plot training data
plt.scatter(X_train, y_train, color='blue', alpha=0.5, s=50,
            label='Training data', edgecolors='black')

# Plot test data
plt.scatter(X_test, y_test, color='green', alpha=0.7, s=50,
            label='Test data', edgecolors='black')

# Plot regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=3,
         label=f'Regression Line: y = {model.coef_[0]:.2f}x + {model.intercept_:.0f}')

plt.xlabel('House Size (Square Feet)', fontsize=12, fontweight='bold')
plt.ylabel('Price (USD)', fontsize=12, fontweight='bold')
plt.title('House Price Prediction - Linear Regression', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Format y-axis to show currency
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x / 1000:.0f}K'))

plt.tight_layout()
plt.savefig('regression_viz_1_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: regression_viz_1_scatter.png")

# Visualization 2: Actual vs Predicted
print("\n📊 Creating actual vs predicted plot...")

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='purple', alpha=0.6, s=80, edgecolors='black')

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
         label='Perfect Prediction')

plt.xlabel('Actual Price (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Price (USD)', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Format axes
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x / 1000:.0f}K'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x / 1000:.0f}K'))

# Add R² score to plot
plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('regression_viz_2_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: regression_viz_2_actual_vs_predicted.png")

# Visualization 3: Residuals (Errors)
print("\n📊 Creating residuals plot...")

residuals = y_test - y_pred

plt.figure(figsize=(12, 5))

# Residuals scatter
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, color='orange', alpha=0.6, s=60, edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
plt.title('Residual Plot', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Residuals histogram
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Residuals (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Frequency', fontsize=11, fontweight='bold')
plt.title('Distribution of Residuals', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('regression_viz_3_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: regression_viz_3_residuals.png")

# ============================================================================
# 8. SAVE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: SAVE MODEL")
print("=" * 70)

import joblib

# Save the trained model
model_filename = 'linear_regression_model.pkl'
joblib.dump(model, model_filename)
print(f"\n✓ Model saved to: {model_filename}")

# Test loading the model
loaded_model = joblib.load(model_filename)
test_prediction = loaded_model.predict([[2000]])
print(f"\n✓ Model loaded successfully")
print(f"  Test prediction for 2000 sq ft house: ${test_prediction[0]:,.0f}")

# ============================================================================
# 9. GENERATE REPORT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: GENERATE REPORT")
print("=" * 70)

report = f"""
{'=' * 70}
LINEAR REGRESSION - HOUSE PRICE PREDICTION REPORT
{'=' * 70}

PROJECT OVERVIEW
{'=' * 70}
Objective: Predict house prices based on house size
Algorithm: Linear Regression
Dataset: {len(df)} houses

DATA SUMMARY
{'=' * 70}
Total Houses: {len(df)}
Training Set: {len(X_train)} houses (80%)
Test Set: {len(X_test)} houses (20%)

House Size Range: {df['Size_SqFt'].min():.0f} - {df['Size_SqFt'].max():.0f} sq ft
Price Range: ${df['Price_USD'].min():,.0f} - ${df['Price_USD'].max():,.0f}

MODEL EQUATION
{'=' * 70}
Price = {model.coef_[0]:.2f} × Size + {model.intercept_:,.2f}

Interpretation:
  • Every additional square foot increases price by ${model.coef_[0]:.2f}
  • Base price (y-intercept): ${model.intercept_:,.0f}

MODEL PERFORMANCE
{'=' * 70}
R² Score: {r2:.4f} ({r2 * 100:.2f}%)
  → The model explains {r2 * 100:.1f}% of variance in house prices

Root Mean Squared Error (RMSE): ${rmse:,.0f}
  → Average prediction error is ${rmse:,.0f}

Mean Squared Error (MSE): ${mse:,.0f}

SAMPLE PREDICTIONS
{'=' * 70}
House Size (sq ft)    Predicted Price
------------------------------------------
     1,000            ${model.predict([[1000]])[0]:>18,.0f}
     1,500            ${model.predict([[1500]])[0]:>18,.0f}
     2,000            ${model.predict([[2000]])[0]:>18,.0f}
     2,500            ${model.predict([[2500]])[0]:>18,.0f}
     3,000            ${model.predict([[3000]])[0]:>18,.0f}

KEY INSIGHTS
{'=' * 70}
• Strong linear relationship between house size and price
• Model performs {'excellently' if r2 > 0.9 else 'well' if r2 > 0.7 else 'moderately'}
• Predictions are within ${rmse:,.0f} of actual prices on average
• Larger houses command proportionally higher prices

CONCLUSION
{'=' * 70}
The linear regression model successfully predicts house prices
based on size with an R² score of {r2:.4f}. This model can be
used to estimate prices for houses in the same market.

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

print(report)

# Save report
with open('regression_report.txt', 'w') as f:
    f.write(report)

print("✓ Report saved to: regression_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PROJECT COMPLETE!")
print("=" * 70)

print(f"\n📊 What We Did:")
print(f"  1. ✓ Created sample data (100 houses)")
print(f"  2. ✓ Split data into training (80%) and testing (20%)")
print(f"  3. ✓ Trained Linear Regression model")
print(f"  4. ✓ Made predictions on test data")
print(f"  5. ✓ Evaluated model performance (R² = {r2:.4f})")
print(f"  6. ✓ Predicted prices for new houses")
print(f"  7. ✓ Created 3 visualizations")
print(f"  8. ✓ Saved trained model")
print(f"  9. ✓ Generated comprehensive report")

print(f"\n📁 Files Generated:")
files = [
    "house_data.csv - Original house data",
    "linear_regression_model.pkl - Trained model",
    "regression_report.txt - Project report",
    "regression_viz_1_scatter.png - Scatter plot with regression line",
    "regression_viz_2_actual_vs_predicted.png - Prediction accuracy",
    "regression_viz_3_residuals.png - Error analysis"
]
for file in files:
    print(f"  • {file}")

print(f"\n🎯 Model Performance:")
print(f"  • R² Score: {r2:.4f} ({r2 * 100:.2f}%)")
print(f"  • Average Error: ${rmse:,.0f}")
print(f"  • Price per sq ft: ${model.coef_[0]:.2f}")

print("\n" + "=" * 70)
print("Linear Regression project completed successfully!")
print("=" * 70)