import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 70)
print("MULTIPLE LINEAR REGRESSION FROM SCRATCH")
print("Predicting House Prices Based on Multiple Features")
print("=" * 70)

# Create sample dataset with multiple features
np.random.seed(42)
n_samples = 100

# Generate features for houses
# Feature 1: Square Footage (1000-4000 sq ft)
square_feet = np.random.randint(1000, 4000, n_samples)

# Feature 2: Number of Bedrooms (2-5)
bedrooms = np.random.randint(2, 6, n_samples)

# Feature 3: Number of Bathrooms (1-4)
bathrooms = np.random.randint(1, 5, n_samples)

# Feature 4: Age of House in years (0-50)
age = np.random.randint(0, 51, n_samples)

# Generate house prices with realistic relationships and noise
# Price = 100*sqft + 20000*bedrooms + 15000*bathrooms - 2000*age + base_price + noise
base_price = 50000
price = (100 * square_feet +
         20000 * bedrooms +
         15000 * bathrooms -
         2000 * age +
         base_price +
         np.random.normal(0, 15000, n_samples))  # Add random noise

# Ensure prices are positive
price = np.maximum(price, 50000)

# Create a DataFrame for better organization
data = pd.DataFrame({
    'Square_Feet': square_feet,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age': age,
    'Price': price
})

print("\nDataset Preview:")
print(data.head(10))
print("\nDataset Statistics:")
print(data.describe())

# Prepare data for Multiple Linear Regression
# X = feature matrix (independent variables)
# y = target vector (dependent variable - Price)

X = data[['Square_Feet', 'Bedrooms', 'Bathrooms', 'Age']].values
y = data['Price'].values

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")


# ============================================================================
# MULTIPLE LINEAR REGRESSION IMPLEMENTATION FROM SCRATCH
# ============================================================================

class MultipleLinearRegression:
    """
    Multiple Linear Regression implementation from scratch using Normal Equation

    Model: y = β0 + β1*x1 + β2*x2 + ... + βn*xn

    Normal Equation: β = (X^T * X)^(-1) * X^T * y
    where X is the design matrix with an added column of ones for the intercept
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """
        Train the model using the Normal Equation

        Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        """
        # Add a column of ones to X for the intercept term
        n_samples = X.shape[0]
        X_with_intercept = np.c_[np.ones(n_samples), X]

        # Calculate coefficients using Normal Equation: β = (X^T * X)^(-1) * X^T * y
        # This is the closed-form solution for linear regression
        X_transpose = X_with_intercept.T
        beta = np.linalg.inv(X_transpose @ X_with_intercept) @ X_transpose @ y

        # Extract intercept and coefficients
        self.intercept = beta[0]
        self.coefficients = beta[1:]

        return self

    def predict(self, X):
        """
        Make predictions using the trained model

        Parameters:
        X: Feature matrix (n_samples, n_features)

        Returns:
        Predicted values
        """
        return self.intercept + X @ self.coefficients

    def score(self, X, y):
        """
        Calculate R-squared score

        Parameters:
        X: Feature matrix
        y: True target values

        Returns:
        R-squared value
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2


# ============================================================================
# TRAIN THE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING THE MODEL")
print("=" * 70)

# Create and train the model
model = MultipleLinearRegression()
model.fit(X, y)

# Display model parameters
print("\nModel Parameters:")
print(f"Intercept (β0): ${model.intercept:,.2f}")
print("\nCoefficients:")
feature_names = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Age']
for name, coef in zip(feature_names, model.coefficients):
    print(f"  {name:15s} (β): ${coef:>10,.2f}")

print("\nRegression Equation:")
equation = f"Price = ${model.intercept:,.2f}"
for name, coef in zip(feature_names, model.coefficients):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} ${abs(coef):,.2f} * {name}"
print(equation)

# ============================================================================
# EVALUATE THE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Make predictions on training data
y_pred = model.predict(X)

# Calculate metrics
r_squared = model.score(X, y)
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))

print(f"\nR-squared (R²): {r_squared:.4f}")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")

# Calculate residuals
residuals = y - y_pred

# ============================================================================
# MAKE PREDICTIONS ON NEW DATA
# ============================================================================

print("\n" + "=" * 70)
print("PREDICTIONS ON NEW HOUSES")
print("=" * 70)

# Create new house examples
new_houses = np.array([
    [2000, 3, 2, 5],  # 2000 sqft, 3 bed, 2 bath, 5 years old
    [3500, 4, 3, 1],  # 3500 sqft, 4 bed, 3 bath, 1 year old
    [1500, 2, 1, 30],  # 1500 sqft, 2 bed, 1 bath, 30 years old
    [2800, 4, 2, 10],  # 2800 sqft, 4 bed, 2 bath, 10 years old
])

new_predictions = model.predict(new_houses)

print("\nNew House Predictions:")
for i, (house, pred) in enumerate(zip(new_houses, new_predictions), 1):
    print(f"\nHouse {i}:")
    print(f"  Square Feet: {house[0]:,} | Bedrooms: {int(house[1])} | "
          f"Bathrooms: {int(house[2])} | Age: {int(house[3])} years")
    print(f"  Predicted Price: ${pred:,.2f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Create a comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# Plot 1: Actual vs Predicted Prices
ax1 = plt.subplot(2, 3, 1)
sns.scatterplot(x=y, y=y_pred, alpha=0.6, s=80, edgecolor='white', linewidth=1)
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Price ($)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax1.set_title('Actual vs Predicted Prices', fontsize=13, fontweight='bold', pad=15)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add R² annotation
textstr = f'$R^2 = {r_squared:.4f}$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# Plot 2: Residual Plot
ax2 = plt.subplot(2, 3, 2)
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, s=80,
                color='#C73E1D', edgecolor='white', linewidth=1)
ax2.axhline(y=0, color='#2E86AB', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of Residuals
ax3 = plt.subplot(2, 3, 3)
sns.histplot(residuals, kde=True, bins=30, color='#2E86AB', alpha=0.7)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax3.set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Distribution of Residuals', fontsize=13, fontweight='bold', pad=15)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Price vs Square Feet
ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(X[:, 0], y, c=X[:, 1], cmap='viridis',
                      alpha=0.6, s=80, edgecolor='white', linewidth=1)
ax4.set_xlabel('Square Feet', fontsize=11, fontweight='bold')
ax4.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax4.set_title('Price vs Square Feet (colored by bedrooms)',
              fontsize=13, fontweight='bold', pad=15)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Bedrooms', fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Feature Importance (Coefficient Magnitudes)
ax5 = plt.subplot(2, 3, 5)
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coefficients
})
colors = ['#2E86AB' if c > 0 else '#C73E1D' for c in coef_df['Coefficient']]
bars = ax5.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors,
                alpha=0.7, edgecolor='white', linewidth=2)
ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax5.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
ax5.set_title('Feature Coefficients', fontsize=13, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Correlation Heatmap
ax6 = plt.subplot(2, 3, 6)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax6)
ax6.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('multiple_regression_analysis.png', dpi=300, bbox_inches='tight')
print("Main visualization saved as 'multiple_regression_analysis.png'")

# ============================================================================
# 3D VISUALIZATION (Price vs 2 Features)
# ============================================================================

# Create 3D plot: Price vs Square Feet and Bedrooms
fig_3d = plt.figure(figsize=(14, 10))

# 3D Scatter Plot
ax_3d = fig_3d.add_subplot(111, projection='3d')
scatter_3d = ax_3d.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis',
                           s=60, alpha=0.6, edgecolor='white', linewidth=0.5)

ax_3d.set_xlabel('Square Feet', fontsize=11, fontweight='bold', labelpad=10)
ax_3d.set_ylabel('Bedrooms', fontsize=11, fontweight='bold', labelpad=10)
ax_3d.set_zlabel('Price ($)', fontsize=11, fontweight='bold', labelpad=10)
ax_3d.set_title('3D View: House Price vs Square Feet and Bedrooms',
                fontsize=14, fontweight='bold', pad=20)

cbar_3d = plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.5, aspect=5)
cbar_3d.set_label('Price ($)', fontsize=10, fontweight='bold')

# Add grid
ax_3d.grid(True, alpha=0.3)

plt.savefig('3d_price_visualization.png', dpi=300, bbox_inches='tight')
print("3D visualization saved as '3d_price_visualization.png'")

# ============================================================================
# PAIRPLOT FOR ALL FEATURES
# ============================================================================

print("\nCreating pairplot for feature relationships...")
pairplot_fig = sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50},
                            corner=False, height=2.5)
pairplot_fig.fig.suptitle('Pairwise Relationships Between Features',
                          fontsize=14, fontweight='bold', y=1.01)
pairplot_fig.savefig('feature_pairplot.png', dpi=300, bbox_inches='tight')
print("Pairplot saved as 'feature_pairplot.png'")


# ============================================================================
# FUNCTION FOR INTERACTIVE PREDICTIONS
# ============================================================================

def predict_house_price(square_feet, bedrooms, bathrooms, age):
    """
    Predict house price based on input features

    Parameters:
    square_feet: Square footage of the house
    bedrooms: Number of bedrooms
    bathrooms: Number of bathrooms
    age: Age of the house in years

    Returns:
    Predicted price
    """
    features = np.array([[square_feet, bedrooms, bathrooms, age]])
    predicted_price = model.predict(features)[0]
    return predicted_price


print("\n" + "=" * 70)
print("INTERACTIVE PREDICTION FUNCTION")
print("=" * 70)
print("\nYou can use the predict_house_price() function to predict prices:")
print("Example: predict_house_price(2500, 3, 2, 10)")
print(f"Result: ${predict_house_price(2500, 3, 2, 10):,.2f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\nGenerated Files:")
print("  1. multiple_regression_analysis.png - Comprehensive analysis plots")
print("  2. 3d_price_visualization.png - 3D visualization of price relationships")
print("  3. feature_pairplot.png - Pairwise feature relationships")
print("=" * 70)