import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Sample dataset: SAT scores and corresponding GPAs
# SAT scores (out of 1600)
sat_scores = np.array([1200, 1350, 1100, 1450, 1250, 1500, 1050, 1400, 1300, 1150,
                       1380, 1280, 1420, 1180, 1320, 1460, 1220, 1340, 1480, 1260])

# GPA (out of 4.0)
gpa = np.array([2.8, 3.2, 2.5, 3.6, 2.9, 3.8, 2.3, 3.5, 3.1, 2.6,
                3.3, 3.0, 3.4, 2.7, 3.2, 3.7, 2.85, 3.25, 3.75, 2.95])

# Step 1: Calculate the mean of SAT scores and GPA
mean_sat = np.mean(sat_scores)
mean_gpa = np.mean(gpa)

print("=" * 50)
print("LINEAR REGRESSION FROM SCRATCH")
print("=" * 50)
print(f"\nDataset Statistics:")
print(f"Number of samples: {len(sat_scores)}")
print(f"Mean SAT Score: {mean_sat:.2f}")
print(f"Mean GPA: {mean_gpa:.2f}")

# Step 2: Calculate the slope (m) and intercept (b)
# Formula for slope: m = Σ((x - x_mean) * (y - y_mean)) / Σ((x - x_mean)^2)
# Formula for intercept: b = y_mean - m * x_mean

numerator = np.sum((sat_scores - mean_sat) * (gpa - mean_gpa))
denominator = np.sum((sat_scores - mean_sat) ** 2)

slope = numerator / denominator
intercept = mean_gpa - slope * mean_sat

print(f"\nModel Parameters:")
print(f"Slope (m): {slope:.6f}")
print(f"Intercept (b): {intercept:.6f}")
print(f"\nRegression Equation: GPA = {slope:.6f} * SAT + {intercept:.6f}")

# Step 3: Make predictions on training data
predictions = slope * sat_scores + intercept

# Step 4: Calculate R-squared (coefficient of determination)
ss_total = np.sum((gpa - mean_gpa) ** 2)
ss_residual = np.sum((gpa - predictions) ** 2)
r_squared = 1 - (ss_residual / ss_total)

print(f"\nModel Performance:")
print(f"R-squared: {r_squared:.4f}")

# Step 5: Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = np.mean((gpa - predictions) ** 2)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Step 6: Make predictions for new SAT scores
print("\n" + "=" * 50)
print("PREDICTIONS FOR NEW SAT SCORES")
print("=" * 50)

new_sat_scores = np.array([1000, 1200, 1400, 1600])
new_predictions = slope * new_sat_scores + intercept

for sat, predicted_gpa in zip(new_sat_scores, new_predictions):
    print(f"SAT Score: {sat} → Predicted GPA: {predicted_gpa:.2f}")

# Step 7: Visualize the results with seaborn
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Regression Line with Data Points ---
ax1 = axes[0]

# Scatter plot of actual data with seaborn
sns.scatterplot(x=sat_scores, y=gpa, s=150, alpha=0.7,
                color='#2E86AB', edgecolor='white', linewidth=1.5,
                label='Actual Data', ax=ax1)

# Plot the regression line
x_line = np.linspace(sat_scores.min() - 50, sat_scores.max() + 50, 100)
y_line = slope * x_line + intercept
ax1.plot(x_line, y_line, color='#A23B72', linewidth=3,
         label='Regression Line', linestyle='--')

# Mark predictions for new data
sns.scatterplot(x=new_sat_scores, y=new_predictions, s=180,
                color='#F18F01', marker='D', edgecolor='white',
                linewidth=2, label='New Predictions', ax=ax1, zorder=5)

ax1.set_xlabel('SAT Score', fontsize=13, fontweight='bold')
ax1.set_ylabel('GPA', fontsize=13, fontweight='bold')
ax1.set_title('Linear Regression: SAT Score vs GPA',
              fontsize=15, fontweight='bold', pad=20)
ax1.legend(fontsize=11, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add R-squared annotation
textstr = f'$R^2 = {r_squared:.4f}$\n$RMSE = {rmse:.4f}$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

# --- Plot 2: Residual Plot ---
ax2 = axes[1]

# Calculate residuals
residuals = gpa - predictions

# Create residual plot
sns.scatterplot(x=predictions, y=residuals, s=150, alpha=0.7,
                color='#C73E1D', edgecolor='white', linewidth=1.5, ax=ax2)

# Add horizontal line at y=0
ax2.axhline(y=0, color='#2E86AB', linestyle='--', linewidth=2, label='Zero Line')

ax2.set_xlabel('Predicted GPA', fontsize=13, fontweight='bold')
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=13, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=15, fontweight='bold', pad=20)
ax2.legend(fontsize=11, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# Save the plot
plt.savefig('gpa_prediction.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 50)
print("Plot saved as 'gpa_prediction.png'")
print("=" * 50)

plt.show()

# Additional seaborn visualization: Joint plot
print("\nCreating additional seaborn joint plot...")
joint_fig = sns.jointplot(x=sat_scores, y=gpa, kind='reg', height=8,
                           color='#2E86AB', scatter_kws={'s': 100, 'alpha': 0.6},
                           line_kws={'color': '#A23B72', 'linewidth': 3})
joint_fig.fig.suptitle('SAT Score vs GPA - Joint Distribution',
                        fontsize=14, fontweight='bold', y=1.02)
joint_fig.savefig('gpa_joint_plot.png', dpi=300, bbox_inches='tight')
print("Joint plot saved as 'gpa_joint_plot.png'")
plt.close()

# Function to predict GPA for any SAT score
def predict_gpa(sat_score):
    """
    Predict GPA based on SAT score using our trained model
    """
    return slope * sat_score + intercept

# Interactive prediction
print("\n" + "=" * 50)
print("INTERACTIVE PREDICTION")
print("=" * 50)
print("\nYou can use the predict_gpa() function to predict GPA for any SAT score.")
print("Example: predict_gpa(1300)")