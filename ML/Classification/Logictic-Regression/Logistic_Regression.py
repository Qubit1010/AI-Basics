"""
LOGISTIC REGRESSION FROM SCRATCH
=================================
Implementing Logistic Regression using plain Python (no sklearn)

Goal: Classify students as PASS or FAIL based on study hours
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

print("=" * 70)
print("LOGISTIC REGRESSION FROM SCRATCH")
print("=" * 70)

# ============================================================================
# 1. CREATE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: CREATE SAMPLE DATA")
print("=" * 70)

# Set random seed
np.random.seed(42)

# Generate study hours (0-10 hours)
study_hours = np.random.uniform(0, 10, 100)

# Generate pass/fail (0 = Fail, 1 = Pass)
# Students who study more are more likely to pass
pass_probability = 1 / (1 + np.exp(-2 * (study_hours - 5)))  # Sigmoid curve
passed = (np.random.random(100) < pass_probability).astype(int)

print(f"\n✓ Generated data for {len(study_hours)} students")
print(f"\nFirst 10 students:")
print(f"{'Study Hours':<15} {'Result':<10}")
print("-" * 25)
for i in range(10):
    result = "PASS" if passed[i] == 1 else "FAIL"
    print(f"{study_hours[i]:<15.2f} {result:<10}")

print(f"\nData Summary:")
print(f"  Total students: {len(study_hours)}")
print(f"  Passed: {np.sum(passed)} students ({np.sum(passed) / len(passed) * 100:.1f}%)")
print(f"  Failed: {len(passed) - np.sum(passed)} students ({(len(passed) - np.sum(passed)) / len(passed) * 100:.1f}%)")

# ============================================================================
# 2. LOGISTIC REGRESSION CLASS (FROM SCRATCH)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: IMPLEMENT LOGISTIC REGRESSION")
print("=" * 70)


class LogisticRegressionScratch:
    """
    Logistic Regression implementation from scratch
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the model

        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        """
        Sigmoid activation function
        σ(z) = 1 / (1 + e^(-z))

        Maps any value to a value between 0 and 1
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the model using gradient descent

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        """
        # Get number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: Calculate predictions
            # Linear combination: z = w*x + b
            linear_model = np.dot(X, self.weights) + self.bias

            # Apply sigmoid function to get probabilities
            y_predicted = self.sigmoid(linear_model)

            # Calculate loss (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(y_predicted + 1e-15) +
                            (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.losses.append(loss)

            # Backward pass: Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i + 1}/{self.n_iterations}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """
        Predict probabilities for samples

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data

        Returns:
        --------
        probabilities : array-like, shape (n_samples,)
            Predicted probabilities
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1)

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        threshold : float
            Classification threshold (default 0.5)

        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


print("\n✓ Logistic Regression class implemented!")
print("\nKey Components:")
print("  • Sigmoid function: Maps values to 0-1 range")
print("  • Binary Cross-Entropy loss: Measures prediction error")
print("  • Gradient Descent: Optimizes weights and bias")
print("  • Prediction: Converts probabilities to class labels")

# ============================================================================
# 3. PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: PREPARE DATA FOR TRAINING")
print("=" * 70)

# Reshape X to be 2D array (n_samples, n_features)
X = study_hours.reshape(-1, 1)
y = passed

# Split data: 80% training, 20% testing
train_size = int(0.8 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print(f"\nData shape:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

print(f"\nTraining set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")

# ============================================================================
# 4. TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: TRAIN THE MODEL")
print("=" * 70)

# Create and train model
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
print(f"\nTraining with:")
print(f"  Learning rate: {model.learning_rate}")
print(f"  Iterations: {model.n_iterations}")
print(f"\nTraining progress:")

model.fit(X_train, y_train)

print(f"\n✓ Training complete!")
print(f"\nLearned Parameters:")
print(f"  Weight: {model.weights[0]:.4f}")
print(f"  Bias: {model.bias:.4f}")
print(f"\nModel equation:")
print(f"  P(Pass) = sigmoid({model.weights[0]:.4f} × study_hours + {model.bias:.4f})")

# ============================================================================
# 5. MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: MAKE PREDICTIONS")
print("=" * 70)

# Predict on test set
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

print(f"\nPredictions on test set:")
print(f"{'Study Hours':<15} {'Actual':<10} {'Predicted':<12} {'Probability':<15}")
print("-" * 52)

for i in range(min(15, len(X_test))):
    actual = "PASS" if y_test[i] == 1 else "FAIL"
    predicted = "PASS" if y_pred[i] == 1 else "FAIL"
    probability = y_pred_proba[i]
    print(f"{X_test[i][0]:<15.2f} {actual:<10} {predicted:<12} {probability:<15.4f}")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: EVALUATE MODEL PERFORMANCE")
print("=" * 70)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)

# Calculate confusion matrix manually
true_positives = np.sum((y_pred == 1) & (y_test == 1))
true_negatives = np.sum((y_pred == 0) & (y_test == 0))
false_positives = np.sum((y_pred == 1) & (y_test == 0))
false_negatives = np.sum((y_pred == 0) & (y_test == 1))

# Calculate metrics
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1_score:.4f}")

print(f"\nConfusion Matrix:")
print(f"                Predicted FAIL  Predicted PASS")
print(f"  Actual FAIL        {true_negatives:^6}          {false_positives:^6}")
print(f"  Actual PASS        {false_negatives:^6}          {true_positives:^6}")

print(f"\nInterpretation:")
print(f"  • Correctly predicted FAIL: {true_negatives}")
print(f"  • Correctly predicted PASS: {true_positives}")
print(f"  • Incorrectly predicted PASS: {false_positives}")
print(f"  • Incorrectly predicted FAIL: {false_negatives}")

# ============================================================================
# 7. PREDICT FOR NEW STUDENTS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: PREDICT FOR NEW STUDENTS")
print("=" * 70)

# Create new data points
new_students = np.array([[1], [3], [5], [7], [9]])

# Predict
new_proba = model.predict_proba(new_students)
new_pred = model.predict(new_students)

print(f"\n🎓 Predictions for New Students:")
print(f"{'Study Hours':<15} {'Prediction':<15} {'Pass Probability':<20}")
print("-" * 50)

for hours, pred, proba in zip(new_students, new_pred, new_proba):
    result = "PASS ✓" if pred == 1 else "FAIL ✗"
    print(f"{hours[0]:<15.0f} {result:<15} {proba:<20.4f} ({proba * 100:.1f}%)")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: CREATE VISUALIZATIONS")
print("=" * 70)

# Visualization 1: Data points and sigmoid curve
print("\n📊 Creating sigmoid curve visualization...")

plt.figure(figsize=(12, 6))

# Plot actual data points
fail_mask = y == 0
pass_mask = y == 1

plt.scatter(X[fail_mask], y[fail_mask], color='red', s=100,
            alpha=0.6, label='Failed', marker='x', linewidths=2)
plt.scatter(X[pass_mask], y[pass_mask], color='green', s=100,
            alpha=0.6, label='Passed', marker='o')

# Plot sigmoid curve
X_range = np.linspace(0, 10, 300).reshape(-1, 1)
y_range = model.predict_proba(X_range)

plt.plot(X_range, y_range, color='blue', linewidth=3,
         label='Logistic Regression Curve')

# Add decision boundary (probability = 0.5)
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
            alpha=0.5, label='Decision Boundary (50%)')

plt.xlabel('Study Hours', fontsize=12, fontweight='bold')
plt.ylabel('Probability of Passing', fontsize=12, fontweight='bold')
plt.title('Logistic Regression: Student Pass/Fail Prediction',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('logistic_viz_1_sigmoid_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: logistic_viz_1_sigmoid_curve.png")

# Visualization 2: Training loss over iterations
print("\n📊 Creating loss curve...")

plt.figure(figsize=(10, 6))

plt.plot(model.losses, color='blue', linewidth=2)
plt.xlabel('Iteration', fontsize=12, fontweight='bold')
plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12, fontweight='bold')
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_viz_2_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: logistic_viz_2_loss_curve.png")

# Visualization 3: Confusion Matrix
print("\n📊 Creating confusion matrix...")

fig, ax = plt.subplots(figsize=(8, 6))

confusion_matrix = np.array([[true_negatives, false_positives],
                             [false_negatives, true_positives]])

im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')

# Add labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted FAIL', 'Predicted PASS'])
ax.set_yticklabels(['Actual FAIL', 'Actual PASS'])

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, confusion_matrix[i, j],
                       ha="center", va="center",
                       color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black",
                       fontsize=20, fontweight='bold')

ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('logistic_viz_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: logistic_viz_3_confusion_matrix.png")

# Visualization 4: Probability predictions
print("\n📊 Creating probability predictions chart...")

fig, ax = plt.subplots(figsize=(12, 6))

# Sort by study hours for better visualization
sorted_indices = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_indices]
y_test_sorted = y_test[sorted_indices]
y_pred_proba_sorted = model.predict_proba(X_test_sorted)

# Plot bars
colors = ['green' if p >= 0.5 else 'red' for p in y_pred_proba_sorted]
bars = ax.bar(range(len(y_pred_proba_sorted)), y_pred_proba_sorted,
              color=colors, alpha=0.6, edgecolor='black')

# Add threshold line
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
           label='Decision Threshold (50%)')

ax.set_xlabel('Student (sorted by study hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Probability of Passing', fontsize=12, fontweight='bold')
ax.set_title('Predicted Probabilities for Test Students', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('logistic_viz_4_probabilities.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: logistic_viz_4_probabilities.png")

# ============================================================================
# 9. SAVE REPORT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: GENERATE REPORT")
print("=" * 70)

report = f"""
{'=' * 70}
LOGISTIC REGRESSION FROM SCRATCH - PROJECT REPORT
{'=' * 70}

PROJECT OVERVIEW
{'=' * 70}
Objective: Classify students as PASS or FAIL based on study hours
Algorithm: Logistic Regression (implemented from scratch)
Dataset: {len(study_hours)} students

IMPLEMENTATION DETAILS
{'=' * 70}
• Sigmoid Activation Function
• Binary Cross-Entropy Loss
• Gradient Descent Optimization
• Learning Rate: {model.learning_rate}
• Training Iterations: {model.n_iterations}

DATA SUMMARY
{'=' * 70}
Total Students: {len(study_hours)}
  Passed: {np.sum(passed)} ({np.sum(passed) / len(passed) * 100:.1f}%)
  Failed: {len(passed) - np.sum(passed)} ({(len(passed) - np.sum(passed)) / len(passed) * 100:.1f}%)

Training Set: {len(X_train)} students (80%)
Test Set: {len(X_test)} students (20%)

LEARNED PARAMETERS
{'=' * 70}
Weight: {model.weights[0]:.4f}
Bias: {model.bias:.4f}

Model Equation:
  P(Pass) = sigmoid({model.weights[0]:.4f} × study_hours + {model.bias:.4f})

Interpretation:
  • Each additional hour of study increases the log-odds of passing
  • Positive weight means more study hours → higher pass probability

MODEL PERFORMANCE
{'=' * 70}
Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1_score:.4f}

Confusion Matrix:
                Predicted FAIL  Predicted PASS
  Actual FAIL        {true_negatives:^6}          {false_positives:^6}
  Actual PASS        {false_negatives:^6}          {true_positives:^6}

Results:
  • Correct predictions: {true_positives + true_negatives}/{len(y_test)}
  • Incorrect predictions: {false_positives + false_negatives}/{len(y_test)}

SAMPLE PREDICTIONS
{'=' * 70}
Study Hours    Prediction    Pass Probability
---------------------------------------------
     1.0          FAIL         {model.predict_proba(np.array([[1]]))[0] * 100:>6.1f}%
     3.0          FAIL         {model.predict_proba(np.array([[3]]))[0] * 100:>6.1f}%
     5.0          {'PASS' if model.predict(np.array([[5]]))[0] == 1 else 'FAIL'}         {model.predict_proba(np.array([[5]]))[0] * 100:>6.1f}%
     7.0          PASS         {model.predict_proba(np.array([[7]]))[0] * 100:>6.1f}%
     9.0          PASS         {model.predict_proba(np.array([[9]]))[0] * 100:>6.1f}%

KEY INSIGHTS
{'=' * 70}
• Students who study more hours are more likely to pass
• The model learned a sigmoid curve that fits the data well
• Decision boundary is around {-model.bias / model.weights[0]:.1f} hours of study
• Model converged successfully with decreasing loss

HOW LOGISTIC REGRESSION WORKS
{'=' * 70}
1. Linear Combination: z = weight × input + bias
2. Sigmoid Function: σ(z) = 1 / (1 + e^(-z))
3. Output: Probability between 0 and 1
4. Classification: If probability ≥ 0.5 → Class 1 (PASS)
                  If probability < 0.5 → Class 0 (FAIL)

MATHEMATICAL FOUNDATION
{'=' * 70}
• Sigmoid maps any value to (0, 1) range
• Binary Cross-Entropy loss measures prediction quality
• Gradient Descent minimizes loss by adjusting weights
• Converges to optimal parameters that best fit the data

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

print(report)

# Save report
# with open('logistic_regression_report.txt', 'w') as f:
with open('logistic_regression_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✓ Report saved to: logistic_regression_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PROJECT COMPLETE!")
print("=" * 70)

print(f"\n🎓 What We Built:")
print(f"  ✓ Logistic Regression from scratch (no sklearn)")
print(f"  ✓ Sigmoid activation function")
print(f"  ✓ Binary Cross-Entropy loss")
print(f"  ✓ Gradient Descent optimizer")
print(f"  ✓ Prediction and probability methods")

print(f"\n📊 What We Did:")
print(f"  1. ✓ Generated student data")
print(f"  2. ✓ Implemented Logistic Regression class")
print(f"  3. ✓ Prepared data (train/test split)")
print(f"  4. ✓ Trained model ({model.n_iterations} iterations)")
print(f"  5. ✓ Made predictions")
print(f"  6. ✓ Evaluated performance ({accuracy * 100:.1f}% accuracy)")
print(f"  7. ✓ Predicted new students")
print(f"  8. ✓ Created 4 visualizations")
print(f"  9. ✓ Generated comprehensive report")

print(f"\n📁 Files Generated:")
files = [
    "logistic_regression_report.txt - Project report",
    "logistic_viz_1_sigmoid_curve.png - Sigmoid curve with data",
    "logistic_viz_2_loss_curve.png - Training loss over time",
    "logistic_viz_3_confusion_matrix.png - Confusion matrix",
    "logistic_viz_4_probabilities.png - Prediction probabilities"
]
for file in files:
    print(f"  • {file}")

print(f"\n🎯 Model Performance:")
print(f"  • Accuracy: {accuracy * 100:.2f}%")
print(f"  • Weight: {model.weights[0]:.4f}")
print(f"  • Bias: {model.bias:.4f}")

print(f"\n💡 Key Insight:")
print(f"  Decision boundary at ~{-model.bias / model.weights[0]:.1f} study hours")
print(f"  Study < {-model.bias / model.weights[0]:.1f} hours → Likely to FAIL")
print(f"  Study > {-model.bias / model.weights[0]:.1f} hours → Likely to PASS")

print("\n" + "=" * 70)
print("Logistic Regression implemented successfully from scratch!")
print("=" * 70)