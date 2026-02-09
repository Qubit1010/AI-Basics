"""
NAIVE BAYES CLASSIFIER FROM SCRATCH
====================================
Implementing Naive Bayes using plain Python (no sklearn)

Goal: Classify emails as SPAM or NOT SPAM based on word features
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

print("=" * 70)
print("NAIVE BAYES CLASSIFIER FROM SCRATCH")
print("=" * 70)

# ============================================================================
# 1. CREATE SAMPLE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: CREATE SAMPLE EMAIL DATA")
print("=" * 70)

# Sample emails (simplified - using word counts)
# Features: [count_of_'free', count_of_'money', count_of_'meeting', count_of_'project']
print("\nExplanation:")
print("  We'll use 4 word features to classify emails:")
print("  - 'free' (spam indicator)")
print("  - 'money' (spam indicator)")
print("  - 'meeting' (not spam indicator)")
print("  - 'project' (not spam indicator)")

# Training data: [free, money, meeting, project]
X_train = np.array([
    [3, 2, 0, 0],  # SPAM
    [2, 3, 0, 0],  # SPAM
    [4, 1, 0, 0],  # SPAM
    [0, 0, 2, 3],  # NOT SPAM
    [0, 0, 3, 2],  # NOT SPAM
    [0, 1, 2, 2],  # NOT SPAM
    [5, 2, 0, 0],  # SPAM
    [0, 0, 4, 1],  # NOT SPAM
    [3, 3, 0, 0],  # SPAM
    [0, 0, 1, 4],  # NOT SPAM
    [4, 2, 0, 0],  # SPAM
    [0, 0, 3, 3],  # NOT SPAM
    [2, 4, 0, 0],  # SPAM
    [0, 1, 3, 2],  # NOT SPAM
    [5, 1, 0, 0],  # SPAM
])

# Labels: 1 = SPAM, 0 = NOT SPAM
y_train = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Test data
X_test = np.array([
    [4, 3, 0, 0],  # Should be SPAM
    [0, 0, 3, 4],  # Should be NOT SPAM
    [3, 2, 1, 1],  # Mixed - could be either
    [1, 1, 2, 2],  # Mixed - could be either
])

y_test = np.array([1, 0, 1, 0])

print(f"\n✓ Generated email dataset")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: 4 word counts")

print(f"\nTraining Data Distribution:")
print(f"  SPAM emails: {np.sum(y_train == 1)} ({np.sum(y_train == 1) / len(y_train) * 100:.1f}%)")
print(f"  NOT SPAM emails: {np.sum(y_train == 0)} ({np.sum(y_train == 0) / len(y_train) * 100:.1f}%)")

print(f"\nSample emails:")
feature_names = ['free', 'money', 'meeting', 'project']
for i in range(5):
    label = "SPAM" if y_train[i] == 1 else "NOT SPAM"
    print(f"  Email {i + 1}: {dict(zip(feature_names, X_train[i]))} -> {label}")

# ============================================================================
# 2. NAIVE BAYES THEORY EXPLANATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: NAIVE BAYES THEORY")
print("=" * 70)

theory = """
Naive Bayes is based on Bayes' Theorem:

  P(Class|Features) = P(Features|Class) * P(Class) / P(Features)

For classification, we choose the class with highest probability.

KEY ASSUMPTIONS:
1. Features are independent (that's why it's "Naive")
2. Each feature contributes independently to the probability

For our spam classifier:
  P(SPAM|words) = P(words|SPAM) * P(SPAM) / P(words)

Since P(words) is same for all classes, we can ignore it and just compare:
  P(words|SPAM) * P(SPAM)  vs  P(words|NOT SPAM) * P(NOT SPAM)

STEPS:
1. Calculate prior probabilities: P(SPAM) and P(NOT SPAM)
2. Calculate likelihoods: P(word|SPAM) and P(word|NOT SPAM)
3. Apply Bayes theorem to classify new emails
"""

print(theory)

# ============================================================================
# 3. NAIVE BAYES CLASS IMPLEMENTATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: IMPLEMENT NAIVE BAYES CLASSIFIER")
print("=" * 70)


class NaiveBayesScratch:
    """
    Gaussian Naive Bayes implementation from scratch
    Assumes features follow a Gaussian (normal) distribution
    """

    def __init__(self):
        """Initialize the classifier"""
        self.classes = None
        self.class_priors = {}  # P(Class)
        self.means = {}  # Mean of each feature for each class
        self.variances = {}  # Variance of each feature for each class

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        """
        self.classes = np.unique(y)
        n_samples = len(y)

        print(f"\nTraining Naive Bayes...")
        print(f"  Number of classes: {len(self.classes)}")
        print(f"  Classes: {self.classes}")

        # Calculate prior probabilities and statistics for each class
        for c in self.classes:
            # Get samples belonging to class c
            X_c = X[y == c]

            # Prior probability: P(Class = c)
            self.class_priors[c] = len(X_c) / n_samples

            # Calculate mean and variance for each feature
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)

            # Add small epsilon to avoid division by zero
            self.variances[c] += 1e-9

            print(f"\n  Class {c}:")
            print(f"    Prior probability: {self.class_priors[c]:.4f}")
            print(f"    Mean per feature: {self.means[c]}")
            print(f"    Variance per feature: {self.variances[c]}")

    def _calculate_likelihood(self, x, mean, variance):
        """
        Calculate Gaussian probability density function

        P(x|class) = (1/sqrt(2*pi*variance)) * exp(-(x-mean)^2 / (2*variance))

        Parameters:
        -----------
        x : float
            Feature value
        mean : float
            Mean of the feature for the class
        variance : float
            Variance of the feature for the class

        Returns:
        --------
        likelihood : float
            Probability of x given the class
        """
        exponent = -((x - mean) ** 2) / (2 * variance)
        coefficient = 1 / np.sqrt(2 * np.pi * variance)
        return coefficient * np.exp(exponent)

    def _calculate_class_probability(self, x, c):
        """
        Calculate posterior probability for a class

        P(Class|x) proportional to P(Class) * Product(P(feature_i|Class))

        We use log probabilities to avoid numerical underflow
        """
        # Start with log of prior probability
        log_prob = np.log(self.class_priors[c])

        # Add log likelihoods for each feature
        for i in range(len(x)):
            likelihood = self._calculate_likelihood(
                x[i], self.means[c][i], self.variances[c][i]
            )
            log_prob += np.log(likelihood + 1e-9)  # Add epsilon to avoid log(0)

        return log_prob

    def predict_proba(self, X):
        """
        Predict class probabilities for samples

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data

        Returns:
        --------
        probabilities : array-like, shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            for j, c in enumerate(self.classes):
                probabilities[i, j] = self._calculate_class_probability(x, c)

        # Convert log probabilities to probabilities
        probabilities = np.exp(probabilities)

        # Normalize to get actual probabilities
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        """
        Predict class labels for samples

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data

        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        # Return class with highest probability
        return self.classes[np.argmax(probabilities, axis=1)]


print("\n✓ Naive Bayes class implemented!")
print("\nKey Components:")
print("  • Prior probabilities: P(Class)")
print("  • Gaussian likelihood: P(Feature|Class)")
print("  • Log probabilities: Avoid numerical underflow")
print("  • Prediction: Choose class with highest probability")

# ============================================================================
# 4. TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: TRAIN THE MODEL")
print("=" * 70)

# Create and train model
model = NaiveBayesScratch()
model.fit(X_train, y_train)

print(f"\n✓ Training complete!")

# ============================================================================
# 5. MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: MAKE PREDICTIONS")
print("=" * 70)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"\nPredictions on test set:")
print(f"{'Email Features':<30} {'Actual':<12} {'Predicted':<12} {'Probability':<15}")
print("-" * 70)

for i in range(len(X_test)):
    features_str = str(dict(zip(feature_names, X_test[i])))
    actual = "SPAM" if y_test[i] == 1 else "NOT SPAM"
    predicted = "SPAM" if y_pred[i] == 1 else "NOT SPAM"
    prob_spam = y_pred_proba[i][1] if len(y_pred_proba[i]) > 1 else 0

    print(f"{features_str:<30} {actual:<12} {predicted:<12} {prob_spam:<15.4f}")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: EVALUATE MODEL PERFORMANCE")
print("=" * 70)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)

# Calculate confusion matrix
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
print(f"                Predicted NOT SPAM  Predicted SPAM")
print(f"  Actual NOT SPAM       {true_negatives:^6}           {false_positives:^6}")
print(f"  Actual SPAM           {false_negatives:^6}           {true_positives:^6}")

print(f"\nInterpretation:")
print(f"  • Correctly identified NOT SPAM: {true_negatives}")
print(f"  • Correctly identified SPAM: {true_positives}")
print(f"  • False alarms (marked as SPAM wrongly): {false_positives}")
print(f"  • Missed SPAM (marked as NOT SPAM wrongly): {false_negatives}")

# ============================================================================
# 7. CLASSIFY NEW EMAILS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: CLASSIFY NEW EMAILS")
print("=" * 70)

# Create new emails to classify
new_emails = np.array([
    [5, 4, 0, 0],  # Lots of 'free' and 'money' - should be SPAM
    [0, 0, 4, 5],  # Lots of 'meeting' and 'project' - should be NOT SPAM
    [1, 0, 2, 3],  # Mixed but more work-related
    [6, 3, 0, 0],  # Very spammy
    [0, 1, 1, 1],  # Slightly mixed
])

# Predict
new_predictions = model.predict(new_emails)
new_probabilities = model.predict_proba(new_emails)

print(f"\n📧 Classifying New Emails:")
print(f"{'Email':<8} {'Features':<30} {'Prediction':<15} {'Spam Probability':<20}")
print("-" * 75)

for i in range(len(new_emails)):
    features_str = str(dict(zip(feature_names, new_emails[i])))
    prediction = "SPAM ⚠️" if new_predictions[i] == 1 else "NOT SPAM ✓"
    prob_spam = new_probabilities[i][1] if len(new_probabilities[i]) > 1 else 0

    print(f"Email {i + 1:<2} {features_str:<30} {prediction:<15} {prob_spam:<20.4f} ({prob_spam * 100:.1f}%)")

# ============================================================================
# 8. DEMONSTRATE PROBABILITY CALCULATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: SHOW PROBABILITY CALCULATIONS")
print("=" * 70)

print("\nExplanation:")
print("  Let's manually calculate probabilities for Email 1: [5, 4, 0, 0]")

test_email = new_emails[0]
print(f"\n  Email features: {dict(zip(feature_names, test_email))}")

for c in model.classes:
    class_name = "SPAM" if c == 1 else "NOT SPAM"
    print(f"\n  For class '{class_name}':")
    print(f"    Prior P({class_name}): {model.class_priors[c]:.4f}")
    print(f"    Feature means: {model.means[c]}")
    print(f"    Feature variances: {model.variances[c]}")

    # Calculate likelihood for each feature
    print(f"    Likelihoods:")
    for i, feature_name in enumerate(feature_names):
        likelihood = model._calculate_likelihood(
            test_email[i], model.means[c][i], model.variances[c][i]
        )
        print(f"      P({feature_name}={test_email[i]}|{class_name}): {likelihood:.6f}")

# ============================================================================
# 9. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: CREATE VISUALIZATIONS")
print("=" * 70)

# Visualization 1: Feature distributions by class
print("\n📊 Creating feature distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    ax = axes[i]

    # Get feature values for each class
    spam_values = X_train[y_train == 1, i]
    not_spam_values = X_train[y_train == 0, i]

    # Create histograms
    ax.hist(spam_values, bins=5, alpha=0.6, label='SPAM', color='red', edgecolor='black')
    ax.hist(not_spam_values, bins=5, alpha=0.6, label='NOT SPAM', color='green', edgecolor='black')

    ax.set_xlabel(f"Count of '{feature}'", fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f"Distribution of '{feature}'", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Word Count Distributions by Email Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('naive_bayes_viz_1_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: naive_bayes_viz_1_distributions.png")

# Visualization 2: Prediction probabilities
print("\n📊 Creating prediction probabilities chart...")

fig, ax = plt.subplots(figsize=(10, 6))

# Get probabilities for all test samples
all_probs = model.predict_proba(np.vstack([X_test, new_emails]))
spam_probs = all_probs[:, 1] if all_probs.shape[1] > 1 else np.zeros(len(all_probs))

# Create bar chart
colors = ['red' if p >= 0.5 else 'green' for p in spam_probs]
bars = ax.bar(range(len(spam_probs)), spam_probs, color=colors, alpha=0.7, edgecolor='black')

# Add threshold line
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (50%)')

ax.set_xlabel('Email Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Spam Probability', fontsize=12, fontweight='bold')
ax.set_title('Spam Probability for Each Email', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

# Add labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    label = f'{height:.2f}'
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
            label, ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('naive_bayes_viz_2_probabilities.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: naive_bayes_viz_2_probabilities.png")

# Visualization 3: Confusion Matrix
print("\n📊 Creating confusion matrix...")

fig, ax = plt.subplots(figsize=(8, 6))

confusion_matrix = np.array([[true_negatives, false_positives],
                             [false_negatives, true_positives]])

im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')

# Add labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted NOT SPAM', 'Predicted SPAM'])
ax.set_yticklabels(['Actual NOT SPAM', 'Actual SPAM'])

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, confusion_matrix[i, j],
                       ha="center", va="center",
                       color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black",
                       fontsize=20, fontweight='bold')

ax.set_title('Confusion Matrix - Naive Bayes', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('naive_bayes_viz_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: naive_bayes_viz_3_confusion_matrix.png")

# Visualization 4: Class comparison
print("\n📊 Creating class comparison chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean comparison
ax1 = axes[0]
x_pos = np.arange(len(feature_names))
width = 0.35

spam_means = model.means[1]
not_spam_means = model.means[0]

ax1.bar(x_pos - width / 2, spam_means, width, label='SPAM', color='red', alpha=0.7, edgecolor='black')
ax1.bar(x_pos + width / 2, not_spam_means, width, label='NOT SPAM', color='green', alpha=0.7, edgecolor='black')

ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Count', fontsize=12, fontweight='bold')
ax1.set_title('Feature Means by Class', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(feature_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Prior probabilities
ax2 = axes[1]
classes_labels = ['NOT SPAM', 'SPAM']
priors = [model.class_priors[0], model.class_priors[1]]
colors_pie = ['green', 'red']

wedges, texts, autotexts = ax2.pie(priors, labels=classes_labels, autopct='%1.1f%%',
                                   colors=colors_pie, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax2.set_title('Class Prior Probabilities', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('naive_bayes_viz_4_class_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: naive_bayes_viz_4_class_comparison.png")

# ============================================================================
# 10. GENERATE REPORT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: GENERATE REPORT")
print("=" * 70)

report = f"""
{'=' * 70}
NAIVE BAYES CLASSIFIER - PROJECT REPORT
{'=' * 70}

PROJECT OVERVIEW
{'=' * 70}
Objective: Classify emails as SPAM or NOT SPAM
Algorithm: Gaussian Naive Bayes (implemented from scratch)
Dataset: 15 training emails, 4 test emails

NAIVE BAYES THEORY
{'=' * 70}
Bayes' Theorem:
  P(Class|Features) = P(Features|Class) * P(Class) / P(Features)

Key Assumption:
  Features are conditionally independent given the class

Implementation:
  • Gaussian likelihood function for continuous features
  • Log probabilities to avoid numerical underflow
  • Maximum a posteriori (MAP) classification

FEATURES USED
{'=' * 70}
1. Count of word 'free' (spam indicator)
2. Count of word 'money' (spam indicator)
3. Count of word 'meeting' (not spam indicator)
4. Count of word 'project' (not spam indicator)

TRAINING DATA
{'=' * 70}
Total emails: {len(X_train)}
  SPAM: {np.sum(y_train == 1)} emails ({np.sum(y_train == 1) / len(y_train) * 100:.1f}%)
  NOT SPAM: {np.sum(y_train == 0)} emails ({np.sum(y_train == 0) / len(y_train) * 100:.1f}%)

LEARNED PARAMETERS
{'=' * 70}
Prior Probabilities:
  P(SPAM): {model.class_priors[1]:.4f}
  P(NOT SPAM): {model.class_priors[0]:.4f}

Feature Means (SPAM):
  {dict(zip(feature_names, model.means[1]))}

Feature Means (NOT SPAM):
  {dict(zip(feature_names, model.means[0]))}

MODEL PERFORMANCE
{'=' * 70}
Test Set Results:
  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1-Score: {f1_score:.4f}

Confusion Matrix:
                Predicted NOT SPAM  Predicted SPAM
  Actual NOT SPAM       {true_negatives}                  {false_positives}
  Actual SPAM           {false_negatives}                  {true_positives}

HOW NAIVE BAYES WORKS
{'=' * 70}
Step 1: Calculate Prior Probabilities
  - Count frequency of each class in training data

Step 2: Calculate Likelihood for Each Feature
  - Assume Gaussian distribution
  - Calculate mean and variance for each feature per class
  - Use probability density function

Step 3: Apply Bayes Theorem
  - Multiply prior by likelihoods
  - Use log probabilities to prevent underflow
  - Choose class with highest probability

Step 4: Make Prediction
  - New email -> Calculate P(SPAM|features)
  - Calculate P(NOT SPAM|features)
  - Predict class with higher probability

KEY INSIGHTS
{'=' * 70}
• SPAM emails have high counts of 'free' and 'money'
• NOT SPAM emails have high counts of 'meeting' and 'project'
• Model achieved {accuracy * 100:.1f}% accuracy on test set
• Naive independence assumption works well for this task
• Simple yet effective algorithm for text classification

ADVANTAGES OF NAIVE BAYES
{'=' * 70}
✓ Fast training and prediction
✓ Works well with small datasets
✓ Handles high-dimensional data
✓ Probabilistic predictions
✓ Simple to implement and understand

LIMITATIONS
{'=' * 70}
✗ Assumes feature independence (often violated)
✗ Sensitive to irrelevant features
✗ Zero-frequency problem (solved with smoothing)
✗ Not always best for complex relationships

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

print(report)

# Save report
with open('naive_bayes_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Report saved to: naive_bayes_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PROJECT COMPLETE!")
print("=" * 70)

print(f"\n📧 What We Built:")
print(f"  ✓ Naive Bayes classifier from scratch (no sklearn)")
print(f"  ✓ Gaussian likelihood function")
print(f"  ✓ Prior probability calculation")
print(f"  ✓ Log probability for numerical stability")
print(f"  ✓ Probability predictions")

print(f"\n📊 What We Did:")
print(f"  1. ✓ Created email dataset (spam/not spam)")
print(f"  2. ✓ Explained Naive Bayes theory")
print(f"  3. ✓ Implemented Naive Bayes class")
print(f"  4. ✓ Trained model on 15 emails")
print(f"  5. ✓ Made predictions on test set")
print(f"  6. ✓ Evaluated performance ({accuracy * 100:.1f}% accuracy)")
print(f"  7. ✓ Classified new emails")
print(f"  8. ✓ Demonstrated probability calculations")
print(f"  9. ✓ Created 4 visualizations")
print(f"  10. ✓ Generated comprehensive report")

print(f"\n📁 Files Generated:")
files = [
    "naive_bayes_report.txt - Project report",
    "naive_bayes_viz_1_distributions.png - Feature distributions",
    "naive_bayes_viz_2_probabilities.png - Spam probabilities",
    "naive_bayes_viz_3_confusion_matrix.png - Confusion matrix",
    "naive_bayes_viz_4_class_comparison.png - Class comparison"
]
for file in files:
    print(f"  • {file}")

print(f"\n🎯 Model Performance:")
print(f"  • Accuracy: {accuracy * 100:.2f}%")
print(f"  • SPAM prior: {model.class_priors[1]:.4f}")
print(f"  • NOT SPAM prior: {model.class_priors[0]:.4f}")

print(f"\n💡 Key Insight:")
print(f"  Naive Bayes works by assuming feature independence")
print(f"  and using probability theory to make predictions!")

print("\n" + "=" * 70)
print("Naive Bayes implemented successfully from scratch!")
print("=" * 70)