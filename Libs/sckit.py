"""
SCIKIT-LEARN BASICS - COMPLETE BEGINNER'S GUIDE
================================================
Covering all fundamental Machine Learning concepts

Topics covered:
- Data preprocessing
- Train-test split
- Classification algorithms
- Regression algorithms
- Clustering algorithms
- Model evaluation metrics
- Feature scaling
- Cross-validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, mean_squared_error,
                            r2_score, silhouette_score)
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print("=" * 80)
print("SCIKIT-LEARN BASICS - MACHINE LEARNING TUTORIAL")
print("=" * 80)

# ============================================================================
# 1. LOADING DATASETS
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATASETS")
print("=" * 80)

# Load built-in datasets
# Classification dataset: Iris
iris = datasets.load_iris()
print("\n📊 Iris Dataset (Classification):")
print(f"  Features: {iris.feature_names}")
print(f"  Target classes: {iris.target_names}")
print(f"  Samples: {iris.data.shape[0]}, Features: {iris.data.shape[1]}")

# Regression dataset: Generate synthetic housing data
print("\n🏠 Synthetic Housing Dataset (Regression):")
np.random.seed(42)
n_samples_housing = 1000
california_data = np.random.randn(n_samples_housing, 8) * 2 + 5
california_target = (california_data[:, 0] * 0.5 +
                    california_data[:, 1] * 0.3 +
                    california_data[:, 2] * 0.2 +
                    np.random.randn(n_samples_housing) * 0.5)
california_feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                           'Population', 'AveOccup', 'Latitude', 'Longitude']

# Create a simple object to mimic the dataset structure
class SimpleDataset:
    def __init__(self, data, target, feature_names):
        self.data = data
        self.target = target
        self.feature_names = feature_names

california = SimpleDataset(california_data, california_target, california_feature_names)

print(f"  Features: {california.feature_names[:4]}... (8 total)")
print(f"  Target: Median house value")
print(f"  Samples: {california.data.shape[0]}, Features: {california.data.shape[1]}")

# Create a simple dataset
X_simple = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_simple = np.array([1, 2, 3, 4, 5])
print("\n🔢 Custom Dataset Created:")
print(f"  X shape: {X_simple.shape}")
print(f"  y shape: {y_simple.shape}")


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA PREPROCESSING")
print("=" * 80)

# Train-Test Split
print("\n--- Train-Test Split ---")
X_iris = iris.data
y_iris = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

print(f"Original dataset size: {len(X_iris)}")
print(f"Training set size: {len(X_train)} (70%)")
print(f"Test set size: {len(X_test)} (30%)")

# Feature Scaling
print("\n--- Feature Scaling (Standardization) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Before scaling (first sample):")
print(f"  {X_train[0]}")
print("After scaling (first sample):")
print(f"  {X_train_scaled[0]}")
print(f"Mean after scaling: {X_train_scaled.mean(axis=0)}")
print(f"Std after scaling: {X_train_scaled.std(axis=0)}")


# ============================================================================
# 3. CLASSIFICATION ALGORITHMS
# ============================================================================
print("\n" + "=" * 80)
print("3. CLASSIFICATION ALGORITHMS")
print("=" * 80)

# Store results for comparison
classification_results = {}

# 3.1 Logistic Regression
print("\n--- 3.1 Logistic Regression ---")
log_reg = LogisticRegression(random_state=42, max_iter=200)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
classification_results['Logistic Regression'] = accuracy_lr
print(f"Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")

# 3.2 Decision Tree
print("\n--- 3.2 Decision Tree Classifier ---")
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
classification_results['Decision Tree'] = accuracy_dt
print(f"Accuracy: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")

# 3.3 Random Forest
print("\n--- 3.3 Random Forest Classifier ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_results['Random Forest'] = accuracy_rf
print(f"Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print(f"Feature importances: {rf_clf.feature_importances_}")

# 3.4 K-Nearest Neighbors (KNN)
print("\n--- 3.4 K-Nearest Neighbors (KNN) ---")
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)
y_pred_knn = knn_clf.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
classification_results['KNN'] = accuracy_knn
print(f"Accuracy: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")

# 3.5 Support Vector Machine (SVM)
print("\n--- 3.5 Support Vector Machine (SVM) ---")
svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
classification_results['SVM'] = accuracy_svm
print(f"Accuracy: {accuracy_svm:.4f} ({accuracy_svm*100:.2f}%)")

# 3.6 Naive Bayes
print("\n--- 3.6 Naive Bayes ---")
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
classification_results['Naive Bayes'] = accuracy_nb
print(f"Accuracy: {accuracy_nb:.4f} ({accuracy_nb*100:.2f}%)")

# Summary
print("\n" + "=" * 50)
print("CLASSIFICATION ALGORITHMS COMPARISON")
print("=" * 50)
for name, acc in sorted(classification_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<25} {acc:.4f} ({acc*100:.2f}%)")


# ============================================================================
# 4. MODEL EVALUATION METRICS
# ============================================================================
print("\n" + "=" * 80)
print("4. MODEL EVALUATION METRICS")
print("=" * 80)

# Using Random Forest as example
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
print(f"\nConfusion Matrix Interpretation:")
print(f"  True Negatives, False Positives")
print(f"  False Negatives, True Positives")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

print("\n--- Individual Metrics ---")
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
f1 = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# ============================================================================
# 5. REGRESSION ALGORITHMS
# ============================================================================
print("\n" + "=" * 80)
print("5. REGRESSION ALGORITHMS")
print("=" * 80)

# Prepare regression data
X_reg = california.data[:1000]  # Use subset for speed
y_reg = california.target[:1000]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

regression_results = {}

# 5.1 Linear Regression
print("\n--- 5.1 Linear Regression ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_lin = lin_reg.predict(X_test_reg_scaled)

mse_lin = mean_squared_error(y_test_reg, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test_reg, y_pred_lin)
regression_results['Linear Regression'] = r2_lin

print(f"Mean Squared Error (MSE): {mse_lin:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lin:.4f}")
print(f"R² Score: {r2_lin:.4f}")

# 5.2 Decision Tree Regressor
print("\n--- 5.2 Decision Tree Regressor ---")
dt_reg = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_reg.fit(X_train_reg, y_train_reg)
y_pred_dt_reg = dt_reg.predict(X_test_reg)

mse_dt = mean_squared_error(y_test_reg, y_pred_dt_reg)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test_reg, y_pred_dt_reg)
regression_results['Decision Tree'] = r2_dt

print(f"Mean Squared Error (MSE): {mse_dt:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_dt:.4f}")
print(f"R² Score: {r2_dt:.4f}")

# 5.3 Random Forest Regressor
print("\n--- 5.3 Random Forest Regressor ---")
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg)

mse_rf = mean_squared_error(y_test_reg, y_pred_rf_reg)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test_reg, y_pred_rf_reg)
regression_results['Random Forest'] = r2_rf

print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
print(f"R² Score: {r2_rf:.4f}")

# Summary
print("\n" + "=" * 50)
print("REGRESSION ALGORITHMS COMPARISON (R² Score)")
print("=" * 50)
for name, r2 in sorted(regression_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<25} {r2:.4f}")


# ============================================================================
# 6. CLUSTERING ALGORITHMS
# ============================================================================
print("\n" + "=" * 80)
print("6. CLUSTERING ALGORITHMS (Unsupervised Learning)")
print("=" * 80)

# K-Means Clustering
print("\n--- K-Means Clustering ---")

# Use first 2 features for easy visualization
X_cluster = iris.data[:, :2]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)

print(f"Number of clusters: 3")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")

# Silhouette score
silhouette = silhouette_score(X_cluster, cluster_labels)
print(f"Silhouette Score: {silhouette:.4f}")

# Print cluster distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nCluster distribution:")
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples")


# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("7. CROSS-VALIDATION")
print("=" * 80)

print("\n--- K-Fold Cross-Validation ---")

# Cross-validation on Random Forest
rf_clf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_clf_cv, X_iris, y_iris, cv=5)

print(f"Cross-validation scores (5 folds): {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
print(f"95% Confidence Interval: {cv_scores.mean():.4f} +/- {1.96 * cv_scores.std():.4f}")


# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("8. FEATURE IMPORTANCE")
print("=" * 80)

# Feature importance from Random Forest
importances = rf_clf.feature_importances_
feature_names = iris.feature_names

print("\n--- Feature Importance Ranking ---")
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance_df.to_string(index=False))


# ============================================================================
# 9. MAKING PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("9. MAKING PREDICTIONS")
print("=" * 80)

# Create a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Similar to setosa
new_sample_scaled = scaler.transform(new_sample)

print("\n--- Single Prediction ---")
print(f"Input features: {new_sample[0]}")
prediction = rf_clf.predict(new_sample)
print(f"Predicted class: {iris.target_names[prediction[0]]}")

# Prediction probabilities
prediction_proba = rf_clf.predict_proba(new_sample)
print(f"\nPrediction probabilities:")
for i, class_name in enumerate(iris.target_names):
    print(f"  {class_name}: {prediction_proba[0][i]:.4f} ({prediction_proba[0][i]*100:.2f}%)")


# ============================================================================
# 10. SAVING AND LOADING MODELS
# ============================================================================
print("\n" + "=" * 80)
print("10. SAVING AND LOADING MODELS")
print("=" * 80)

import joblib

# Save model
model_filename = 'random_forest_model.pkl'
joblib.dump(rf_clf, model_filename)
print(f"\n✓ Model saved to: {model_filename}")

# Load model
loaded_model = joblib.load(model_filename)
print(f"✓ Model loaded successfully")

# Test loaded model
test_prediction = loaded_model.predict(new_sample)
print(f"\nPrediction from loaded model: {iris.target_names[test_prediction[0]]}")


# ============================================================================
# 11. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("11. CREATING VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Classification Results Comparison
print("\n📊 Creating classification comparison chart...")
fig, ax = plt.subplots(figsize=(12, 6))
models = list(classification_results.keys())
accuracies = list(classification_results.values())
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

bars = ax.barh(models, accuracies, color=colors)
ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Classification Algorithms Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.1)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('sklearn_01_classification_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: sklearn_01_classification_comparison.png")

# Visualization 2: Confusion Matrix Heatmap
print("\n📊 Creating confusion matrix heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues', aspect='auto')

# Add labels
ax.set_xticks(np.arange(len(iris.target_names)))
ax.set_yticks(np.arange(len(iris.target_names)))
ax.set_xticklabels(iris.target_names)
ax.set_yticklabels(iris.target_names)

# Add text annotations
for i in range(len(iris.target_names)):
    for j in range(len(iris.target_names)):
        text = ax.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black",
                      fontsize=14, fontweight='bold')

ax.set_title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('sklearn_02_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: sklearn_02_confusion_matrix.png")

# Visualization 3: Feature Importance
print("\n📊 Creating feature importance chart...")
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(importances)
pos = np.arange(sorted_idx.shape[0])

ax.barh(pos, importances[sorted_idx], color='skyblue', edgecolor='navy')
ax.set_yticks(pos)
ax.set_yticklabels(np.array(feature_names)[sorted_idx])
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('sklearn_03_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: sklearn_03_feature_importance.png")

# Visualization 4: K-Means Clustering
print("\n📊 Creating clustering visualization...")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data points
scatter = ax.scatter(X_cluster[:, 0], X_cluster[:, 1],
                    c=cluster_labels, cmap='viridis',
                    s=100, alpha=0.6, edgecolors='black')

# Plot cluster centers
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1],
          c='red', s=300, alpha=0.8, marker='X',
          edgecolors='black', linewidths=2,
          label='Cluster Centers')

ax.set_xlabel(iris.feature_names[0], fontsize=12, fontweight='bold')
ax.set_ylabel(iris.feature_names[1], fontsize=12, fontweight='bold')
ax.set_title('K-Means Clustering (k=3)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig('sklearn_04_kmeans_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: sklearn_04_kmeans_clustering.png")

# Visualization 5: Regression Predictions vs Actual
print("\n📊 Creating regression predictions chart...")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual vs predicted for Random Forest
ax.scatter(y_test_reg, y_pred_rf_reg, alpha=0.6, s=50, edgecolors='black')
ax.plot([y_test_reg.min(), y_test_reg.max()],
        [y_test_reg.min(), y_test_reg.max()],
        'r--', lw=2, label='Perfect Prediction')

ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
ax.set_title(f'Regression: Actual vs Predicted (R² = {r2_rf:.4f})',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sklearn_05_regression_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: sklearn_05_regression_predictions.png")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SCIKIT-LEARN BASICS COVERED:")
print("=" * 80)

topics = [
    "1. Loading Datasets (Built-in and Custom)",
    "2. Data Preprocessing (Train-Test Split, Feature Scaling)",
    "3. Classification Algorithms (6 algorithms)",
    "   - Logistic Regression",
    "   - Decision Tree",
    "   - Random Forest",
    "   - K-Nearest Neighbors (KNN)",
    "   - Support Vector Machine (SVM)",
    "   - Naive Bayes",
    "4. Model Evaluation Metrics (Accuracy, Precision, Recall, F1-Score)",
    "5. Regression Algorithms (3 algorithms)",
    "   - Linear Regression",
    "   - Decision Tree Regressor",
    "   - Random Forest Regressor",
    "6. Clustering (K-Means)",
    "7. Cross-Validation",
    "8. Feature Importance",
    "9. Making Predictions",
    "10. Saving and Loading Models",
    "11. Visualizations (5 charts created)"
]

for topic in topics:
    print(f"✓ {topic}")

print("\n" + "=" * 80)
print("FILES GENERATED:")
print("=" * 80)

files = [
    "sklearn_01_classification_comparison.png - Algorithm accuracy comparison",
    "sklearn_02_confusion_matrix.png - Confusion matrix heatmap",
    "sklearn_03_feature_importance.png - Feature importance ranking",
    "sklearn_04_kmeans_clustering.png - K-Means clustering visualization",
    "sklearn_05_regression_predictions.png - Actual vs Predicted values",
    "random_forest_model.pkl - Saved trained model"
]

for i, file in enumerate(files, 1):
    print(f"{i}. {file}")

print("\n" + "=" * 80)
print("All Scikit-Learn basics covered successfully!")
print("=" * 80)