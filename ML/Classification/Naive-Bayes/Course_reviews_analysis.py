"""
USER COURSE REVIEWS - COMPLETE DATA SCIENCE PIPELINE
====================================================
1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Naive Bayes Classification
4. Model Evaluation

Goal: Predict review ratings based on review comments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("USER COURSE REVIEWS - DATA SCIENCE PIPELINE")
print("=" * 80)

# ============================================================================
# PART 1: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: DATA PREPROCESSING")
print("=" * 80)

# Load data
print("\n--- Step 1.1: Load Data ---")
df = pd.read_csv('user_courses_review_09_2023.csv',
                 quotechar='"', escapechar='\\', on_bad_lines='skip')
print(f"✓ Data loaded successfully")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

print("\n--- Step 1.2: Initial Inspection ---")
print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\n--- Step 1.3: Check Missing Values ---")
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

# Check data types
print("\n--- Step 1.4: Check Data Types ---")
print(df.dtypes)

# Handle missing values
print("\n--- Step 1.5: Handle Missing Values ---")
print(f"Rows before cleaning: {len(df)}")

# Drop rows with missing values
df_clean = df.dropna()
print(f"Rows after removing missing values: {len(df_clean)}")
print(f"Rows removed: {len(df) - len(df_clean)}")

# Convert review_rating to numeric and filter valid ratings (1-5)
print("\n--- Step 1.5b: Clean Review Ratings ---")
df_clean['review_rating'] = pd.to_numeric(df_clean['review_rating'], errors='coerce')
df_clean = df_clean[df_clean['review_rating'].isin([1, 2, 3, 4, 5])]
print(f"Rows after filtering valid ratings (1-5): {len(df_clean)}")

# Convert to integer
df_clean['review_rating'] = df_clean['review_rating'].astype(int)

# Check review_rating distribution
print("\n--- Step 1.6: Review Rating Distribution ---")
rating_counts = df_clean['review_rating'].value_counts().sort_index()
print("\nRating distribution:")
print(rating_counts)
print(f"\nRating percentages:")
for rating, count in rating_counts.items():
    print(f"  Rating {rating}: {count} ({count / len(df_clean) * 100:.2f}%)")

# Check for duplicates
print("\n--- Step 1.7: Check Duplicates ---")
duplicates = df_clean.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Clean text data
print("\n--- Step 1.8: Text Preprocessing ---")
print("Converting text to lowercase and removing extra whitespace...")

df_clean['review_comment_clean'] = df_clean['review_comment'].str.lower().str.strip()
df_clean['course_name_clean'] = df_clean['course_name'].str.strip()
df_clean['lecture_name_clean'] = df_clean['lecture_name'].str.strip()

print("✓ Text cleaned")

# Add text length feature
print("\n--- Step 1.9: Feature Engineering ---")
df_clean['comment_length'] = df_clean['review_comment_clean'].str.len()
df_clean['word_count'] = df_clean['review_comment_clean'].str.split().str.len()

print("✓ Created new features:")
print(f"  - comment_length: Character count")
print(f"  - word_count: Word count")

print(f"\nComment length statistics:")
print(df_clean['comment_length'].describe())

print(f"\nWord count statistics:")
print(df_clean['word_count'].describe())

# Save preprocessed data
preprocessed_file = 'reviews_preprocessed.csv'
df_clean.to_csv(preprocessed_file, index=False)
print(f"\n✓ Preprocessed data saved to: {preprocessed_file}")

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Distribution analysis
print("\n--- Step 2.1: Distribution Analysis ---")

# Rating distribution
print("\nRating statistics:")
print(df_clean['review_rating'].describe())

# Top courses
print("\n--- Step 2.2: Top Courses ---")
top_courses = df_clean['course_name_clean'].value_counts().head(10)
print("\nTop 10 courses by review count:")
print(top_courses)

# Average rating by course
print("\n--- Step 2.3: Average Rating by Course ---")
avg_rating_by_course = df_clean.groupby('course_name_clean')['review_rating'].mean().sort_values(ascending=False).head(
    10)
print("\nTop 10 courses by average rating:")
print(avg_rating_by_course)

# Correlation analysis
print("\n--- Step 2.4: Correlation Analysis ---")
numeric_cols = ['review_rating', 'comment_length', 'word_count']
correlation = df_clean[numeric_cols].corr()
print("\nCorrelation matrix:")
print(correlation)

# Outlier detection
print("\n--- Step 2.5: Outlier Detection ---")

# Comment length outliers
Q1 = df_clean['comment_length'].quantile(0.25)
Q3 = df_clean['comment_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['comment_length'] < lower_bound) |
                    (df_clean['comment_length'] > upper_bound)]

print(f"\nComment length outliers:")
print(f"  Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"  Outlier bounds: < {lower_bound:.1f} or > {upper_bound:.1f}")
print(f"  Number of outliers: {len(outliers)} ({len(outliers) / len(df_clean) * 100:.2f}%)")

# Create visualizations
print("\n--- Step 2.6: Creating Visualizations ---")

# Visualization 1: Rating Distribution
print("\n📊 Creating rating distribution chart...")
fig, ax = plt.subplots(figsize=(10, 6))
rating_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
ax.set_xlabel('Review Rating', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Review Ratings', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

# Add count labels
for i, v in enumerate(rating_counts.values):
    ax.text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_viz_1_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_viz_1_rating_distribution.png")

# Visualization 2: Top Courses
print("\n📊 Creating top courses chart...")
fig, ax = plt.subplots(figsize=(12, 6))
top_courses.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
ax.set_xlabel('Number of Reviews', fontsize=12, fontweight='bold')
ax.set_ylabel('Course Name', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Courses by Review Count', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_viz_2_top_courses.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_viz_2_top_courses.png")

# Visualization 3: Comment Length Distribution
print("\n📊 Creating comment length distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_clean['comment_length'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax.axvline(df_clean['comment_length'].mean(), color='red', linestyle='--',
           linewidth=2, label=f"Mean: {df_clean['comment_length'].mean():.0f}")
ax.axvline(df_clean['comment_length'].median(), color='blue', linestyle='--',
           linewidth=2, label=f"Median: {df_clean['comment_length'].median():.0f}")
ax.set_xlabel('Comment Length (characters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Comment Lengths', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_viz_3_comment_length.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_viz_3_comment_length.png")

# Visualization 4: Rating vs Comment Length
print("\n📊 Creating rating vs comment length boxplot...")
fig, ax = plt.subplots(figsize=(10, 6))
df_clean.boxplot(column='comment_length', by='review_rating', ax=ax)
ax.set_xlabel('Review Rating', fontsize=12, fontweight='bold')
ax.set_ylabel('Comment Length (characters)', fontsize=12, fontweight='bold')
ax.set_title('Comment Length by Rating', fontsize=14, fontweight='bold')
plt.suptitle('')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eda_viz_4_rating_vs_length.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_viz_4_rating_vs_length.png")

# Visualization 5: Correlation Heatmap
print("\n📊 Creating correlation heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm',
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_viz_5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_viz_5_correlation_heatmap.png")

# Visualization 6: Word Count Distribution by Rating
print("\n📊 Creating word count distribution by rating...")
fig, ax = plt.subplots(figsize=(12, 6))
for rating in sorted(df_clean['review_rating'].unique()):
    data = df_clean[df_clean['review_rating'] == rating]['word_count']
    ax.hist(data, bins=30, alpha=0.5, label=f'Rating {rating}', edgecolor='black')
ax.set_xlabel('Word Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Word Count Distribution by Rating', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_viz_6_wordcount_by_rating.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_viz_6_wordcount_by_rating.png")

# ============================================================================
# PART 3: PREPARE DATA FOR NAIVE BAYES
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: PREPARE DATA FOR NAIVE BAYES")
print("=" * 80)

# For better classification, let's simplify to binary or 3-class problem
print("\n--- Step 3.1: Create Target Variable ---")

# Option 1: Binary classification (Good vs Bad)
df_clean['rating_binary'] = (df_clean['review_rating'] >= 4).astype(int)
print("\nBinary classification:")
print(df_clean['rating_binary'].value_counts())
print(f"  0 (Bad - ratings 1-3): {(df_clean['rating_binary'] == 0).sum()}")
print(f"  1 (Good - ratings 4-5): {(df_clean['rating_binary'] == 1).sum()}")


# Option 2: 3-class classification
def categorize_rating(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'


df_clean['rating_category'] = df_clean['review_rating'].apply(categorize_rating)
print("\n3-class classification:")
print(df_clean['rating_category'].value_counts())

# Text vectorization
print("\n--- Step 3.2: Text Vectorization ---")
print("\nExplanation:")
print("  Converting text to numerical features using TF-IDF")
print("  (Term Frequency - Inverse Document Frequency)")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=1000, stop_words='english',
                        min_df=2, max_df=0.8)
X_tfidf = tfidf.fit_transform(df_clean['review_comment_clean'])

print(f"\n✓ Text vectorized")
print(f"  Shape: {X_tfidf.shape}")
print(f"  Features (words): {X_tfidf.shape[1]}")
print(f"  Samples (reviews): {X_tfidf.shape[0]}")

# Top features
print("\n--- Step 3.3: Top TF-IDF Features ---")
feature_names = tfidf.get_feature_names_out()
print(f"\nTop 20 important words:")
# Get average TF-IDF scores
tfidf_scores = np.asarray(X_tfidf.mean(axis=0)).flatten()
top_indices = tfidf_scores.argsort()[-20:][::-1]
for i, idx in enumerate(top_indices[:20], 1):
    print(f"  {i}. {feature_names[idx]}: {tfidf_scores[idx]:.4f}")

# Split data
print("\n--- Step 3.4: Train-Test Split ---")

# For binary classification
y_binary = df_clean['rating_binary'].values
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\nBinary Classification Split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}")

# ============================================================================
# PART 4: TRAIN NAIVE BAYES MODEL
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: TRAIN NAIVE BAYES MODEL")
print("=" * 80)

print("\n--- Step 4.1: Multinomial Naive Bayes ---")
print("\nExplanation:")
print("  Multinomial Naive Bayes is ideal for text classification")
print("  It works well with word count or TF-IDF features")

# Train Multinomial Naive Bayes
nb_model = MultinomialNB(alpha=1.0)  # alpha is Laplace smoothing
nb_model.fit(X_train, y_train)

print(f"\n✓ Model trained successfully")
print(f"  Algorithm: Multinomial Naive Bayes")
print(f"  Smoothing parameter (alpha): {nb_model.alpha}")
print(f"  Number of classes: {len(nb_model.classes_)}")
print(f"  Class labels: {nb_model.classes_}")

# Class probabilities
print(f"\n--- Step 4.2: Class Prior Probabilities ---")
class_priors = np.exp(nb_model.class_log_prior_)
for i, class_label in enumerate(nb_model.classes_):
    class_name = "Good (4-5)" if class_label == 1 else "Bad (1-3)"
    print(f"  P({class_name}): {class_priors[i]:.4f}")

# ============================================================================
# PART 5: MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: MAKE PREDICTIONS")
print("=" * 80)

# Predict on test set
y_pred = nb_model.predict(X_test)
y_pred_proba = nb_model.predict_proba(X_test)

print("\n--- Sample Predictions ---")
print(f"{'Actual':<10} {'Predicted':<10} {'Confidence':<15} {'Review Comment':<60}")
print("-" * 95)

# Show first 10 predictions
sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
for i in sample_indices:
    actual = "Good" if y_test[i] == 1 else "Bad"
    predicted = "Good" if y_pred[i] == 1 else "Bad"
    confidence = np.max(y_pred_proba[i])
    # Get the original review comment from the test set
    original_idx = df_clean.index[X_train.shape[0] + i]
    comment = df_clean.loc[original_idx, 'review_comment_clean'][:50]

    print(f"{actual:<10} {predicted:<10} {confidence:<15.4f} {comment:<60}")

# ============================================================================
# PART 6: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 6: MODEL EVALUATION")
print("=" * 80)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Overall Performance ---")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Classification Report
print(f"\n--- Detailed Classification Report ---")
target_names = ['Bad (1-3)', 'Good (4-5)']
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n--- Confusion Matrix ---")
print(cm)
print(f"\nInterpretation:")
print(f"  True Negatives (Correctly predicted Bad): {cm[0, 0]}")
print(f"  False Positives (Incorrectly predicted Good): {cm[0, 1]}")
print(f"  False Negatives (Incorrectly predicted Bad): {cm[1, 0]}")
print(f"  True Positives (Correctly predicted Good): {cm[1, 1]}")

# Calculate metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"\n--- Weighted Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Per-class metrics
print(f"\n--- Per-Class Performance ---")
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    y_test, y_pred, average=None
)

for i, class_name in enumerate(target_names):
    print(f"\n{class_name}:")
    print(f"  Precision: {precision_per_class[i]:.4f}")
    print(f"  Recall: {recall_per_class[i]:.4f}")
    print(f"  F1-Score: {f1_per_class[i]:.4f}")
    print(f"  Support: {support_per_class[i]}")

# ============================================================================
# PART 7: EVALUATION VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 7: EVALUATION VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Confusion Matrix Heatmap
print("\n📊 Creating confusion matrix heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Naive Bayes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eval_viz_1_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eval_viz_1_confusion_matrix.png")

# Visualization 2: Performance Metrics
print("\n📊 Creating performance metrics chart...")
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Precision', 'Recall', 'F1-Score']
bad_scores = [precision_per_class[0], recall_per_class[0], f1_per_class[0]]
good_scores = [precision_per_class[1], recall_per_class[1], f1_per_class[1]]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width / 2, bad_scores, width, label='Bad (1-3)', color='#ff6b6b', edgecolor='black')
bars2 = ax.bar(x + width / 2, good_scores, width, label='Good (4-5)', color='#4ecdc4', edgecolor='black')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics by Class', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('eval_viz_2_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eval_viz_2_metrics.png")

# Visualization 3: Prediction Confidence Distribution
print("\n📊 Creating prediction confidence distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

# Get max probability for each prediction (confidence)
confidence_scores = np.max(y_pred_proba, axis=1)

ax.hist(confidence_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(confidence_scores.mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {confidence_scores.mean():.3f}')
ax.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Prediction Confidence Scores', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eval_viz_3_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eval_viz_3_confidence.png")

# Visualization 4: Feature Importance (Top words per class)
print("\n📊 Creating feature importance visualization...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Get feature log probabilities
feature_log_prob = nb_model.feature_log_prob_

# Top features for Bad reviews
bad_top_indices = feature_log_prob[0].argsort()[-15:][::-1]
bad_top_features = [feature_names[i] for i in bad_top_indices]
bad_top_scores = feature_log_prob[0][bad_top_indices]

axes[0].barh(range(15), bad_top_scores, color='#ff6b6b', edgecolor='black')
axes[0].set_yticks(range(15))
axes[0].set_yticklabels(bad_top_features)
axes[0].set_xlabel('Log Probability', fontsize=11, fontweight='bold')
axes[0].set_title('Top 15 Features for Bad Reviews', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Top features for Good reviews
good_top_indices = feature_log_prob[1].argsort()[-15:][::-1]
good_top_features = [feature_names[i] for i in good_top_indices]
good_top_scores = feature_log_prob[1][good_top_indices]

axes[1].barh(range(15), good_top_scores, color='#4ecdc4', edgecolor='black')
axes[1].set_yticks(range(15))
axes[1].set_yticklabels(good_top_features)
axes[1].set_xlabel('Log Probability', fontsize=11, fontweight='bold')
axes[1].set_title('Top 15 Features for Good Reviews', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('eval_viz_4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: eval_viz_4_feature_importance.png")

# ============================================================================
# PART 8: TEST ON NEW REVIEWS
# ============================================================================
print("\n" + "=" * 80)
print("PART 8: TEST ON NEW REVIEWS")
print("=" * 80)

# Create some new review examples
new_reviews = [
    "This course is absolutely amazing! I learned so much and the instructor was excellent.",
    "Terrible course, waste of time and money. Very disappointed.",
    "It was okay, not great but not terrible either.",
    "Best course I've ever taken! Highly recommend to everyone!",
    "Poor quality content and boring lectures."
]

# Transform and predict
new_reviews_tfidf = tfidf.transform([r.lower() for r in new_reviews])
new_predictions = nb_model.predict(new_reviews_tfidf)
new_probabilities = nb_model.predict_proba(new_reviews_tfidf)

print("\n🆕 Predictions for New Reviews:")
print(f"{'Review':<70} {'Prediction':<15} {'Confidence':<12}")
print("-" * 100)

for i, review in enumerate(new_reviews):
    pred = "Good (4-5)" if new_predictions[i] == 1 else "Bad (1-3)"
    conf = np.max(new_probabilities[i])
    print(f"{review:<70} {pred:<15} {conf:<12.4f}")

# ============================================================================
# PART 9: GENERATE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("PART 9: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'=' * 80}
USER COURSE REVIEWS - COMPLETE ANALYSIS REPORT
{'=' * 80}

DATASET SUMMARY
{'=' * 80}
Original file: user_courses_review_09_2023.csv
Processed file: reviews_preprocessed.csv

Total records: {len(df)}
Records after cleaning: {len(df_clean)}
Records removed: {len(df) - len(df_clean)}

Columns: {df.columns.tolist()}

PREPROCESSING STEPS
{'=' * 80}
1. ✓ Data Loading - Loaded {len(df)} reviews
2. ✓ Missing Value Handling - Removed {len(df) - len(df_clean)} rows
3. ✓ Text Cleaning - Lowercase, strip whitespace
4. ✓ Feature Engineering - Added comment_length, word_count
5. ✓ Text Vectorization - TF-IDF with {X_tfidf.shape[1]} features

EXPLORATORY DATA ANALYSIS
{'=' * 80}
Rating Distribution:
{rating_counts.to_string()}

Comment Statistics:
  Average length: {df_clean['comment_length'].mean():.2f} characters
  Average words: {df_clean['word_count'].mean():.2f} words
  Median length: {df_clean['comment_length'].median():.2f} characters

Top 3 Courses:
{chr(10).join([f'  {i + 1}. {course}: {count} reviews' for i, (course, count) in enumerate(top_courses.head(3).items())])}

Correlations:
  Rating vs Comment Length: {correlation.loc['review_rating', 'comment_length']:.4f}
  Rating vs Word Count: {correlation.loc['review_rating', 'word_count']:.4f}

NAIVE BAYES CLASSIFICATION
{'=' * 80}
Task: Binary classification (Good vs Bad reviews)
  - Bad (1-3 stars): {(y_binary == 0).sum()} reviews
  - Good (4-5 stars): {(y_binary == 1).sum()} reviews

Algorithm: Multinomial Naive Bayes
Features: TF-IDF vectorization ({X_tfidf.shape[1]} features)
Training samples: {X_train.shape[0]}
Test samples: {X_test.shape[0]}

MODEL PERFORMANCE
{'=' * 80}
Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)

Per-Class Metrics:
  Bad Reviews (1-3):
    Precision: {precision_per_class[0]:.4f}
    Recall: {recall_per_class[0]:.4f}
    F1-Score: {f1_per_class[0]:.4f}

  Good Reviews (4-5):
    Precision: {precision_per_class[1]:.4f}
    Recall: {recall_per_class[1]:.4f}
    F1-Score: {f1_per_class[1]:.4f}

Weighted Averages:
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1-Score: {f1:.4f}

Confusion Matrix:
                Predicted Bad  Predicted Good
  Actual Bad         {cm[0, 0]:^6}        {cm[0, 1]:^6}
  Actual Good        {cm[1, 0]:^6}        {cm[1, 1]:^6}

KEY INSIGHTS
{'=' * 80}
• {(y_binary == 1).sum() / len(y_binary) * 100:.1f}% of reviews are positive (4-5 stars)
• Model achieves {accuracy * 100:.1f}% accuracy in predicting review sentiment
• Average prediction confidence: {confidence_scores.mean():.3f}
• TF-IDF effectively captures review sentiment from text
• Multinomial Naive Bayes performs well on text classification

TOP PREDICTIVE WORDS
{'=' * 80}
Words indicating Bad reviews:
{chr(10).join([f'  {i + 1}. {feature_names[idx]}' for i, idx in enumerate(bad_top_indices[:10])])}

Words indicating Good reviews:
{chr(10).join([f'  {i + 1}. {feature_names[idx]}' for i, idx in enumerate(good_top_indices[:10])])}

FILES GENERATED
{'=' * 80}
Data Files:
  • reviews_preprocessed.csv - Cleaned and preprocessed data

EDA Visualizations:
  • eda_viz_1_rating_distribution.png
  • eda_viz_2_top_courses.png
  • eda_viz_3_comment_length.png
  • eda_viz_4_rating_vs_length.png
  • eda_viz_5_correlation_heatmap.png
  • eda_viz_6_wordcount_by_rating.png

Model Evaluation:
  • eval_viz_1_confusion_matrix.png
  • eval_viz_2_metrics.png
  • eval_viz_3_confidence.png
  • eval_viz_4_feature_importance.png

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

print(report)

# Save report
with open('analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Report saved to: analysis_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPLETE DATA SCIENCE PIPELINE FINISHED!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"  ✓ Preprocessed {len(df_clean)} reviews")
print(f"  ✓ Performed comprehensive EDA")
print(f"  ✓ Trained Naive Bayes classifier")
print(f"  ✓ Achieved {accuracy * 100:.2f}% accuracy")
print(f"  ✓ Generated 10 visualizations")
print(f"  ✓ Created comprehensive report")

print(f"\n🎯 Model Performance:")
print(f"  Accuracy: {accuracy * 100:.2f}%")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

print("\n" + "=" * 80)
print("All analysis complete! Check the output files for detailed results.")
print("=" * 80)