"""
K-Means Clustering with Scikit-Learn
Simple Project: Customer Segmentation for Marketing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("=" * 70)
print("CUSTOMER SEGMENTATION - K-MEANS CLUSTERING")
print("=" * 70)

# ============================================================================
# 1. GENERATE CUSTOMER DATA
# ============================================================================

n_customers = 500

# Generate realistic customer data
data = {
    'age': np.random.randint(18, 70, n_customers),
    'annual_income': np.random.randint(15000, 150000, n_customers),
    'spending_score': np.random.randint(1, 100, n_customers),  # 1-100 scale
    'purchase_frequency': np.random.randint(1, 50, n_customers),  # purchases per year
    'avg_transaction_value': np.random.uniform(10, 500, n_customers)
}

df = pd.DataFrame(data)

print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nStatistical Summary:")
print(df.describe())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Select features for clustering
X = df[['annual_income', 'spending_score', 'purchase_frequency', 'avg_transaction_value']].values

# Standardize features (important for K-means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nOriginal data shape: {X.shape}")
print(f"Scaled data shape: {X_scaled.shape}")
print(f"\nScaled data statistics:")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}")

# ============================================================================
# 3. FIND OPTIMAL NUMBER OF CLUSTERS - ELBOW METHOD
# ============================================================================

print("\n" + "=" * 70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)

# Test different numbers of clusters
k_range = range(2, 11)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))

    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, "
          f"Silhouette={silhouette_score(X_scaled, kmeans.labels_):.3f}")

# ============================================================================
# 4. TRAIN FINAL MODEL
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

# Choose optimal K (let's say 4 based on elbow method)
optimal_k = 4

# Train final model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['cluster'] = clusters

print(f"\nOptimal K: {optimal_k}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, clusters):.3f}")

# ============================================================================
# 5. ANALYZE CLUSTERS
# ============================================================================

print("\n" + "=" * 70)
print("CLUSTER ANALYSIS")
print("=" * 70)

# Cluster statistics
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]
    print(f"\n{'=' * 50}")
    print(f"CLUSTER {i} - {len(cluster_data)} customers ({len(cluster_data) / len(df) * 100:.1f}%)")
    print(f"{'=' * 50}")
    print(f"Average Age: {cluster_data['age'].mean():.1f} years")
    print(f"Average Income: ${cluster_data['annual_income'].mean():,.0f}")
    print(f"Average Spending Score: {cluster_data['spending_score'].mean():.1f}/100")
    print(f"Average Purchase Frequency: {cluster_data['purchase_frequency'].mean():.1f} times/year")
    print(f"Average Transaction Value: ${cluster_data['avg_transaction_value'].mean():.2f}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))

# --------------------- Plot 1: Elbow Method ---------------------
ax1 = plt.subplot(3, 3, 1)
ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='blue')
ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
            label=f'Optimal K={optimal_k}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Inertia (WCSS)', fontsize=11, fontweight='bold')
ax1.set_title('Elbow Method', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_range)

# --------------------- Plot 2: Silhouette Score ---------------------
ax2 = plt.subplot(3, 3, 2)
ax2.plot(k_range, silhouette_scores, marker='s', linewidth=2,
         markersize=8, color='green')
ax2.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
            label=f'Optimal K={optimal_k}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
ax2.set_title('Silhouette Analysis', fontweight='bold', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(k_range)

# --------------------- Plot 3: Davies-Bouldin Index ---------------------
ax3 = plt.subplot(3, 3, 3)
ax3.plot(k_range, davies_bouldin_scores, marker='^', linewidth=2,
         markersize=8, color='orange')
ax3.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
            label=f'Optimal K={optimal_k}')
ax3.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
ax3.set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(k_range)

# --------------------- Plot 4: Income vs Spending Score ---------------------
ax4 = plt.subplot(3, 3, 4)
scatter = ax4.scatter(df['annual_income'], df['spending_score'],
                      c=df['cluster'], cmap='viridis', alpha=0.6,
                      edgecolors='black', linewidth=0.5, s=50)
# Plot centroids (unscaled back to original space)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax4.scatter(centroids_original[:, 0], centroids_original[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2,
            label='Centroids', zorder=10)
ax4.set_xlabel('Annual Income ($)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Spending Score', fontsize=11, fontweight='bold')
ax4.set_title('Income vs Spending Score', fontweight='bold', fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Cluster')

# --------------------- Plot 5: Purchase Frequency vs Transaction Value ---------------------
ax5 = plt.subplot(3, 3, 5)
scatter2 = ax5.scatter(df['purchase_frequency'], df['avg_transaction_value'],
                       c=df['cluster'], cmap='plasma', alpha=0.6,
                       edgecolors='black', linewidth=0.5, s=50)
ax5.scatter(centroids_original[:, 2], centroids_original[:, 3],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2,
            label='Centroids', zorder=10)
ax5.set_xlabel('Purchase Frequency (per year)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Avg Transaction Value ($)', fontsize=11, fontweight='bold')
ax5.set_title('Purchase Behavior', fontweight='bold', fontsize=13)
ax5.legend()
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax5, label='Cluster')

# --------------------- Plot 6: Cluster Distribution ---------------------
ax6 = plt.subplot(3, 3, 6)
cluster_counts = df['cluster'].value_counts().sort_index()
colors_bar = plt.cm.viridis(np.linspace(0, 1, optimal_k))
bars = ax6.bar(cluster_counts.index, cluster_counts.values,
               color=colors_bar, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax6.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
ax6.set_title('Cluster Size Distribution', fontweight='bold', fontsize=13)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xticks(range(optimal_k))

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    percentage = (height / len(df)) * 100
    ax6.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height)}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# --------------------- Plot 7: Income Distribution by Cluster ---------------------
ax7 = plt.subplot(3, 3, 7)
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]['annual_income']
    ax7.hist(cluster_data, bins=20, alpha=0.5, label=f'Cluster {cluster}',
             edgecolor='black')
ax7.set_xlabel('Annual Income ($)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7.set_title('Income Distribution by Cluster', fontweight='bold', fontsize=13)
ax7.legend()
ax7.grid(True, alpha=0.3)

# --------------------- Plot 8: Spending Score by Cluster ---------------------
ax8 = plt.subplot(3, 3, 8)
cluster_spending = [df[df['cluster'] == i]['spending_score'].values
                    for i in range(optimal_k)]
bp = ax8.boxplot(cluster_spending, labels=[f'C{i}' for i in range(optimal_k)],
                 patch_artist=True, showmeans=True)

# Color the boxes
for patch, color in zip(bp['boxes'], colors_bar):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax8.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax8.set_ylabel('Spending Score', fontsize=11, fontweight='bold')
ax8.set_title('Spending Score by Cluster', fontweight='bold', fontsize=13)
ax8.grid(True, alpha=0.3, axis='y')

# --------------------- Plot 9: PCA Visualization ---------------------
ax9 = plt.subplot(3, 3, 9)
# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

scatter3 = ax9.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'],
                       cmap='coolwarm', alpha=0.6, edgecolors='black',
                       linewidth=0.5, s=50)

# Transform centroids to PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax9.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2,
            label='Centroids', zorder=10)

ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)',
               fontsize=11, fontweight='bold')
ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)',
               fontsize=11, fontweight='bold')
ax9.set_title('PCA Visualization (2D Projection)', fontweight='bold', fontsize=13)
ax9.legend()
ax9.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax9, label='Cluster')

plt.suptitle('Customer Segmentation Analysis - K-Means Clustering',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('kmeans_sklearn_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved!")

# ============================================================================
# 7. CUSTOMER SEGMENT PROFILES
# ============================================================================

print("\n" + "=" * 70)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 70)

segment_names = {
    0: "Budget Shoppers",
    1: "Premium Customers",
    2: "Occasional Buyers",
    3: "Frequent Shoppers"
}

print("\nSegment Characteristics:")
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]
    print(f"\n📊 {segment_names.get(i, f'Segment {i}')}:")
    print(f"   Size: {len(cluster_data)} customers")
    print(f"   Avg Age: {cluster_data['age'].mean():.0f} years")
    print(f"   Avg Income: ${cluster_data['annual_income'].mean():,.0f}")
    print(f"   Avg Spending Score: {cluster_data['spending_score'].mean():.1f}/100")
    print(f"   Purchase Frequency: {cluster_data['purchase_frequency'].mean():.0f}/year")

# ============================================================================
# 8. PREDICT NEW CUSTOMER SEGMENT
# ============================================================================

print("\n" + "=" * 70)
print("PREDICT NEW CUSTOMER SEGMENTS")
print("=" * 70)

# Example new customers
new_customers = pd.DataFrame({
    'age': [25, 55, 40],
    'annual_income': [35000, 120000, 65000],
    'spending_score': [80, 50, 65],
    'purchase_frequency': [30, 10, 20],
    'avg_transaction_value': [50, 250, 120]
})

# Prepare features
new_X = new_customers[['annual_income', 'spending_score',
                       'purchase_frequency', 'avg_transaction_value']].values
new_X_scaled = scaler.transform(new_X)

# Predict clusters
predictions = kmeans.predict(new_X_scaled)

print("\nNew Customer Predictions:")
for i, (_, customer) in enumerate(new_customers.iterrows()):
    cluster = predictions[i]
    print(f"\nCustomer {i + 1}:")
    print(f"  Age: {customer['age']}")
    print(f"  Income: ${customer['annual_income']:,}")
    print(f"  Spending Score: {customer['spending_score']}")
    print(f"  → Assigned to: Cluster {cluster} ({segment_names.get(cluster, 'Unknown')})")

print("\n" + "=" * 70)
print("CLUSTERING COMPLETE!")
print("=" * 70)

plt.show()