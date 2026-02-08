"""
K-Means Clustering - Implementation from Scratch
Using only NumPy and Matplotlib (no scikit-learn)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
np.random.seed(42)


class KMeans:
    """K-Means Clustering Algorithm from Scratch"""

    def __init__(self, n_clusters=3, max_iterations=100, tolerance=1e-4):
        """
        Initialize K-Means

        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence threshold
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.history = []  # Store history for visualization

    def initialize_centroids(self, X):
        """Randomly initialize centroids from data points"""
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        return self.centroids

    def compute_distance(self, X, centroids):
        """
        Compute Euclidean distance between points and centroids

        Returns: distance matrix of shape (n_samples, n_clusters)
        """
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i, centroid in enumerate(centroids):
            # Euclidean distance: sqrt(sum((x - centroid)^2))
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

        return distances

    def assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid"""
        distances = self.compute_distance(X, centroids)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            # Get all points assigned to cluster k
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                # Calculate mean of cluster points
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep old centroid or reinitialize
                new_centroids[k] = self.centroids[k]

        return new_centroids

    def has_converged(self, old_centroids, new_centroids):
        """Check if centroids have converged"""
        distances = np.sqrt(np.sum((new_centroids - old_centroids) ** 2, axis=1))
        return np.all(distances < self.tolerance)

    def fit(self, X):
        """
        Fit K-Means to data

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)

        # Store initial state
        self.history = [{
            'centroids': self.centroids.copy(),
            'labels': np.zeros(X.shape[0], dtype=int)
        }]

        for iteration in range(self.max_iterations):
            # Assign points to nearest centroid
            self.labels = self.assign_clusters(X, self.centroids)

            # Update centroids
            new_centroids = self.update_centroids(X, self.labels)

            # Store iteration state
            self.history.append({
                'centroids': new_centroids.copy(),
                'labels': self.labels.copy()
            })

            # Check convergence
            if self.has_converged(self.centroids, new_centroids):
                print(f"Converged at iteration {iteration + 1}")
                break

            self.centroids = new_centroids
        else:
            print(f"Reached maximum iterations ({self.max_iterations})")

        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X, self.centroids)

    def compute_inertia(self, X):
        """
        Compute within-cluster sum of squares (inertia)
        Lower is better
        """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia


# ============================================================================
# GENERATE SAMPLE DATA
# ============================================================================

def generate_blobs(n_samples=300, n_features=2, n_centers=3):
    """Generate synthetic clustered data"""
    X = []
    y = []

    centers = np.random.randn(n_centers, n_features) * 5

    for i, center in enumerate(centers):
        # Generate points around each center
        points = np.random.randn(n_samples // n_centers, n_features) + center
        X.append(points)
        y.extend([i] * (n_samples // n_centers))

    X = np.vstack(X)
    y = np.array(y)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


print("=" * 70)
print("K-MEANS CLUSTERING - FROM SCRATCH")
print("=" * 70)

# Generate data
n_clusters = 4
X, true_labels = generate_blobs(n_samples=400, n_features=2, n_centers=n_clusters)

print(f"\nDataset Shape: {X.shape}")
print(f"Number of clusters: {n_clusters}")

# ============================================================================
# TRAIN K-MEANS
# ============================================================================

print("\nTraining K-Means...")
kmeans = KMeans(n_clusters=n_clusters, max_iterations=100)
kmeans.fit(X)

predicted_labels = kmeans.labels
inertia = kmeans.compute_inertia(X)

print(f"\nFinal Inertia (WCSS): {inertia:.2f}")
print(f"Number of iterations: {len(kmeans.history) - 1}")

# ============================================================================
# ELBOW METHOD - Find Optimal K
# ============================================================================

print("\nCalculating Elbow Method...")
k_range = range(1, 10)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, max_iterations=100)
    km.fit(X)
    inertias.append(km.compute_inertia(X))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# --------------------- Plot 1: Original Data (if available) ---------------------
ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis',
                       alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Feature 1', fontsize=11)
ax1.set_ylabel('Feature 2', fontsize=11)
ax1.set_title('True Clusters (Ground Truth)', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='True Cluster')

# --------------------- Plot 2: K-Means Result ---------------------
ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='plasma',
                       alpha=0.6, edgecolors='black', linewidth=0.5)
# Plot centroids
ax2.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2,
            label='Centroids', zorder=10)
ax2.set_xlabel('Feature 1', fontsize=11)
ax2.set_ylabel('Feature 2', fontsize=11)
ax2.set_title('K-Means Clustering Result', fontweight='bold', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Predicted Cluster')

# --------------------- Plot 3: Cluster Iterations ---------------------
ax3 = plt.subplot(2, 3, 3)
# Show initial, middle, and final states
iterations_to_show = [0, len(kmeans.history) // 2, -1]
colors = ['red', 'orange', 'green']
markers = ['o', 's', 'X']

for idx, (iter_idx, color, marker) in enumerate(zip(iterations_to_show, colors, markers)):
    centroids = kmeans.history[iter_idx]['centroids']
    label = f"Iteration {iter_idx if iter_idx >= 0 else len(kmeans.history) - 1}"
    ax3.scatter(centroids[:, 0], centroids[:, 1],
                c=color, marker=marker, s=200, edgecolors='black',
                linewidth=2, label=label, alpha=0.7, zorder=5 - idx)

# Plot data points lightly
ax3.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.3, s=20)
ax3.set_xlabel('Feature 1', fontsize=11)
ax3.set_ylabel('Feature 2', fontsize=11)
ax3.set_title('Centroid Evolution', fontweight='bold', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)

# --------------------- Plot 4: Elbow Method ---------------------
ax4 = plt.subplot(2, 3, 4)
ax4.plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='blue')
ax4.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2,
            label=f'Chosen K={n_clusters}')
ax4.set_xlabel('Number of Clusters (K)', fontsize=11)
ax4.set_ylabel('Inertia (WCSS)', fontsize=11)
ax4.set_title('Elbow Method - Optimal K', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xticks(k_range)

# --------------------- Plot 5: Cluster Sizes ---------------------
ax5 = plt.subplot(2, 3, 5)
unique_labels, counts = np.unique(predicted_labels, return_counts=True)
colors_bar = plt.cm.plasma(np.linspace(0, 1, n_clusters))
bars = ax5.bar(unique_labels, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Cluster ID', fontsize=11)
ax5.set_ylabel('Number of Points', fontsize=11)
ax5.set_title('Cluster Size Distribution', fontweight='bold', fontsize=13)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticks(unique_labels)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# --------------------- Plot 6: Distance to Centroids ---------------------
ax6 = plt.subplot(2, 3, 6)
distances_to_centroids = kmeans.compute_distance(X, kmeans.centroids)
min_distances = np.min(distances_to_centroids, axis=1)

ax6.hist(min_distances, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax6.axvline(x=np.mean(min_distances), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(min_distances):.2f}')
ax6.set_xlabel('Distance to Nearest Centroid', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Distribution of Distances to Centroids', fontweight='bold', fontsize=13)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('kmeans_clustering_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Main visualization saved!")

# ============================================================================
# ANIMATION - Show clustering process
# ============================================================================

print("\nCreating animation of clustering process...")

fig_anim, ax_anim = plt.subplots(figsize=(10, 8))


def animate(frame):
    ax_anim.clear()

    # Get data for this iteration
    state = kmeans.history[frame]
    centroids = state['centroids']
    labels = state['labels']

    # Plot data points
    scatter = ax_anim.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma',
                              alpha=0.6, edgecolors='black', linewidth=0.5, s=50)

    # Plot centroids
    ax_anim.scatter(centroids[:, 0], centroids[:, 1],
                    c='red', marker='X', s=400, edgecolors='black',
                    linewidth=3, zorder=10, label='Centroids')

    ax_anim.set_xlabel('Feature 1', fontsize=12)
    ax_anim.set_ylabel('Feature 2', fontsize=12)
    ax_anim.set_title(f'K-Means Iteration {frame}/{len(kmeans.history) - 1}',
                      fontweight='bold', fontsize=14)
    ax_anim.legend(loc='upper right')
    ax_anim.grid(True, alpha=0.3)

    return scatter,


# Create animation (show first few and last iterations)
frames_to_show = min(10, len(kmeans.history))
anim = FuncAnimation(fig_anim, animate, frames=frames_to_show,
                     interval=500, blit=False, repeat=True)

plt.tight_layout()
anim.save('kmeans_animation.gif', writer='pillow', fps=2)
print("✓ Animation saved!")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("CLUSTERING STATISTICS")
print("=" * 70)

for k in range(n_clusters):
    cluster_points = X[predicted_labels == k]
    print(f"\nCluster {k}:")
    print(f"  Size: {len(cluster_points)}")
    print(f"  Centroid: [{kmeans.centroids[k][0]:.2f}, {kmeans.centroids[k][1]:.2f}]")
    print(f"  Variance: {np.var(cluster_points, axis=0)}")

# ============================================================================
# EXAMPLE: PREDICT NEW POINTS
# ============================================================================

print("\n" + "=" * 70)
print("PREDICTION ON NEW POINTS")
print("=" * 70)

new_points = np.array([
    [0, 0],
    [5, 5],
    [-5, -5],
    [10, -10]
])

predictions = kmeans.predict(new_points)

print("\nNew Points and Their Predicted Clusters:")
for i, (point, cluster) in enumerate(zip(new_points, predictions)):
    print(f"Point {i + 1}: {point} → Cluster {cluster}")

print("\n" + "=" * 70)
print("CLUSTERING COMPLETE!")
print("=" * 70)

plt.show()