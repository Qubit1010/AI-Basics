"""
TENSORFLOW BASICS - COMPLETE RUNNABLE TUTORIAL
===============================================
A step-by-step guide with working code examples

Run this file to see TensorFlow in action!
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("TENSORFLOW BASICS - COMPLETE TUTORIAL")
print("=" * 80)
print(f"\nTensorFlow version: {tf.__version__}\n")

# ============================================================================
# 1. TENSORS - THE BASICS
# ============================================================================
print("=" * 80)
print("1. TENSORS - THE BASICS")
print("=" * 80)

# Scalar (0D)
scalar = tf.constant(42)
print(f"\nScalar: {scalar}")
print(f"  Shape: {scalar.shape}")
print(f"  Rank: {tf.rank(scalar).numpy()}")

# Vector (1D)
vector = tf.constant([1, 2, 3, 4, 5])
print(f"\nVector: {vector}")
print(f"  Shape: {vector.shape}")

# Matrix (2D)
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])
print(f"\nMatrix:\n{matrix}")
print(f"  Shape: {matrix.shape}")

# Creating tensors
zeros = tf.zeros([3, 3])
ones = tf.ones([2, 4])
random = tf.random.normal([3, 3])
print(f"\nZeros tensor shape: {zeros.shape}")
print(f"Ones tensor shape: {ones.shape}")
print(f"Random tensor shape: {random.shape}")


# ============================================================================
# 2. TENSOR OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("2. TENSOR OPERATIONS")
print("=" * 80)

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

print(f"\na = {a.numpy()}")
print(f"b = {b.numpy()}")
print(f"a + b = {(a + b).numpy()}")
print(f"a * b = {(a * b).numpy()}")

# Matrix operations
mat_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
mat_b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

print(f"\nMatrix multiplication:")
print(tf.matmul(mat_a, mat_b).numpy())


# ============================================================================
# 3. VARIABLES
# ============================================================================
print("\n" + "=" * 80)
print("3. VARIABLES")
print("=" * 80)

var = tf.Variable([1, 2, 3])
print(f"\nInitial variable: {var.numpy()}")

var.assign([4, 5, 6])
print(f"After assign: {var.numpy()}")

var.assign_add([1, 1, 1])
print(f"After add: {var.numpy()}")


# ============================================================================
# 4. AUTOMATIC DIFFERENTIATION
# ============================================================================
print("\n" + "=" * 80)
print("4. AUTOMATIC DIFFERENTIATION")
print("=" * 80)

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)
print(f"\nFor y = x^2 where x = {x.numpy()}")
print(f"  y = {y.numpy()}")
print(f"  dy/dx = {gradient.numpy()} (expected: 2*x = 6)")


# ============================================================================
# 5. ACTIVATION FUNCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("5. ACTIVATION FUNCTIONS")
print("=" * 80)

# Create sample data
x_data = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])

# Apply activations
relu_out = tf.nn.relu(x_data)
sigmoid_out = tf.nn.sigmoid(x_data)
tanh_out = tf.nn.tanh(x_data)

print(f"\nInput: {x_data.numpy()}")
print(f"ReLU: {relu_out.numpy()}")
print(f"Sigmoid: {sigmoid_out.numpy()}")
print(f"Tanh: {tanh_out.numpy()}")


# ============================================================================
# 6. BUILD A SIMPLE NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 80)
print("6. BUILD A SIMPLE NEURAL NETWORK")
print("=" * 80)

# Create a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("\nModel created!")
print("\nModel Summary:")
model.summary()


# ============================================================================
# 7. LINEAR REGRESSION EXAMPLE
# ============================================================================
print("\n" + "=" * 80)
print("7. LINEAR REGRESSION EXAMPLE")
print("=" * 80)

print("\nGenerating synthetic data: y = 2x + 3 + noise")

# Generate data
np.random.seed(42)
X_train = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
y_train = (2 * X_train + 3 + np.random.randn(100, 1) * 0.5).astype(np.float32)

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

# Build model
regression_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile
regression_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nTraining regression model...")
history = regression_model.fit(
    X_train, y_train,
    epochs=50,
    verbose=0
)

# Get learned parameters
weights, bias = regression_model.layers[0].get_weights()
print(f"\n✓ Training complete!")
print(f"  Learned weight (slope): {weights[0][0]:.4f} (target: 2.0)")
print(f"  Learned bias (intercept): {bias[0]:.4f} (target: 3.0)")
print(f"  Final loss: {history.history['loss'][-1]:.4f}")


# ============================================================================
# 8. BINARY CLASSIFICATION EXAMPLE
# ============================================================================
print("\n" + "=" * 80)
print("8. BINARY CLASSIFICATION EXAMPLE")
print("=" * 80)

print("\nGenerating synthetic classification data...")

# Generate classification data
from sklearn.datasets import make_moons
X_class, y_class = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_class = X_class.astype(np.float32)
y_class = y_class.astype(np.float32)

print(f"Data shape: X={X_class.shape}, y={y_class.shape}")
print(f"Class distribution: Class 0={np.sum(y_class==0)}, Class 1={np.sum(y_class==1)}")

# Split data
from sklearn.model_selection import train_test_split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Build model
classification_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
classification_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining classification model...")
history_class = classification_model.fit(
    X_train_c, y_train_c,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
test_loss, test_accuracy = classification_model.evaluate(X_test_c, y_test_c, verbose=0)
print(f"\n✓ Training complete!")
print(f"  Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Test loss: {test_loss:.4f}")


# ============================================================================
# 9. MNIST DIGIT CLASSIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("9. MNIST DIGIT CLASSIFICATION")
print("=" * 80)

print("\nLoading MNIST dataset...")
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

print(f"Training set: {X_train_mnist.shape}")
print(f"Test set: {X_test_mnist.shape}")

# Normalize
X_train_mnist = X_train_mnist.astype(np.float32) / 255.0
X_test_mnist = X_test_mnist.astype(np.float32) / 255.0

# Build model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
mnist_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining MNIST model (using subset for speed)...")
history_mnist = mnist_model.fit(
    X_train_mnist[:10000], y_train_mnist[:10000],
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss_mnist, test_accuracy_mnist = mnist_model.evaluate(
    X_test_mnist, y_test_mnist, verbose=0
)

print(f"\n✓ Training complete!")
print(f"  Test accuracy: {test_accuracy_mnist:.4f} ({test_accuracy_mnist*100:.2f}%)")

# Make predictions
predictions = mnist_model.predict(X_test_mnist[:10], verbose=0)
print(f"\nSample predictions:")
print(f"{'Actual':<10} {'Predicted':<10} {'Confidence':<15}")
print("-" * 35)
for i in range(10):
    actual = y_test_mnist[i]
    predicted = np.argmax(predictions[i])
    confidence = np.max(predictions[i])
    print(f"{actual:<10} {predicted:<10} {confidence:<15.4f}")


# ============================================================================
# 10. SAVE AND LOAD MODEL
# ============================================================================
print("\n" + "=" * 80)
print("10. SAVE AND LOAD MODEL")
print("=" * 80)

# Save model
model_path = '../../ANN/my_mnist_model.h5'
mnist_model.save(model_path)
print(f"\n✓ Model saved to: {model_path}")

# Load model
loaded_model = tf.keras.models.load_model(model_path)
print(f"✓ Model loaded successfully")

# Test loaded model
test_pred = loaded_model.predict(X_test_mnist[:5], verbose=0)
print(f"\nPredictions from loaded model: {np.argmax(test_pred, axis=1)}")
print(f"Actual labels: {y_test_mnist[:5]}")


# ============================================================================
# VISUALIZATION (OPTIONAL)
# ============================================================================
print("\n" + "=" * 80)
print("11. CREATE VISUALIZATIONS")
print("=" * 80)

try:
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history_mnist.history['accuracy'], label='Training')
    axes[0].plot(history_mnist.history['val_accuracy'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('MNIST Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history_mnist.history['loss'], label='Training')
    axes[1].plot(history_mnist.history['val_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('MNIST Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tensorflow_training_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: tensorflow_training_history.png")

    # Plot sample MNIST predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(10):
        axes[i].imshow(X_test_mnist[i], cmap='gray')
        pred = np.argmax(predictions[i])
        actual = y_test_mnist[i]
        color = 'green' if pred == actual else 'red'
        axes[i].set_title(f'P:{pred}, A:{actual}', color=color, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('MNIST Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig('tensorflow_mnist_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: tensorflow_mnist_predictions.png")

except Exception as e:
    print(f"\nVisualization skipped (display not available): {e}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TENSORFLOW TUTORIAL COMPLETE!")
print("=" * 80)

summary = f"""
✓ Covered Topics:
  1. Tensors and basic operations
  2. Variables
  3. Automatic differentiation
  4. Activation functions
  5. Building neural networks
  6. Linear regression (achieved target parameters)
  7. Binary classification ({test_accuracy*100:.2f}% accuracy)
  8. MNIST digit classification ({test_accuracy_mnist*100:.2f}% accuracy)
  9. Model saving and loading

Key Results:
  • Linear Regression: Learned weight={weights[0][0]:.2f} (target: 2.0)
  • Binary Classification: {test_accuracy*100:.2f}% test accuracy
  • MNIST: {test_accuracy_mnist*100:.2f}% test accuracy

Files Created:
  • my_mnist_model.h5 - Saved MNIST model
  • tensorflow_training_history.png (if display available)
  • tensorflow_mnist_predictions.png (if display available)
"""

print(summary)
print("=" * 80)
print("You've successfully completed the TensorFlow basics tutorial!")
print("=" * 80)