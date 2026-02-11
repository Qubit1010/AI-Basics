"""
NEURAL NETWORK FROM SCRATCH WITH NUMPY
=======================================
Building a complete neural network without using any ML libraries
Only NumPy for matrix operations!

Topics:
- Forward propagation
- Backward propagation (backpropagation)
- Activation functions
- Loss functions
- Training loop
- Real classification example
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

print("=" * 80)
print("NEURAL NETWORK FROM SCRATCH - USING ONLY NUMPY")
print("=" * 80)

# ============================================================================
# 1. ACTIVATION FUNCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("1. ACTIVATION FUNCTIONS")
print("=" * 80)

print("\nExplanation:")
print("  Activation functions introduce non-linearity to the network")
print("  Without them, the network would just be linear regression")


class ActivationFunctions:
    """Collection of activation functions and their derivatives"""

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid: 1 / (1 + e^(-x))
        Output range: (0, 1)
        Used for binary classification
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        """
        ReLU: max(0, x)
        Most popular activation for hidden layers
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        """
        Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
        Output range: (-1, 1)
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh: 1 - tanh^2(x)"""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x):
        """
        Softmax: converts logits to probabilities
        Used for multi-class classification
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


print("\n✓ Activation functions implemented:")
print("  • Sigmoid: Binary classification output")
print("  • ReLU: Hidden layer activation")
print("  • Tanh: Alternative to sigmoid")
print("  • Softmax: Multi-class output")

# ============================================================================
# 2. NEURAL NETWORK CLASS
# ============================================================================
print("\n" + "=" * 80)
print("2. BUILDING THE NEURAL NETWORK CLASS")
print("=" * 80)


class NeuralNetwork:
    """
    A flexible neural network implementation from scratch

    Architecture:
    - Input layer
    - Multiple hidden layers (configurable)
    - Output layer
    """

    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        """
        Initialize the neural network

        Parameters:
        -----------
        layer_sizes : list
            Number of neurons in each layer [input, hidden1, hidden2, ..., output]
            Example: [2, 4, 3, 1] -> input:2, hidden:4,3, output:1
        activation : str
            Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
        learning_rate : float
            Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.activation_name = activation

        # Set activation function
        if activation == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif activation == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # Xavier/He initialization
        for i in range(self.num_layers - 1):
            # He initialization (good for ReLU)
            if activation == 'relu':
                w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier initialization (good for sigmoid/tanh)
                w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1.0 / layer_sizes[i])

            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

        # For storing values during forward pass (needed for backprop)
        self.z_values = []  # Linear combinations (before activation)
        self.activations = []  # Activations (after activation function)

        # Training history
        self.loss_history = []
        self.accuracy_history = []

    def forward(self, X):
        """
        Forward propagation

        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        output : numpy array
            Network predictions
        """
        # Clear previous values
        self.z_values = []
        self.activations = [X]

        # Propagate through layers
        current_activation = X

        for i in range(self.num_layers - 1):
            # Linear combination: z = W^T * a + b
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            # Apply activation function
            if i < self.num_layers - 2:
                # Hidden layers
                current_activation = self.activation(z)
            else:
                # Output layer - use sigmoid for binary classification
                current_activation = ActivationFunctions.sigmoid(z)

            self.activations.append(current_activation)

        return current_activation

    def backward(self, X, y):
        """
        Backward propagation (backpropagation)

        Computes gradients and updates weights and biases

        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Input data
        y : numpy array, shape (n_samples, 1)
            True labels
        """
        m = X.shape[0]  # Number of samples

        # Initialize gradients
        dW = [None] * (self.num_layers - 1)
        db = [None] * (self.num_layers - 1)

        # Output layer error
        # For binary classification with sigmoid output
        delta = self.activations[-1] - y

        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients
            dW[i] = np.dot(self.activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m

            # Propagate error to previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])

        # Update weights and biases
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def compute_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss

        Loss = -1/m * sum(y*log(y_pred) + (1-y)*log(1-y_pred))
        """
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network

        Parameters:
        -----------
        X : numpy array
            Training data
        y : numpy array
            Training labels
        epochs : int
            Number of training iterations
        verbose : bool
            Print progress
        """
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)

            # Compute accuracy
            predictions = (output >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            self.accuracy_history.append(accuracy)

            # Backward propagation
            self.backward(X, y)

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

    def predict(self, X):
        """Make predictions (probabilities)"""
        return self.forward(X)

    def predict_classes(self, X):
        """Make binary predictions (0 or 1)"""
        probabilities = self.predict(X)
        return (probabilities >= 0.5).astype(int)


print("\n✓ Neural Network class implemented!")
print("\nKey Components:")
print("  • __init__: Initialize weights and biases")
print("  • forward: Forward propagation")
print("  • backward: Backpropagation and weight updates")
print("  • compute_loss: Binary cross-entropy")
print("  • train: Complete training loop")
print("  • predict: Make predictions")

# ============================================================================
# 3. HOW IT WORKS - MATHEMATICAL EXPLANATION
# ============================================================================
print("\n" + "=" * 80)
print("3. HOW NEURAL NETWORKS WORK")
print("=" * 80)

explanation = """
FORWARD PROPAGATION
===================
For each layer l:
  1. Linear combination: z[l] = W[l] * a[l-1] + b[l]
  2. Activation: a[l] = activation(z[l])

Where:
  - W[l] = weights for layer l
  - b[l] = biases for layer l
  - a[l-1] = activations from previous layer
  - z[l] = pre-activation values
  - a[l] = post-activation values

BACKWARD PROPAGATION (BACKPROPAGATION)
=======================================
Calculate gradients by chain rule:

  1. Output layer error:
     delta[L] = (a[L] - y)  (for sigmoid + cross-entropy)

  2. Hidden layer errors (propagate backwards):
     delta[l] = (delta[l+1] * W[l+1]^T) * activation'(z[l])

  3. Compute gradients:
     dW[l] = a[l-1]^T * delta[l] / m
     db[l] = sum(delta[l]) / m

  4. Update weights:
     W[l] = W[l] - learning_rate * dW[l]
     b[l] = b[l] - learning_rate * db[l]

LOSS FUNCTION (Binary Cross-Entropy)
=====================================
  Loss = -1/m * sum(y*log(y_pred) + (1-y)*log(1-y_pred))

This measures how far predictions are from true labels.
Goal: Minimize this loss!

GRADIENT DESCENT
================
Iteratively update weights in the direction that reduces loss:
  W = W - learning_rate * gradient

Learning rate controls step size (typically 0.001 to 0.1)
"""

print(explanation)

# ============================================================================
# 4. GENERATE SYNTHETIC DATA
# ============================================================================
print("\n" + "=" * 80)
print("4. GENERATE SYNTHETIC DATA")
print("=" * 80)

np.random.seed(42)


# Create XOR-like pattern (non-linearly separable)
def generate_data(n_samples=1000):
    """Generate synthetic binary classification data"""
    X = np.random.randn(n_samples, 2)

    # Create circular decision boundary
    # Points inside circle are class 1, outside are class 0
    radius = 1.5
    distances = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = (distances < radius).astype(int).reshape(-1, 1)

    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]

    return X, y


# Generate training and test data
X_train, y_train = generate_data(800)
X_test, y_test = generate_data(200)

print(f"\n✓ Data generated:")
print(f"  Training set: {X_train.shape}")
print(f"  Test set: {X_test.shape}")
print(f"\nClass distribution (training):")
print(f"  Class 0: {np.sum(y_train == 0)} samples")
print(f"  Class 1: {np.sum(y_train == 1)} samples")

# ============================================================================
# 5. CREATE AND TRAIN NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 80)
print("5. CREATE AND TRAIN NEURAL NETWORK")
print("=" * 80)

# Create network
# Architecture: Input(2) -> Hidden(8) -> Hidden(4) -> Output(1)
nn = NeuralNetwork(
    layer_sizes=[2, 8, 4, 1],
    activation='relu',
    learning_rate=0.1
)

print("\nNetwork Architecture:")
print(f"  Input layer: 2 neurons")
print(f"  Hidden layer 1: 8 neurons (ReLU)")
print(f"  Hidden layer 2: 4 neurons (ReLU)")
print(f"  Output layer: 1 neuron (Sigmoid)")
print(f"\nTotal parameters:")
total_params = sum(w.size for w in nn.weights) + sum(b.size for b in nn.biases)
print(f"  {total_params} trainable parameters")

# Train
print("\n--- Training Neural Network ---")
nn.train(X_train, y_train, epochs=1000, verbose=True)

# Evaluate
train_predictions = nn.predict_classes(X_train)
train_accuracy = np.mean(train_predictions == y_train)

test_predictions = nn.predict_classes(X_test)
test_accuracy = np.mean(test_predictions == y_test)

print(f"\n✓ Training complete!")
print(f"  Training accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
print(f"  Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# ============================================================================
# 6. VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("6. CREATE VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Training history
print("\n📊 Creating training history plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(nn.loss_history, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(nn.accuracy_history, 'g-', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('nn_viz_1_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: nn_viz_1_training_history.png")

# Visualization 2: Decision boundary
print("\n📊 Creating decision boundary visualization...")

# Create mesh
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 5))

# Plot decision boundary
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=20)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(),
            cmap='RdYlBu', edgecolors='black', s=50, alpha=0.7)
plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
plt.title('Decision Boundary - Training Data', fontsize=14, fontweight='bold')
plt.colorbar(label='Probability')
plt.grid(True, alpha=0.3)

# Test data
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=20)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(),
            cmap='RdYlBu', edgecolors='black', s=50, alpha=0.7)
plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
plt.title('Decision Boundary - Test Data', fontsize=14, fontweight='bold')
plt.colorbar(label='Probability')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nn_viz_2_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: nn_viz_2_decision_boundary.png")

# Visualization 3: Activation function comparisons
print("\n📊 Creating activation functions comparison...")

x_vals = np.linspace(-5, 5, 100)
sigmoid_vals = ActivationFunctions.sigmoid(x_vals)
relu_vals = ActivationFunctions.relu(x_vals)
tanh_vals = ActivationFunctions.tanh(x_vals)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Activations
axes[0, 0].plot(x_vals, sigmoid_vals, 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoid', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

axes[0, 1].plot(x_vals, relu_vals, 'r-', linewidth=2)
axes[0, 1].set_title('ReLU', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

axes[0, 2].plot(x_vals, tanh_vals, 'g-', linewidth=2)
axes[0, 2].set_title('Tanh', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Derivatives
sigmoid_der = ActivationFunctions.sigmoid_derivative(x_vals)
relu_der = ActivationFunctions.relu_derivative(x_vals)
tanh_der = ActivationFunctions.tanh_derivative(x_vals)

axes[1, 0].plot(x_vals, sigmoid_der, 'b-', linewidth=2)
axes[1, 0].set_title('Sigmoid Derivative', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

axes[1, 1].plot(x_vals, relu_der, 'r-', linewidth=2)
axes[1, 1].set_title('ReLU Derivative', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

axes[1, 2].plot(x_vals, tanh_der, 'g-', linewidth=2)
axes[1, 2].set_title('Tanh Derivative', fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.suptitle('Activation Functions and Their Derivatives', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('nn_viz_3_activations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: nn_viz_3_activations.png")

# ============================================================================
# 7. TEST WITH DIFFERENT ARCHITECTURES
# ============================================================================
print("\n" + "=" * 80)
print("7. COMPARE DIFFERENT ARCHITECTURES")
print("=" * 80)

architectures = [
    ([2, 4, 1], "Shallow (4 hidden)"),
    ([2, 8, 4, 1], "Medium (8, 4 hidden)"),
    ([2, 16, 8, 4, 1], "Deep (16, 8, 4 hidden)")
]

results = []

for arch, name in architectures:
    print(f"\n--- Testing {name} ---")
    nn_test = NeuralNetwork(arch, activation='relu', learning_rate=0.1)
    nn_test.train(X_train, y_train, epochs=500, verbose=False)

    test_pred = nn_test.predict_classes(X_test)
    test_acc = np.mean(test_pred == y_test)

    results.append((name, test_acc, nn_test.loss_history[-1]))
    print(f"  Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Final loss: {nn_test.loss_history[-1]:.4f}")

# Compare results
print("\n--- Architecture Comparison ---")
print(f"{'Architecture':<30} {'Test Accuracy':<15} {'Final Loss':<15}")
print("-" * 60)
for name, acc, loss in results:
    print(f"{name:<30} {acc:<15.4f} {loss:<15.4f}")

# ============================================================================
# 8. SUMMARY AND KEY CONCEPTS
# ============================================================================
print("\n" + "=" * 80)
print("8. SUMMARY")
print("=" * 80)

summary = f"""
NEURAL NETWORK FROM SCRATCH - COMPLETED!
=========================================

What We Built:
--------------
✓ Complete neural network using only NumPy
✓ Forward propagation
✓ Backward propagation (backpropagation)
✓ Multiple activation functions (ReLU, Sigmoid, Tanh)
✓ Binary cross-entropy loss
✓ Gradient descent optimization
✓ Flexible architecture (any number of layers)

Results:
--------
• Training accuracy: {train_accuracy * 100:.2f}%
• Test accuracy: {test_accuracy * 100:.2f}%
• Final loss: {nn.loss_history[-1]:.4f}

Architecture Used:
------------------
Input (2) -> Hidden (8, ReLU) -> Hidden (4, ReLU) -> Output (1, Sigmoid)
Total parameters: {total_params}

Key Concepts:
-------------
1. Forward Propagation:
   - Pass data through layers
   - Apply weights, biases, and activations

2. Backward Propagation:
   - Calculate gradients using chain rule
   - Propagate errors backwards
   - Update weights and biases

3. Activation Functions:
   - Introduce non-linearity
   - ReLU: max(0, x)
   - Sigmoid: 1/(1 + e^-x)

4. Loss Function:
   - Binary cross-entropy for classification
   - Measures prediction error

5. Gradient Descent:
   - Iteratively update weights
   - Move in direction that reduces loss

Mathematics:
------------
• Forward: a[l] = activation(W[l] * a[l-1] + b[l])
• Backward: delta[l] = (delta[l+1] * W[l+1]^T) * activation'(z[l])
• Update: W = W - learning_rate * gradient

Files Generated:
-----------------
• nn_viz_1_training_history.png - Loss and accuracy curves
• nn_viz_2_decision_boundary.png - Classification regions
• nn_viz_3_activations.png - Activation function plots
"""

print(summary)

print("\n" + "=" * 80)
print("NEURAL NETWORK IMPLEMENTATION COMPLETE!")
print("=" * 80)
print("\nYou now understand how neural networks work at the lowest level!")
print("This is exactly what frameworks like TensorFlow and PyTorch do,")
print("but optimized and with many more features.")
print("=" * 80)