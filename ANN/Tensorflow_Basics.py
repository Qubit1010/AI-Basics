"""
TENSORFLOW BASICS - COMPREHENSIVE GUIDE
========================================

This tutorial covers fundamental TensorFlow concepts with code examples.
Note: TensorFlow requires installation: pip install tensorflow

Author: AI Tutorial
Date: 2026
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================
"""
1. Introduction to TensorFlow
2. Tensors - The Foundation
3. Tensor Operations
4. Variables
5. Automatic Differentiation
6. Building Neural Networks
7. Activation Functions
8. Loss Functions and Optimizers
9. Training Models
10. Practical Examples
"""

# ============================================================================
# 1. INTRODUCTION TO TENSORFLOW
# ============================================================================
"""
WHAT IS TENSORFLOW?
-------------------
TensorFlow is an open-source machine learning framework developed by Google.
It's used for:
  • Deep Learning
  • Neural Networks
  • Machine Learning models
  • Production deployment

KEY FEATURES:
  ✓ Automatic differentiation (for backpropagation)
  ✓ GPU acceleration
  ✓ High-level APIs (Keras)
  ✓ Model deployment tools
  ✓ Large community and ecosystem

INSTALLATION:
  pip install tensorflow

BASIC IMPORT:
"""
import tensorflow as tf
import numpy as np

# Check version
print(tf.__version__)

# ============================================================================
# 2. TENSORS - THE FOUNDATION
# ============================================================================
"""
TENSORS
-------
Tensors are multi-dimensional arrays (similar to NumPy arrays).
They are the fundamental data structure in TensorFlow.

TENSOR RANKS:
  • Rank 0: Scalar (just a number)
  • Rank 1: Vector (1D array)
  • Rank 2: Matrix (2D array)
  • Rank 3+: Higher dimensional arrays
"""

# Scalar (Rank 0)
scalar = tf.constant(42)
print(f"Scalar: {scalar}")
print(f"Shape: {scalar.shape}")  # Shape: ()
print(f"Rank: {tf.rank(scalar)}")  # Rank: 0

# Vector (Rank 1)
vector = tf.constant([1, 2, 3, 4, 5])
print(f"Vector: {vector}")
print(f"Shape: {vector.shape}")  # Shape: (5,)
print(f"Rank: {tf.rank(vector)}")  # Rank: 1

# Matrix (Rank 2)
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])
print(f"Matrix: {matrix}")
print(f"Shape: {matrix.shape}")  # Shape: (2, 3)
print(f"Rank: {tf.rank(matrix)}")  # Rank: 2

# 3D Tensor (Rank 3)
tensor_3d = tf.constant([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])
print(f"Shape: {tensor_3d.shape}")  # Shape: (2, 2, 2)
print(f"Rank: {tf.rank(tensor_3d)}")  # Rank: 3

"""
CREATING TENSORS
----------------
"""
# Zeros
zeros = tf.zeros([3, 3])  # 3x3 matrix of zeros

# Ones
ones = tf.ones([2, 4])  # 2x4 matrix of ones

# Random values
random_normal = tf.random.normal([3, 3], mean=0, stddev=1)
random_uniform = tf.random.uniform([2, 2], minval=0, maxval=1)

# From numpy array
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = tf.constant(numpy_array)

# Convert tensor to numpy
numpy_from_tensor = tensor_from_numpy.numpy()

# ============================================================================
# 3. TENSOR OPERATIONS
# ============================================================================
"""
BASIC ARITHMETIC
----------------
"""
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Element-wise operations
add = a + b  # or tf.add(a, b)
subtract = a - b  # or tf.subtract(a, b)
multiply = a * b  # or tf.multiply(a, b)
divide = a / b  # or tf.divide(a, b)

"""
MATRIX OPERATIONS
-----------------
"""
matrix_a = tf.constant([[1, 2],
                        [3, 4]], dtype=tf.float32)
matrix_b = tf.constant([[5, 6],
                        [7, 8]], dtype=tf.float32)

# Element-wise multiplication
elementwise = matrix_a * matrix_b

# Matrix multiplication (dot product)
matmul = tf.matmul(matrix_a, matrix_b)
# or use @ operator
matmul2 = matrix_a @ matrix_b

# Transpose
transpose = tf.transpose(matrix_a)

"""
RESHAPING
---------
"""
tensor = tf.constant([1, 2, 3, 4, 5, 6])
reshaped_2x3 = tf.reshape(tensor, [2, 3])  # [[1, 2, 3], [4, 5, 6]]
reshaped_3x2 = tf.reshape(tensor, [3, 2])  # [[1, 2], [3, 4], [5, 6]]

"""
AGGREGATION OPERATIONS
----------------------
"""
values = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])

sum_val = tf.reduce_sum(values)  # 15.0
mean_val = tf.reduce_mean(values)  # 3.0
max_val = tf.reduce_max(values)  # 5.0
min_val = tf.reduce_min(values)  # 1.0
std_val = tf.math.reduce_std(values)  # Standard deviation

# Axis-specific operations
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]], dtype=tf.float32)
sum_axis_0 = tf.reduce_sum(matrix, axis=0)  # Sum columns: [5, 7, 9]
sum_axis_1 = tf.reduce_sum(matrix, axis=1)  # Sum rows: [6, 15]

# ============================================================================
# 4. VARIABLES
# ============================================================================
"""
VARIABLES
---------
Variables are mutable tensors used to store model parameters (weights, biases).
Unlike constants, they can be updated during training.
"""

# Create a variable
var = tf.Variable([1, 2, 3])
print(var)

# Update variable
var.assign([4, 5, 6])  # Set new values
var.assign_add([1, 1, 1])  # Add to current values
var.assign_sub([2, 2, 2])  # Subtract from current values

# Variables are trainable by default
print(var.trainable)  # True

# ============================================================================
# 5. AUTOMATIC DIFFERENTIATION
# ============================================================================
"""
GRADIENTS
---------
TensorFlow can automatically compute gradients (derivatives).
This is essential for training neural networks via backpropagation.

GradientTape records operations for automatic differentiation.
"""

# Example 1: Simple derivative
# If y = x^2, then dy/dx = 2x
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

# Compute gradient
gradient = tape.gradient(y, x)
print(f"x = {x.numpy()}")  # 3.0
print(f"y = x^2 = {y.numpy()}")  # 9.0
print(f"dy/dx = 2x = {gradient.numpy()}")  # 6.0

# Example 2: Multiple variables
x = tf.Variable(2.0)
w = tf.Variable(3.0)
b = tf.Variable(1.0)

with tf.GradientTape() as tape:
    y = w * x + b  # Linear function

# Compute gradients with respect to all variables
gradients = tape.gradient(y, [w, b, x])
print(f"dy/dw = {gradients[0].numpy()}")  # 2.0 (value of x)
print(f"dy/db = {gradients[1].numpy()}")  # 1.0
print(f"dy/dx = {gradients[2].numpy()}")  # 3.0 (value of w)

# ============================================================================
# 6. BUILDING NEURAL NETWORKS
# ============================================================================
"""
KERAS API
---------
Keras is TensorFlow's high-level API for building neural networks.

SEQUENTIAL MODEL
----------------
The simplest way to build a model - stack layers sequentially.
"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Model summary
model.summary()

"""
FUNCTIONAL API
--------------
More flexible approach for complex architectures.
"""
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)

"""
COMMON LAYERS
-------------
"""
# Dense (Fully Connected)
dense = tf.keras.layers.Dense(units=32, activation='relu')

# Dropout (for regularization)
dropout = tf.keras.layers.Dropout(rate=0.2)

# Flatten (convert multi-dimensional to 1D)
flatten = tf.keras.layers.Flatten()

# Conv2D (for images)
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')

# MaxPooling (downsample)
pooling = tf.keras.layers.MaxPooling2D(pool_size=2)

# LSTM (for sequences)
lstm = tf.keras.layers.LSTM(units=64)

# ============================================================================
# 7. ACTIVATION FUNCTIONS
# ============================================================================
"""
ACTIVATION FUNCTIONS
--------------------
Introduce non-linearity to neural networks.

COMMON ACTIVATIONS:

1. ReLU (Rectified Linear Unit)
   f(x) = max(0, x)
   • Most popular for hidden layers
   • Fast to compute
   • Helps with vanishing gradient problem
"""
relu = tf.nn.relu(x)

"""
2. Sigmoid
   f(x) = 1 / (1 + e^(-x))
   • Output range: (0, 1)
   • Used for binary classification
   • Can cause vanishing gradients
"""
sigmoid = tf.nn.sigmoid(x)

"""
3. Tanh (Hyperbolic Tangent)
   f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   • Output range: (-1, 1)
   • Zero-centered (better than sigmoid)
"""
tanh = tf.nn.tanh(x)

"""
4. Softmax
   Converts logits to probabilities (sum to 1)
   • Used for multi-class classification
"""
softmax = tf.nn.softmax(x)

"""
5. Leaky ReLU
   f(x) = x if x > 0, else alpha * x
   • Prevents "dying ReLU" problem
"""
leaky_relu = tf.nn.leaky_relu(x, alpha=0.01)

# ============================================================================
# 8. LOSS FUNCTIONS AND OPTIMIZERS
# ============================================================================
"""
LOSS FUNCTIONS
--------------
Measure how far predictions are from actual values.

REGRESSION LOSSES:
"""
# Mean Squared Error (MSE)
mse = tf.keras.losses.MeanSquaredError()

# Mean Absolute Error (MAE)
mae = tf.keras.losses.MeanAbsoluteError()

# Huber Loss (robust to outliers)
huber = tf.keras.losses.Huber()

"""
CLASSIFICATION LOSSES:
"""
# Binary Cross-Entropy (for binary classification)
binary_ce = tf.keras.losses.BinaryCrossentropy()

# Categorical Cross-Entropy (for multi-class, one-hot encoded)
categorical_ce = tf.keras.losses.CategoricalCrossentropy()

# Sparse Categorical Cross-Entropy (for multi-class, integer labels)
sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy()

"""
OPTIMIZERS
----------
Algorithms to update model weights based on gradients.
"""
# SGD (Stochastic Gradient Descent)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)

# Adam (Adaptive Moment Estimation) - Most popular
adam = tf.keras.optimizers.Adam(learning_rate=0.001)

# RMSprop
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# AdaGrad
adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.01)

# ============================================================================
# 9. TRAINING MODELS
# ============================================================================
"""
MODEL COMPILATION
-----------------
Configure the model before training.
"""
model.compile(
    optimizer='adam',  # or tf.keras.optimizers.Adam(learning_rate=0.001)
    loss='binary_crossentropy',  # or tf.keras.losses.BinaryCrossentropy()
    metrics=['accuracy']  # Metrics to monitor
)

"""
MODEL TRAINING
--------------
"""
# Assuming X_train, y_train are your data
history = model.fit(
    X_train, y_train,
    epochs=50,  # Number of complete passes through data
    batch_size=32,  # Samples per gradient update
    validation_split=0.2,  # Use 20% of data for validation
    verbose=1  # Show progress
)

"""
MODEL EVALUATION
----------------
"""
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

"""
MAKING PREDICTIONS
------------------
"""
predictions = model.predict(X_new)

# ============================================================================
# 10. PRACTICAL EXAMPLES
# ============================================================================

"""
EXAMPLE 1: LINEAR REGRESSION
-----------------------------
"""
# Generate data
X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = 2 * X_train + 3 + np.random.randn(100, 1) * 0.5

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(X_train, y_train, epochs=50, verbose=0)

# Get learned parameters
weights, bias = model.layers[0].get_weights()
print(f"Learned weight: {weights[0][0]:.4f}")  # Should be ~2
print(f"Learned bias: {bias[0]:.4f}")  # Should be ~3

"""
EXAMPLE 2: BINARY CLASSIFICATION
---------------------------------
"""
# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(features,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

"""
EXAMPLE 3: MNIST DIGIT CLASSIFICATION
--------------------------------------
"""
# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 28x28 -> 784
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

"""
EXAMPLE 4: CUSTOM TRAINING LOOP
--------------------------------
For more control over training process.
"""
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


@tf.function  # Optimize with graph compilation
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    # Backward pass
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch}, Loss: {loss}")

# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================
"""
TENSORFLOW KEY CONCEPTS
=======================

1. TENSOR
   - Multi-dimensional array
   - Basic data structure in TensorFlow
   - Similar to NumPy arrays

2. VARIABLE
   - Trainable tensor
   - Stores model parameters (weights, biases)
   - Can be updated during training

3. LAYER
   - Building block of neural networks
   - Transforms input data
   - Has weights and activation function

4. MODEL
   - Collection of layers
   - Sequential or Functional API
   - Can be trained and make predictions

5. ACTIVATION FUNCTION
   - Introduces non-linearity
   - Common: ReLU, Sigmoid, Tanh, Softmax

6. LOSS FUNCTION
   - Measures prediction error
   - Regression: MSE, MAE
   - Classification: Cross-Entropy

7. OPTIMIZER
   - Updates weights to minimize loss
   - Common: Adam, SGD, RMSprop

8. GRADIENT
   - Derivative of loss with respect to weights
   - Computed via automatic differentiation
   - Used in backpropagation

9. EPOCH
   - One complete pass through training data
   - Model sees all samples once per epoch

10. BATCH
    - Subset of data processed together
    - Smaller batches = more updates but noisier
    - Typical sizes: 32, 64, 128

BEST PRACTICES
==============
✓ Normalize input data (0-1 or standardize)
✓ Use ReLU for hidden layers
✓ Use Adam optimizer (good default)
✓ Start with simple architectures
✓ Monitor validation metrics to avoid overfitting
✓ Use Dropout for regularization
✓ Batch normalization for deeper networks
✓ Learning rate scheduling for better convergence
✓ Early stopping to prevent overfitting
✓ Save models regularly during training
"""

# ============================================================================
# COMMON WORKFLOW
# ============================================================================
"""
TYPICAL TENSORFLOW WORKFLOW
===========================

1. PREPARE DATA
   - Load dataset
   - Split into train/validation/test
   - Normalize/preprocess
   - Create batches

2. BUILD MODEL
   - Define architecture
   - Choose layers and activations
   - Set input/output shapes

3. COMPILE MODEL
   - Choose optimizer
   - Choose loss function
   - Set metrics to monitor

4. TRAIN MODEL
   - Call model.fit()
   - Monitor training/validation metrics
   - Use callbacks (early stopping, checkpoints)

5. EVALUATE MODEL
   - Test on unseen data
   - Calculate final metrics
   - Analyze errors

6. MAKE PREDICTIONS
   - Use model.predict()
   - Post-process outputs if needed

7. SAVE/DEPLOY MODEL
   - Save weights or entire model
   - Deploy to production
   - Serve predictions
"""

print("\n" + "=" * 80)
print("TENSORFLOW BASICS TUTORIAL COMPLETE")
print("=" * 80)
print("\nThis tutorial covered:")
print("  ✓ Tensors and operations")
print("  ✓ Variables and gradients")
print("  ✓ Building neural networks")
print("  ✓ Training and evaluation")
print("  ✓ Practical examples")
print("\nNext steps:")
print("  • Practice with real datasets")
print("  • Explore CNNs for images")
print("  • Learn RNNs for sequences")
print("  • Study transfer learning")
print("  • Experiment with different architectures")
print("=" * 80)