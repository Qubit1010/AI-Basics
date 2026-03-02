"""
=============================================================
CNN SCENARIO: Manufacturing Defect Detection
=============================================================

SCENARIO:
---------
A manufacturing company "AeroParts Inc." produces high-precision metal parts.
Occasionally, the machining process causes micro-cracks (defects) on the surface.
Manual inspection is slow and prone to human error.

The company wants to build an AI vision system to:
1. Automatically inspect 28x28 grayscale images of the parts.
2. Flag defective parts for manual review or scrapping.

FEATURES:
- Image Data : 28x28 pixel arrays representing the surface of the part.
               (Generated synthetically: Normal parts have random noise,
               defective parts contain a distinct dark line simulating a crack).

TARGET:
- Status: 1 = Defective (crack present), 0 = Normal/Flawless
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Generate Realistic Synthetic Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  AeroParts Defect Detection using CNN")
print("=" * 60)
print("\n[STEP 1] Generating synthetic image dataset...\n")

np.random.seed(42)
n_samples = 3000  # Kept moderate to ensure quick training times on standard CPUs
img_size = 28

# Simulate base metal surfaces (light gray with random noise)
X_data = np.random.normal(loc=0.7, scale=0.1, size=(n_samples, img_size, img_size))
X_data = np.clip(X_data, 0, 1)

# Target: 30% of parts are defective
y_target = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# Inject "defects" (micro-cracks) into the defective samples
for i in range(n_samples):
    if y_target[i] == 1:
        # Simulate a crack by drawing a random dark, jagged line
        start_row = np.random.randint(2, 15)
        start_col = np.random.randint(2, 15)
        length = np.random.randint(8, 18)

        # Random direction for the crack
        row_dir = np.random.choice([-1, 0, 1])
        col_dir = np.random.choice([-1, 0, 1])

        # Ensure it doesn't just stay in one spot
        if row_dir == 0 and col_dir == 0: row_dir = 1

        r, c = start_row, start_col
        for _ in range(length):
            if 0 <= r < img_size and 0 <= c < img_size:
                X_data[i, r, c] = np.random.uniform(0.0, 0.3)  # Dark pixel (crack)
                # Make it slightly thicker occasionally
                if np.random.random() > 0.5 and r + 1 < img_size:
                    X_data[i, r + 1, c] = np.random.uniform(0.0, 0.4)
            r += row_dir
            c += col_dir

print(f"Dataset Shape: {X_data.shape}")
print(f"Defect Rate:   {y_target.mean() * 100:.1f}%")

# ─────────────────────────────────────────────
# STEP 2: Exploratory Data Analysis
# ─────────────────────────────────────────────
print("\n[STEP 2] Exploratory Data Analysis...\n")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('AeroParts - Surface Inspection Samples', fontsize=16, fontweight='bold')

normal_idx = np.where(y_target == 0)[0][:4]
defect_idx = np.where(y_target == 1)[0][:4]

# Plot Normal Parts
for i, idx in enumerate(normal_idx):
    axes[0, i].imshow(X_data[idx], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title("Normal (0)", color='green')
    axes[0, i].axis('off')

# Plot Defective Parts
for i, idx in enumerate(defect_idx):
    axes[1, i].imshow(X_data[idx], cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title("Defective (1)", color='red')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('cnn_eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA sample images saved.")

# ─────────────────────────────────────────────
# STEP 3: Data Preprocessing
# ─────────────────────────────────────────────
print("\n[STEP 3] Preprocessing Image Data...\n")

# CNNs require a channel dimension (e.g., shape: N, 28, 28, 1 for grayscale)
X_processed = X_data[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_target, test_size=0.2, random_state=42, stratify=y_target)

print(f"Training set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")

# ─────────────────────────────────────────────
# STEP 4: Build CNN Model
# ─────────────────────────────────────────────
print("\n[STEP 4] Building CNN Architecture...\n")

model = Sequential([
    # Convolutional Block 1
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
           input_shape=(img_size, img_size, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Convolutional Block 2
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # Output Layer
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ─────────────────────────────────────────────
# STEP 5: Train Model
# ─────────────────────────────────────────────
print("\n[STEP 5] Training the CNN...\n")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# STEP 6: Evaluate Model
# ─────────────────────────────────────────────
print("\n[STEP 6] Evaluating Model Performance...\n")

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"ROC-AUC Score : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Defective']))

# ─────────────────────────────────────────────
# STEP 7: Visualize Results
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CNN Model Results - Defect Detection', fontsize=16, fontweight='bold')

# Training History - Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', color='#3498db', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='#e74c3c',
                linewidth=2, linestyle='--')
axes[0, 0].set_title('Model Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training History - Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train Acc', color='#2ecc71', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc', color='#f39c12',
                linewidth=2, linestyle='--')
axes[0, 1].set_title('Model Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
            xticklabels=['Normal', 'Defective'],
            yticklabels=['Normal', 'Defective'])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1, 1].plot(fpr, tpr, color='#9b59b6', linewidth=2, label=f'AUC = {auc:.4f}')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nModel results plot saved.")

# ─────────────────────────────────────────────
# STEP 8: Predict on a New Image
# ─────────────────────────────────────────────
print("\n[STEP 8] Real-World Prediction Example...\n")

# Generate a brand new defective image
new_image = np.random.normal(loc=0.7, scale=0.1, size=(img_size, img_size))
new_image = np.clip(new_image, 0, 1)

# Inject a heavy crack
for i in range(5, 20):
    new_image[i, i] = 0.0
    new_image[i, i + 1] = 0.0

# Prepare for model (Add batch and channel dimensions)
new_image_processed = new_image[np.newaxis, ..., np.newaxis]

defect_probability = model.predict(new_image_processed)[0][0]

print(f"\nDefect Probability: {defect_probability * 100:.1f}%")
print(
    f"Prediction:         {'⚠️  DEFECT DETECTED - Route to Manual QC!' if defect_probability > 0.5 else '✅  Normal - Cleared for shipping'}")

print("\n" + "=" * 60)
print("  All outputs saved to working directory")
print("=" * 60)