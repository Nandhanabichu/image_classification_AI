#Transfer Learning with MobileNetV2

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import cv2
X_train_resized = np.array([cv2.resize(img, (128, 128)) for img in X_train])
X_val_resized = np.array([cv2.resize(img, (128, 128)) for img in X_val])

#Preprocessing

X_train_pre = preprocess_input(X_train_resized*255)
X_val_pre = preprocess_input(X_val_resized*255)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model_mobilenet = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_mobilenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_mobilenet = model_mobilenet.fit(
    X_train_pre, Y_train,
    validation_data=(X_val_pre, Y_val),
    epochs=3,
    batch_size=32
)


#Evaluation

import tensorflow as tf

# Resize validation data to 128x128 for MobileNetV2
X_val_resized = tf.image.resize(X_val, (128, 128))

# Predictions
mobilenet_probs = model_mobilenet.predict(X_val_resized)
mobilenet_preds = (mobilenet_probs > 0.5).astype("int32")

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

accuracy = accuracy_score(Y_val, mobilenet_preds)
precision = precision_score(Y_val, mobilenet_preds)
recall = recall_score(Y_val, mobilenet_preds)
f1 = f1_score(Y_val, mobilenet_preds)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(Y_val, mobilenet_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat','Dog'], yticklabels=['Cat','Dog'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_val, mobilenet_preds))

