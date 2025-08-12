
#Evaluation_Transfer_Learning_with_MobileNetV2

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

