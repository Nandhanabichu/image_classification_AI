#Custom CNN

import tensorflow as tf
from tensorflow.keras import layers, models

model_cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_cnn = model_cnn.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=5,
    batch_size=32
)


#Evaluation

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = (model_cnn.predict(X_val) > 0.5).astype("int32")

print(classification_report(Y_val, y_pred, target_names=['Cat','Dog']))

cm = confusion_matrix(Y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat','Dog'], yticklabels=['Cat','Dog'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()