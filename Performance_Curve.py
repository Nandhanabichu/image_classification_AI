#Performance Curve

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Accuracy curves
plt.subplot(1,2,1)
plt.plot(history_cnn.history['accuracy'], label='CNN Train')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Val')
plt.plot(history_mobilenet.history['accuracy'], label='MobileNet Train')
plt.plot(history_mobilenet.history['val_accuracy'], label='MobileNet Val')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss curves
plt.subplot(1,2,2)
plt.plot(history_cnn.history['loss'], label='CNN Train')
plt.plot(history_cnn.history['val_loss'], label='CNN Val')
plt.plot(history_mobilenet.history['loss'], label='MobileNet Train')
plt.plot(history_mobilenet.history['val_loss'], label='MobileNet Val')
plt.title('Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()