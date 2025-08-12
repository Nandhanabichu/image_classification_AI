#Load Dataset

from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

cat_dog_classes = [3, 5]

# Filter only cats and dogs
train_filter = np.isin(y_train, cat_dog_classes)
test_filter = np.isin(y_test, cat_dog_classes)

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

y_train = np.where(y_train == 3, 0, 1)
y_test = np.where(y_test == 5, 1, 0)

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Train-validation split
X_train, X_val, Y_train, Y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)


#Sample Visualization

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i])
    plt.title("Cat" if Y_train[i] == 0 else "Dog")
    plt.axis('off')
plt.show()
