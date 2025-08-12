#Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# Visualize augmented images
sample_img = X_train[0]
sample_img = np.expand_dims(sample_img, 0)
plt.figure(figsize=(6,6))
for i, batch in enumerate(datagen.flow(sample_img, batch_size=1)):
    plt.subplot(2, 2, i+1)
    plt.imshow(batch[0])
    plt.axis('off')
    if i == 3:
        break
plt.show()
