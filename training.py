import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Set up data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'archive (12)/TrashType_Image_Dataset',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# 2. Create a tiny model
model = models.Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(6, activation='softmax')  # 6 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train (just 3 epochs for speed)
print("Training... (this will take a few minutes)")
model.fit(train_generator, epochs=3)

# 4. Save the model
model.save('simple_garbage_model.h5')
print("\nDone! Model saved as 'simple_garbage_model.h5'")
