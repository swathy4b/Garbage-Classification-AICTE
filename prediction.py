import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the model
model = load_model('simple_garbage_model.h5')

# Class names (must match training order)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Get image path from user
img_path = input("Enter image path: ")

# Load and prepare image
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

# Show result
plt.imshow(img)
plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.1f}%')
plt.axis('off')
plt.show()

print(f"\nPredicted class: {predicted_class}")
print(f"Confidence: {confidence:.1f}%")
