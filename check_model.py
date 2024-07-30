import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model (assuming the model is saved as 'gesture_model.h5')
# If the model is not saved, you can use the model instance directly.
model = tf.keras.models.load_model('gesture_model.h5')

# Path to the new image
new_image_path = r'C:\Users\amank\Documents\coursera\ML\Reports\CNN\Gesture\0_img.jpg'
  # Update this path to your image file

# Load and preprocess the image
image = load_img(new_image_path, target_size=(64, 64))  # Resize image to match model's input shape
image_array = img_to_array(image) / 255.0  # Convert image to array and rescale
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(image_array)
predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability

# Mapping class index to class names (assuming you have a list of class names)
class_names = ['Gesture1', 'Gesture2', 'Gesture3', 'Gesture4', 'Gesture5',  # Add all 20 gesture names
               'Gesture6']

predicted_label = class_names[predicted_class[0]]

print(f'The model predicts the gesture is: {predicted_label}')
