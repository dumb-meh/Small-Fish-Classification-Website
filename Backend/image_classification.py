# Load the saved model
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

convnextnet = load_model('convnextnet_model.h5')

def preprocess_and_predict(image_path, model, image_size=(224, 224)):
    # Load the image with the correct target size
    img = image.load_img(image_path, target_size=image_size)
    
    # Convert the image to a numpy array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Normalize the image if your model expects it (e.g., 0-1 or -1 to 1)
    img_array = img_array / 255.0  # Normalize if model was trained with 0-1 normalization
    
    # Make a prediction
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability (i.e., the predicted class)
    predicted_class = np.argmax(predictions, axis=-1)
    
    # Get the confidence score (probability) for the predicted class
    confidence_score = np.max(predictions)
    
    return predicted_class, confidence_score

# Example usage:
image_path = "path_to_your_image.jpg"  # Path to the image you want to predict on
predicted_class, confidence_score = preprocess_and_predict(image_path, convnextnet)

