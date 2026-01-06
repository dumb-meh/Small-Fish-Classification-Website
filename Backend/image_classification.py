import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.convnext import ConvNeXtTiny
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def lw(bottom_model, num_classes):
    """Function to create the top layers for the model (same as training)"""
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(top_model)
    top_model = Dropout(0.6)(top_model) 
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

def load_custom_model(model_path):
    """Load the trained model with the same architecture"""
    try:
        # Try to load the model directly first
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Direct loading failed: {e}")
        print("Attempting to rebuild model architecture...")
        
        # Rebuild the model architecture
        convnextnet_base = ConvNeXtTiny(
            weights=None,  # We'll load weights from saved model
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze layers (same as training)
        for layer in convnextnet_base.layers:
            layer.trainable = False
        
        for layer in convnextnet_base.layers[-20:]:
            layer.trainable = True
        
        # Add custom head
        FC_Head5 = lw(convnextnet_base, 10)
        model = Model(inputs=convnextnet_base.input, outputs=FC_Head5)
        
        # Load weights
        model.load_weights(model_path)
        print("Model rebuilt and weights loaded!")
        return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image in the same way as training"""
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert to tensor and add batch dimension
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_array = tf.expand_dims(img_array, axis=0)
    
    # Note: ConvNeXt models typically expect images in range [0, 255]
    # The model's preprocessing is usually handled internally
    return img_array, img

def predict_single_image(model, image_path, class_names=None):
    """Make prediction on a single image"""
    
    # Preprocess the image
    img_array, original_img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    
    # Print detailed results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    if class_names:
        predicted_class_name = class_names[predicted_class_idx]
        print(f"Predicted class: {predicted_class_name} (Class {predicted_class_idx})")
    else:
        print(f"Predicted class: Class {predicted_class_idx}")
    
    print(f"Confidence: {confidence:.4f}")
    
    # Print all class probabilities
    print("\nAll class probabilities:")
    for i, prob in enumerate(predictions[0]):
        if class_names:
            print(f"  {class_names[i]:20s}: {prob:.6f}")
        else:
            print(f"  Class {i:2d}: {prob:.6f}")
    
    return predicted_class_idx, confidence

# Define class names (must match training order)
CLASS_NAMES = [
    'Bele',
    'Chela',
    'Guchi',
    'Kachki',
    'Kata Phasa',
    'Mola',
    'Nama Chanda',
    'Pabda',
    'Puti',
    'Tengra'
]

# Global model instance
_model = None
_model_loaded = False
_model_error = None

def load_model_once():
    """Load the model once and cache it"""
    global _model, _model_loaded, _model_error
    
    if _model_loaded:
        return _model
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model", "convnextnet_model.h5")
        print(f"[ImageClassification] Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            _model_error = f"Model file not found at {model_path}"
            print(f"[ImageClassification] ERROR: {_model_error}")
            return None
        
        _model = load_custom_model(model_path)
        _model_loaded = True
        print("[ImageClassification] Model loaded successfully!")
        return _model
        
    except Exception as e:
        _model_error = str(e)
        print(f"[ImageClassification] ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def classify_image(image_path):
    """
    Classify an uploaded fish image
    
    Args:
        image_path: Path to the uploaded image file
        
    Returns:
        tuple: (label, confidence, method) where:
            - label: predicted fish species name
            - confidence: confidence score (0-1)
            - method: 'dl' for deep learning or 'fallback' for filename-based
    """
    print(f"[ImageClassification] classify_image called with: {image_path}")
    
    # Try to load model
    model = load_model_once()
    
    if model is None:
        print("[ImageClassification] Model not available, using fallback")
        # Fallback: try to extract from filename
        basename = os.path.basename(image_path)
        for class_name in CLASS_NAMES:
            if class_name.lower() in basename.lower():
                print(f"[ImageClassification] Fallback detected '{class_name}' in filename")
                return class_name, 0.5, 'fallback'
        
        print("[ImageClassification] Fallback: no class name found in filename")
        return "Unknown", 0.0, 'fallback'
    
    # Use model for prediction
    try:
        print("[ImageClassification] Running model prediction...")
        predicted_class_idx, confidence = predict_single_image(model, image_path, CLASS_NAMES)
        label = CLASS_NAMES[predicted_class_idx]
        print(f"[ImageClassification] Prediction complete: {label} ({confidence:.4f})")
        return label, float(confidence), 'dl'
        
    except Exception as e:
        print(f"[ImageClassification] ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0, 'error'

def model_status():
    """Return diagnostic information about model loading status"""
    return {
        'loaded': _model_loaded,
        'error': _model_error,
        'model_available': _model is not None,
        'class_count': len(CLASS_NAMES),
        'classes': CLASS_NAMES
    }