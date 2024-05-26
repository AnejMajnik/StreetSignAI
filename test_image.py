import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up TensorFlow to use GPU
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# nalaganje modela
model = load_model('medium_model.h5')

# imena razredov
class_names = [
    "Speed limit (20km/h)",         # Class 0
    "Speed limit (30km/h)",         # Class 1
    "Speed limit (50km/h)",         # Class 2
    "Speed limit (60km/h)",         # Class 3
    "Speed limit (70km/h)",         # Class 4
    "Speed limit (80km/h)",         # Class 5
    "End of speed limit (80km/h)",  # Class 6
    "Speed limit (100km/h)",        # Class 7
    "Speed limit (120km/h)",        # Class 8
    "No passing",                   # Class 9
    "No passing for vehicles over 3.5 metric tons",  # Class 10
    "Right-of-way at the next intersection",         # Class 11
    "Priority road",                # Class 12
    "Yield",                        # Class 13
    "Stop",                         # Class 14
    "No vehicles",                  # Class 15
    "Vehicles over 3.5 metric tons prohibited",      # Class 16
    "No entry",                     # Class 17
    "General caution",              # Class 18
    "Dangerous curve to the left",  # Class 19
    "Dangerous curve to the right", # Class 20
    "Double curve",                 # Class 21
    "Bumpy road",                   # Class 22
    "Slippery road",                # Class 23
    "Road narrows on the right",    # Class 24
    "Road work",                    # Class 25
    "Traffic signals",              # Class 26
    "Pedestrians",                  # Class 27
    "Children crossing",            # Class 28
    "Bicycles crossing",            # Class 29
    "Beware of ice/snow",           # Class 30
    "Wild animals crossing",        # Class 31
    "End of all speed and passing limits",           # Class 32
    "Turn right ahead",             # Class 33
    "Turn left ahead",              # Class 34
    "Ahead only",                   # Class 35
    "Go straight or right",         # Class 36
    "Go straight or left",          # Class 37
    "Keep right",                   # Class 38
    "Keep left",                    # Class 39
    "Roundabout mandatory",         # Class 40
    "End of no passing",            # Class 41
    "End of no passing by vehicles over 3.5 metric tons" # Class 42
]


# preprocesiranje slike
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# prediction
def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

img_path = './TestingImages/12315.ppm'
predicted_class_index = predict_image_class(img_path)
predicted_class_name = class_names[predicted_class_index]
print(f'Predicted class index: {predicted_class_index:05d}')
print(f'Predicted class name: {predicted_class_name}')
