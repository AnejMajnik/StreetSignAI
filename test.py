import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
from tqdm import tqdm

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up TensorFlow to use GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# nalaganje modela
model = load_model('small_model_v2.h5')

# nalaganje oznak
annotations_path = 'GTSRB_Final_Test_GT/GT-final_test.csv'
annotations = pd.read_csv(annotations_path, sep=';')

def preprocess_images(img_paths):
    images = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        images.append(img_array)
    return np.array(images)

correct_predictions = 0
total_predictions = 0
batch_size = 128

# grupiranje imena datotek
img_paths = ['TestingImages/' + row["Filename"] for _, row in annotations.iterrows()]
true_classes = [row["ClassId"] for _, row in annotations.iterrows()]

# procesiranje v batches
for start in tqdm(range(0, len(img_paths), batch_size), desc="Testing", unit="batch"):
    end = min(start + batch_size, len(img_paths))
    batch_img_paths = img_paths[start:end]
    batch_true_classes = true_classes[start:end]

    # Preprocess the batch of images
    batch_images = preprocess_images(batch_img_paths)

    # Predict the classes for the batch of images
    predictions = model.predict(batch_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate the number of correct predictions
    correct_predictions += np.sum(predicted_classes == batch_true_classes)
    total_predictions += len(batch_true_classes)

# izračun natančnosti
accuracy = correct_predictions / total_predictions
print(f'Test Accuracy: {accuracy * 100:.2f}%')
