import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up TensorFlow to use GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# poti do podatkov
train_dir = 'DividedImages/train'
validation_dir = 'DividedImages/validation'

# preprocesiranje slik
train_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical',
    color_mode='rgb',
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical',
    color_mode='rgb',
)

# .repeat()
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
    )
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, validation_generator.num_classes), dtype=tf.float32)
    )
).repeat()

# definicija modela
model = Sequential()

# konvolucijski del
N = 32
for _ in range(4):
    model.add(Conv2D(N, (3, 3), padding='same', input_shape=(64, 64, 3) if _ == 0 else None))
    model.add(Activation('softplus'))
    model.add(Conv2D(N, (3, 3), strides=2, padding='same'))
    model.add(Activation('softplus'))
    model.add(Dropout(0.05))  # Adding slight Dropout layer
    N *= 2

# linearizacija
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.05))  # Adding slight Dropout layer
model.add(Dense(43, activation='softmax'))

learning_rate = 0.0002  # You can adjust this value as needed

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.025,
    patience=4,
    verbose=1,
    restore_best_weights=True
)

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

start_time = time.time()

# treniranje modela
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    epochs=40,
    callbacks=[early_stopping]
)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

model.save('big_model.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
