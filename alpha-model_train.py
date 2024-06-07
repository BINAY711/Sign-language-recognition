import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os

# Set GPU memory limit
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# Define image size
image_size = 128

# Step 1 - Building the CNN
classifier = Sequential()

# Convolutional layers
classifier.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
classifier.add(Flatten())

# Fully connected layers with dropout
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.20))
classifier.add(Dense(units=112, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=80, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=64, activation='relu'))

# Output layer
classifier.add(Dense(units=26, activation='softmax'))  # 26 classes for alphabets

# Compile the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
classifier.summary()

# Image data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing data
training_set = train_datagen.flow_from_directory(
    'dataset-alpha/train',
    target_size=(image_size, image_size),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'dataset-alpha/test',
    target_size=(image_size, image_size),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

# Train the model
classifier.fit_generator(
    training_set,
    steps_per_epoch=training_set.n // training_set.batch_size,  # Number of images in training set / batch size
    epochs=10,
    validation_data=test_set,
    validation_steps=test_set.n // test_set.batch_size  # Number of images in test set / batch size
)

# Save the trained model
classifier.save("model-all1-alpha.h5")
print('Model Saved')
print(training_set.n // training_set.batch_size)
print(test_set.n // test_set.batch_size)
