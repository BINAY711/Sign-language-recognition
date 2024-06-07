import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Limit GPU memory usage
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 10
EPOCHS = 10

# CNN model
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.20))
classifier.add(Dense(units=112, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=80, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.summary()

# Data generators
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset-1/train',
                                                 target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                 batch_size=BATCH_SIZE,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset-1/test',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=BATCH_SIZE,
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Model training
classifier.fit_generator(
        training_set,
        steps_per_epoch=training_set.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_set,
        validation_steps=test_set.samples // BATCH_SIZE)

# Save the model
classifier.save("model-mix-1.h5")
print('Model saved')
