import numpy as np
import cv2
import keras
import tensorflow as tf
from string import ascii_uppercase

# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# Load the pre-trained model
model = keras.models.load_model("C:\\Users\\lamab\\Downloads\\Sign-Language-Recognition\\final-models\\model-alpha.h5")

# Create a dictionary to map predictions to alphabet characters
alpha_dict = {i: char for i, char in enumerate(ascii_uppercase)}

# Initialize the camera
cam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
    roi = frame[10:300, 320:620]

    # Preprocess the region of interest (ROI)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
    ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("BW", final_image)
    final_image = cv2.resize(final_image, (128, 128))

    # Reshape the image and make prediction
    final_image = np.reshape(final_image, (1, final_image.shape[0], final_image.shape[1], 1))
    pred = model.predict(final_image)
    predicted_char = alpha_dict[np.argmax(pred)]

    # Display the predicted character on the frame
    cv2.putText(frame, predicted_char, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow("Frame", frame)

    # Break the loop if 'Esc' key is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
