import numpy as np
import cv2
import keras
import tensorflow as tf

# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# Load pre-trained model
model_path = "C:\\Users\\lamab\\Downloads\\Sign-Language-Recognition\\final-models\\model-alpha.h5"
model = keras.models.load_model(model_path)

# Define dictionary for mapping output indices to sign labels
number_dict = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI)
    roi = frame[10:300, 320:620]

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and adaptive thresholding to obtain binary image
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
    ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize image to match model input size
    final_image = cv2.resize(final_image, (128, 128))

    # Reshape image to match model input shape
    final_image = np.reshape(final_image, (1, final_image.shape[0], final_image.shape[1], 1))

    # Make prediction using the model
    pred = model.predict(final_image)

    # Get predicted sign label
    sign_label = number_dict[np.argmax(pred)]

    # Overlay predicted sign label on frame
    cv2.putText(frame, sign_label, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # Display frame with overlay
    cv2.imshow("Frame", frame)

    # Check for ESC key press to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
