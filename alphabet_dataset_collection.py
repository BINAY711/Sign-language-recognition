import cv2
import os
from string import ascii_uppercase

# Function to create directories if they don't exist
def create_directories(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# Constants
BASE_DIR = "dataset-alpha"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Create directories
create_directories(BASE_DIR, ["train", "test"])
create_directories(TRAIN_DIR, ascii_uppercase)
create_directories(TEST_DIR, ascii_uppercase)

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    # Display number of images for each letter
    x, y = 10, 50
    for letter in ascii_uppercase:
        num_images = len(os.listdir(os.path.join(TRAIN_DIR, letter)))
        cv2.putText(frame, f"{letter}: {num_images}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        y += 20
        if letter == 'V':
            x += 60
            y = 50
    
    # Display frame
    cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
    cv2.imshow("Frame", frame)
    
    # Extract ROI and preprocess
    roi = frame[10:300, 320:620]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
    _, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    final_image = cv2.resize(final_image, (300, 300))
    
    # Display binary image
    cv2.imshow("BW", final_image)
    
    # Save images based on key presses
    interrupt = cv2.waitKey(1) & 0xFF
    for letter in ascii_uppercase:
        if interrupt == ord(letter):
            cv2.imwrite(os.path.join(TRAIN_DIR, letter, f"{len(os.listdir(os.path.join(TRAIN_DIR, letter)))}.jpg"), final_image)
    if interrupt == 27:  # ESC key
        break

# Release webcam and close windows
cam.release()
cv2.destroyAllWindows()
