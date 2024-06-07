import cv2
import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def main():
    dataset_folder = "dataset"
    train_folder = os.path.join(dataset_folder, "train")
    test_folder = os.path.join(dataset_folder, "test")

    create_folder_if_not_exists(dataset_folder)
    create_folder_if_not_exists(train_folder)
    create_folder_if_not_exists(test_folder)

    for i in range(10):
        create_folder_if_not_exists(os.path.join(train_folder, str(i)))
        create_folder_if_not_exists(os.path.join(test_folder, str(i)))

    mode = 'test'  # or 'train'
    folder = os.path.join(dataset_folder, mode)
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        draw_roi_rectangle(frame)

        cv2.imshow("Frame", frame)

        binary_image = process_roi(frame)
        cv2.imshow("BW", binary_image)

        key = cv2.waitKey(1)
        if key in [ord(str(i)) for i in range(10)]:
            digit = chr(key)
            save_image(binary_image, folder, digit)
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def draw_roi_rectangle(frame):
    cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)

def process_roi(frame):
    roi = frame[10:300, 320:620]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
    ret, binary_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image = cv2.resize(binary_image, (300, 300))
    return binary_image

def save_image(image, folder, digit):
    digit_folder = os.path.join(folder, digit)
    image_count = len(os.listdir(digit_folder))
    image_path = os.path.join(digit_folder, f"{image_count}.jpg")
    cv2.imwrite(image_path, image)

if __name__ == "__main__":
    main()
