import cv2
import numpy as np

def detect_reflection(image_path, threshold=0.1):
    """
    Detects reflections in an image.

    Args:
        image_path: The path to the image file.
        threshold: The percentage of bright pixels to be considered a reflection.

    Returns:
        True if a reflection is detected, False otherwise.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return False

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for bright/white color in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 55, 255])

        # Create a mask for the bright regions
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Calculate the percentage of the image that is white
        white_percentage = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100

        # If the percentage is above the threshold, it's likely a reflection
        if white_percentage > threshold:
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    image_to_test = "ltg_video_int_img (1).png"
    if detect_reflection(image_to_test):
        print(f"Reflection detected in {image_to_test}")
    else:
        print(f"No reflection detected in {image_to_test}")
