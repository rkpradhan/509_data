import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_bright_spot(image_path, brightness_threshold=200):
    """
    Detects a bright spot in the left half of an image.

    Args:
        image_path: The path to the image file.
        brightness_threshold: The average brightness level to qualify as a reflection.

    Returns:
        A tuple (bool, box), where bool is True if a bright spot is detected,
        and box is the bounding rectangle of the spot.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return False, None

        # Focus on the left half of the image
        height, width, _ = img.shape
        left_half = img[:, :width // 2]

        gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)

        # Find the brightest contour
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None

        # Find the largest contour by area, which is likely the main reflection
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Check if the average brightness within the contour is high enough
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]

        if mean_val > brightness_threshold:
            return True, (x, y, w, h)
        else:
            return False, None

    except Exception as e:
        print(f"An error occurred: {e}")
        return False, None

def visualize_detected_spot(image_path, box):
    """
    Draws a box around the detected bright spot, shows a histogram,
    and saves the combined image.

    Args:
        image_path: The path to the original image.
        box: The bounding box of the detected spot.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        # Prepare the image with the bounding box
        if box:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Reflection Analysis')

        # Display the image on the left
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Detected Bright Spot')
        ax1.axis('off')

        # Calculate and display the histogram on the right
        height, width, _ = img.shape
        left_half_gray = cv2.cvtColor(img[:, :width // 2], cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([left_half_gray], [0], None, [256], [0, 256])

        ax2.plot(hist)
        ax2.set_title('Brightness Histogram (Left Half)')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Pixel Count')
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('bright_spot_visualization.png')
        print("Bright spot visualization with histogram saved to bright_spot_visualization.png")

    except Exception as e:
        print(f"An error occurred during visualization: {e}")

if __name__ == "__main__":
    image_to_test = "ltg_video_int_img (1).png"

    # --- Test Bright Spot Detection ---
    is_detected, spot_box = detect_bright_spot(image_to_test, brightness_threshold=100)

    if is_detected:
        print(f"Bright spot detected in {image_to_test}.")
        visualize_detected_spot(image_to_test, spot_box)
    else:
        print(f"No bright spot detected in {image_to_test}.")
