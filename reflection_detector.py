import cv2
import numpy as np

def find_feature_mask(image_path, threshold_value=100, min_area=100, aspect_ratio_threshold=2.0, top_margin_filter=50):
    """
    Finds a prominent vertical feature in the right half of an image using thresholding,
    morphological operations, and contour filtering.

    Args:
        image_path: The path to the image file.
        threshold_value: The threshold for binary segmentation.
        min_area: The minimum contour area to consider.
        aspect_ratio_threshold: The minimum height/width ratio to be considered vertical.
        top_margin_filter: The number of pixels from the top to ignore.

    Returns:
        A binary mask of the detected feature, or None if no feature is found.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Isolate the right half of the image
        height, width = enhanced_gray.shape
        right_half = enhanced_gray[:, width // 2:]

        # Apply binary thresholding
        _, thresh = cv2.threshold(right_half, threshold_value, 255, cv2.THRESH_BINARY)
        cv2.imwrite("debug_threshold.png", thresh)

        # Use a vertical kernel for morphological closing to connect parts of the lightning bolt
        kernel = np.ones((10, 3), np.uint8)
        morph_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("debug_morph_closed.png", morph_closed)

        # Find contours
        contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found after morphological operations.")
            return None

        # Filter contours
        filtered_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter by area, position (not in the top margin), and aspect ratio
            if cv2.contourArea(c) > min_area and y > top_margin_filter and h > w * aspect_ratio_threshold:
                filtered_contours.append(c)

        if not filtered_contours:
            print("No contours passed the filtering criteria.")
            return None

        # Select the largest contour from the filtered list
        largest_contour = max(filtered_contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        mask = np.zeros(right_half.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        return mask

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    image_to_test = "ltg_video_int_img (1).png"

    feature_mask = find_feature_mask(image_to_test, threshold_value=100)

    if feature_mask is not None:
        output_path = 'feature_mask.png'
        cv2.imwrite(output_path, feature_mask)
        print(f"Feature mask saved to {output_path}")
    else:
        print("Could not generate feature mask.")
