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
        A tuple of (mask, bounding_box) or (None, None) if no feature is found.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        height, width = enhanced_gray.shape
        right_half = enhanced_gray[:, width // 2:]

        _, thresh = cv2.threshold(right_half, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((10, 3), np.uint8)
        morph_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        filtered_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) > min_area and y > top_margin_filter and h > w * aspect_ratio_threshold:
                filtered_contours.append(c)

        if not filtered_contours:
            return None, None

        largest_contour = max(filtered_contours, key=cv2.contourArea)
        mask = np.zeros(right_half.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        # Get the bounding box of the feature on the right side
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Adjust x to be relative to the full image width
        x += width // 2

        return mask, (x, y, w, h)

    except Exception as e:
        print(f"An error occurred in find_feature_mask: {e}")
        return None, None

def find_reflection_source(image_path, feature_mask, feature_box):
    """
    Finds the source of a reflection on the left side of the image using template matching.

    Args:
        image_path: The path to the original image.
        feature_mask: The mask of the feature to be matched.
        feature_box: The bounding box of the feature on the right side.

    Returns:
        The bounding box of the matched source on the left side, or None.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        height, width = enhanced_gray.shape
        left_half = enhanced_gray[:, :width // 2]

        # Extract the template from the right side using the bounding box
        x, y, w, h = feature_box
        template = enhanced_gray[y:y+h, x:x+w]

        # Flip the template horizontally for matching
        flipped_template = cv2.flip(template, 1)

        # Perform template matching on the left half
        res = cv2.matchTemplate(left_half, flipped_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Get the coordinates of the best match
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return (top_left[0], top_left[1], w, h)

    except Exception as e:
        print(f"An error occurred in find_reflection_source: {e}")
        return None

if __name__ == "__main__":
    image_to_test = "ltg_video_int_img (1).png"

    mask, right_box = find_feature_mask(image_to_test, threshold_value=100)

    if mask is not None and right_box is not None:
        print("Feature mask created and feature found on the right side.")

        left_box = find_reflection_source(image_to_test, mask, right_box)

        if left_box is not None:
            print("Matching reflection source found on the left side.")

            # Draw rectangles on the original image
            img_to_draw = cv2.imread(image_to_test)

            # Right side box (reflection) in red
            rx, ry, rw, rh = right_box
            cv2.rectangle(img_to_draw, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

            # Left side box (source) in green
            lx, ly, lw, lh = left_box
            cv2.rectangle(img_to_draw, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)

            output_path = 'reflection_match.png'
            cv2.imwrite(output_path, img_to_draw)
            print(f"Result with matched reflection saved to {output_path}")
        else:
            print("Could not find a matching source for the reflection.")
    else:
        print("Could not generate feature mask or find the feature.")
