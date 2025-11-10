import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def visualize_check(image_path):
    """
    Visualizes the symmetry check by displaying the image halves.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Vertical split
    left_half = gray[:, :width // 2]
    right_half = gray[:, width // 2:]
    flipped_right_half = cv2.flip(right_half, 1)
    min_width = min(left_half.shape[1], flipped_right_half.shape[1])
    left_half_v = left_half[:, :min_width]
    flipped_right_half_v = flipped_right_half[:, :min_width]

    # Horizontal split
    top_half = gray[:height // 2, :]
    bottom_half = gray[height // 2:, :]
    flipped_bottom_half = cv2.flip(bottom_half, 0)
    min_height = min(top_half.shape[0], flipped_bottom_half.shape[0])
    top_half_h = top_half[:min_height, :]
    flipped_bottom_half_h = flipped_bottom_half[:min_height, :]

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Symmetry Visualization Check')

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(left_half_v, cmap='gray')
    axes[0, 1].set_title('Left Half')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(flipped_right_half_v, cmap='gray')
    axes[0, 2].set_title('Flipped Right Half')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(top_half_h, cmap='gray')
    axes[1, 1].set_title('Top Half')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(flipped_bottom_half_h, cmap='gray')
    axes[1, 2].set_title('Flipped Bottom Half')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('symmetry_visualization.png')
    print("Symmetry visualization saved to symmetry_visualization.png")


def detect_mirror_reflection(image_path, similarity_threshold=0.7):
    """
    Detects mirror-like reflections in an image by checking for symmetry.

    Args:
        image_path: The path to the image file.
        similarity_threshold: The SSIM score above which a reflection is considered detected.

    Returns:
        True if a mirror reflection is detected, False otherwise.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Check for vertical symmetry
        left_half = gray[:, :width // 2]
        right_half = gray[:, width // 2:]
        flipped_right_half = cv2.flip(right_half, 1)

        # Ensure halves are the same size for comparison
        min_width = min(left_half.shape[1], flipped_right_half.shape[1])
        left_half_resized = left_half[:, :min_width]
        flipped_right_half_resized = flipped_right_half[:, :min_width]

        vertical_ssim = ssim(left_half_resized, flipped_right_half_resized)

        if vertical_ssim > similarity_threshold:
            return True

        # Check for horizontal symmetry
        top_half = gray[:height // 2, :]
        bottom_half = gray[height // 2:, :]
        flipped_bottom_half = cv2.flip(bottom_half, 0)

        # Ensure halves are the same size for comparison
        min_height = min(top_half.shape[0], flipped_bottom_half.shape[0])
        top_half_resized = top_half[:min_height, :]
        flipped_bottom_half_resized = flipped_bottom_half[:min_height, :]

        horizontal_ssim = ssim(top_half_resized, flipped_bottom_half_resized)

        if horizontal_ssim > similarity_threshold:
            return True

        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    image_to_test = "ltg_video_int_img (1).png"

    # --- Test Mirror Reflection Detection ---
    if detect_mirror_reflection(image_to_test):
        print(f"Mirror reflection detected in {image_to_test}")
    else:
        print(f"No mirror reflection detected in {image_to_test}")

    # --- Visualize the symmetry check ---
    print("Displaying symmetry visualization...")
    visualize_check(image_to_test)
