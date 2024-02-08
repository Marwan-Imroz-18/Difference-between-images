# Image Difference Detection

This project aims to detect differences between two images and visualize them by drawing bounding boxes around the detected differences. It provides multiple implementations using different techniques such as simple pixel-wise comparison, structural similarity index (SSIM), and feature-based comparison using a pre-trained ResNet model.

## Implementation Details

### `detection.py`

This script performs image difference detection by directly comparing pixel values between two images. It uses OpenCV for image processing and contour detection to draw bounding boxes around the detected differences. The similarity score is calculated based on the count of non-zero pixels in the grayscale difference image.

### `detection1.py`

Similar to `detection.py`, this script utilizes OpenCV for image processing and contour detection to detect differences between two images. It also calculates the similarity score based on the count of non-zero pixels in the grayscale difference image. Additionally, it employs the SSIM metric from scikit-image for comparing the structural similarity between images.

### `detect.py`

This script provides an alternative approach to image difference detection using a pre-trained ResNet50 model. It extracts features from the images using ResNet50 and computes the cosine similarity between the feature vectors. Bounding boxes are drawn around the detected differences based on the absolute difference between the images.

## Usage

1. Install the required dependencies by running:

2. Run the desired script (`detection.py`, `detection1.py`, or `detect.py`) with the input images as arguments.

Example:

3. The script will output the similarity score and save the result image with bounding boxes as `changes.png`.

## Dependencies

- OpenCV (cv2)
- imutils
- scikit-image
- TensorFlow (for ResNet50 feature extraction)

## Notes

- Ensure that the input images have the same dimensions.
- Adjust threshold values and other parameters based on the specific requirements of your images.

