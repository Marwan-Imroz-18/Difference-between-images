# import cv2 as cv2
# import numpy as np
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from sklearn.metrics.pairwise import cosine_similarity

# # Load pre-trained ResNet model
# model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# # Load images
# image1 = cv2.imread("input1.png")
# image2 = cv2.imread("input2.png")

# # Preprocess images
# image1 = cv2.resize(image1, (224, 224))
# image2 = cv2.resize(image2, (224, 224))
# image1 = preprocess_input(image1)
# image2 = preprocess_input(image2)

# # Extract features using ResNet
# features1 = model.predict(np.expand_dims(image1, axis=0))
# features2 = model.predict(np.expand_dims(image2, axis=0))

# # Compute cosine similarity
# similarity_score = cosine_similarity(features1, features2)[0][0]

# # Threshold for considering a region as a difference
# threshold = 0.5

# # Compute absolute difference between images
# diff_image = cv2.absdiff(image1, image2)

# # Convert difference image to grayscale
# diff_image_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

# # Threshold the difference image to get binary image
# _, thresh = cv2.threshold(diff_image_gray, 30, 255, cv2.THRESH_BINARY)

# # Find contours (bounding boxes) in the binary image
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw bounding boxes around the contours
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Visualize the images with bounding boxes
# cv2.imshow("Image 1 with Bounding Boxes", image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Print similarity score
# print("Similarity Score:", similarity_score)
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ResNet model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load images
image1 = cv2.imread("input1.png")
image2 = cv2.imread("input2.png")

# Preprocess images
image1 = cv2.resize(image1, (224, 224))
image2 = cv2.resize(image2, (224, 224))
image1 = preprocess_input(image1)
image2 = preprocess_input(image2)

# Extract features using ResNet
image1 = np.array(image1, np.uint8)
image2 = np.array(image2, np.uint8)
features1 = model.predict(np.expand_dims(image1, axis=0))
features2 = model.predict(np.expand_dims(image2, axis=0))

# Compute cosine similarity
similarity_score = cosine_similarity(features1, features2)[0][0]

# Threshold for considering a region as a difference
threshold = 0.5

# Compute absolute difference between images
diff_image = cv2.absdiff(image1, image2)

# Convert difference image to grayscale
diff_image_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

# Threshold the difference image to get binary image
_, thresh = cv2.threshold(diff_image_gray, 30, 255, cv2.THRESH_BINARY)

# Invert the binary image
thresh = cv2.bitwise_not(thresh)

# Find contours (bounding boxes) in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around the contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Visualize the images with bounding boxes
cv2.imwrite("Image 1 with Bounding Boxes.png", image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Print similarity score
print("Similarity Score:", similarity_score)
