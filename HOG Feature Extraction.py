import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the HOG descriptor for the image
hog = cv2.HOGDescriptor()
descriptor = hog.compute(gray)
print(descriptor)
print(descriptor.shape)