import cv2
import numpy as np

# TODO loading images in grey and color
imgColor = cv2.imread("images/cgb_blue.jpg", cv2.IMREAD_COLOR)
imgGray = cv2.imread("images/logo.png", cv2.IMREAD_GRAYSCALE)
# TODO do some print out about the loaded data using type, dtype and shape
print(type(imgColor))
print(imgColor.dtype)
print(imgColor.shape)
# TODO Continue with the grayscale image
print(type(imgGray))
print(imgGray.dtype)
print(imgGray.shape)
# TODO Extract the size or resolution of the image
img = imgGray.copy()
imageHeight = img.shape[0]
imageWidth = img.shape[1]
imageDimension = img.shape[-1]
print("height: " + str(imageHeight))
print("width: " + str(imageWidth))
print("dimension: " + str(imageDimension))

# TODO resize image
newRes = (500, 400)
img = cv2.resize(img, newRes)
# row and column access, see
#   https://numpy.org/doc/stable/reference/arrays.ndarray.html
#   for general access on ndarrays
# TODO print first row
print(img[40])

# TODO print first column
print(img[:, 0])

# TODO continue with the color image
img = imgColor.copy()
img = cv2.resize(img, newRes)
# TODO set an area of the image to black
for i in range(40):
    for j in range(400):
        img[j][i] = [0, 0, 0]
# TODO show the image and wait until key pressed
title = "Image with black bar"
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.imshow(title, img)
cv2.waitKey(0)
# TODO find all used colors in the image
all_rgb_codes = img.reshape(-1, img.shape[-1])
unique_rgb_codes = np.unique(all_rgb_codes, axis=0, return_counts=False)
print("Those color values are in the image:\n" + str(unique_rgb_codes))
# TODO copy one part of an image into another one

copyArray = img[50:100, 30:120]
img[70:120, 50:140] = copyArray
cv2.imshow("copied square", img)
cv2.waitKey(0)
# TODO save image to a file

# TODO show the image again

# TODO show the original image (copy demo)
