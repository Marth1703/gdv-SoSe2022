import cv2
import numpy as np

# Goal: Count the number of green smarties in the images
# define green in HSV
hue = 10  # 60 is pure green
hue_range = 8
saturation = 155
saturation_range = 100
value = 155
value_range = 100
lower_green = np.array([hue - hue_range, saturation -
                       saturation_range, value - value_range])
upper_green = np.array([hue + hue_range, saturation +
                       saturation_range, value + value_range])

# load image
img = cv2.imread('images/chewing_gum_balls02.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (800, 600))

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# create a mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# TODO morphological operations code
default_Kernel = np.ones((5, 5), np.uint8)
dilation_Kernel = np.ones((15, 15), np.uint8)
erosion_Kernel = np.ones((15, 15), np.uint8)


def Image_Opening(Image):
    Image = cv2.erode(Image, erosion_Kernel, iterations=1)
    return cv2.dilate(Image, dilation_Kernel, iterations=1)


def Image_Closing(Image):
    Image = cv2.dilate(Image, dilation_Kernel, iterations=1)
    return cv2.erode(Image, erosion_Kernel, iterations=1)
  

mask = cv2.dilate(mask, default_Kernel, iterations=3)
mask = Image_Opening(mask)
mask = cv2.erode(mask, erosion_Kernel, iterations=3)


# TODO find connected components
connectivity = 8
(numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
    mask, connectivity, cv2.CV_32S)
# TODO go through all (reasonable) found connected components

# TODO (optional) check size and roundness as plausibility

# TODO find and draw center

# TODO find and draw bounding box

# print out number of connected components
print('We have found ' + str(numLabels - 1) + ' blue smarties.')


# show the original image with drawings in one window

cv2.imshow('Original image', img)

# show the masked image in another window

# show the mask image in another window

cv2.imshow('Mask image', mask)

# cv2.imwrite('mask.jpg',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
