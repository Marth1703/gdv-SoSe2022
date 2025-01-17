# TODO first step is to import the opencv module which is called 'cv2'
import cv2
# TODO check the opencv version
print(cv2.__version__)
# TODO load an image with image reading modes using 'imread'
# cv2.IMREAD_UNCHANGED  - If set, return the loaded image as is (with alpha
#                         channel, otherwise it gets cropped). Ignore EXIF
#                         orientation.
# cv2.IMREAD_GRAYSCALE  - If set, always convert image to the single channel
#                         grayscale image (codec internal conversion).
# cv2.IMREAD_COLOR      - If set, always convert image to the 3 channel BGR
#                         color image.
img = cv2.imread("images/chessboard-contrast-squares.jpg", cv2.IMREAD_GRAYSCALE)
imageSize = (400, 400)
# TODO resize image with 'resize'
img = cv2.resize(img, (400, 400))
# TODO rotate image (but keep it rectangular) with 'rotate'
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# TODO save image with 'imwrite'
img = cv2.imwrite("images/editedChessboard", img)
# TODO show the image with 'imshow'

cv2.imshow("title", img)
cv2.waitKey(0)
