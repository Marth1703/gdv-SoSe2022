import numpy as np
import cv2

# TODO capture webcam image
capture = cv2.VideoCapture(0)

# TODO get camera image parameters from get()
cameraWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
cameraHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
cameraCodec = int(capture.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))

print("  Width = " + str(cameraWidth))
print("  Height = " + str(cameraHeight))
print("  Codec = " + str(cameraCodec))
# TODO create a window for the video

title = "First video"
cv2.namedWindow(title, cv2.WINDOW_FREERATIO)

# TODO start a loop
while True:

    returnValue, frameValue = capture.read()
    if returnValue is False:
        print("Camera capture failed!")
        break
    else:
        img = np.zeros(frameValue.shape, np.uint8)
        smaller_frame = cv2.resize(frameValue, (0, 0), fx=0.5, fy=0.5)
        # top left (original)
        img[:cameraHeight//2, :cameraWidth//2] = smaller_frame
        # bottom left flipped horizontally
        img[cameraHeight//2:, :cameraWidth//2] = cv2.flip(smaller_frame, 0)
        # bottom left flipped both horizontally and vertically
        img[cameraHeight//2:, cameraWidth//2:] = cv2.flip(smaller_frame, -1)
        # top right flipped vertically
        img[:cameraHeight//2, cameraWidth//2:] = cv2.flip(smaller_frame, 1)

        # display the image
        cv2.imshow(title, img)

        # press q to close the window
    if cv2.waitKey(10) == ord('q'):
        break
# TODO (in loop) read a camera frame and check if that was successful

# TODO (in loop) create four flipped tiles of the image

# TODO (in loop) display the image

# TODO (in loop) press q to close the window and exit the loop


# TODO release the video capture object and window
