import numpy as np
import cv2

# TODO open a video file
cap = cv2.VideoCapture("videos/video_illusion.mp4")
# TODO get camera image parameters from get()
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
# TODO start a loop
while True:
# TODO (in loop) read one video frame
    returnValue, frameValue = cap.read()

# TODO (in loop) create four tiles of the image
    if (returnValue):
        img = np.zeros(frameValue.shape, np.uint8)
        smallerFrame = cv2.resize(frameValue, (0, 0), fx=0.5, fy=0.5)
        img[:capHeight//2, capWidth//2:] = smallerFrame
        img[:capHeight//2, :capWidth//2] = smallerFrame
        img[capHeight//2:, capWidth//2:] = smallerFrame
        cv2.imshow("Video", img)
        if cv2.waitKey(10) == ord("q"):
            break
    else:
        print("Frame error")
        break
cap.release()
cv2.destroyAllWindows()
# TODO (in loop) show the image

# TODO (in loop) close the window and stop the loop if 'q' is pressed

# TODO release the video and close all windows
