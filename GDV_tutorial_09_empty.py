# Demonstrating usage of filters in OpenCV
import cv2
import numpy as np
import time


# TODO implement the convolution with opencv
def convolution_with_opencv(image, kernel):
    kernel = cv2.flip(kernel, -1)
    ddepth = -1
    result = cv2.filter2D(image, ddepth, kernel)
    return result


def show_kernel(kernel):
    # show the kernel as image
    title_kernel = 'Kernel'
    cv2.namedWindow(title_kernel, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_kernel, 300, 300)
    # scale kernel to make it visually more appealing
    kernel_img = cv2.normalize(
        kernel, kernel, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow(title_kernel, kernel_img)
    cv2.waitKey(0)


def show_resulting_images(image, result):
    title_original = 'Original image'
    cv2.namedWindow(title_original, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title_original, image)

    title_result = 'Resulting image'
    cv2.namedWindow(title_result, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title_result, result)

    key = cv2.waitKey(0)
    if key == ord('s'):
        # save resulting image
        res_filename = 'filtered_with_%dx%d_gauss_kernel_with_sigma_%d.png' % (
            kernel_size, kernel_size, sigma)
        cv2.imwrite(res_filename, result)
    cv2.destroyAllWindows()


# Load the image.
image_name = 'images/Bumbu_Rawon.jpg'
image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (320,213))

# TODO define kernel

kernel_size = 17
# TODO define Gaussian standard deviation (sigma). If it is non-positive,

# sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
sigma = 6

# TODO create the kernel with OpenCV

kernel1D = cv2.getGaussianKernel(kernel_size, sigma)
kernel = np.transpose(kernel1D) * kernel1D
show_kernel(kernel)

# TODO run convolution and measure the time it takes
# start time to calculate computation duration
start_time = time.time()
# run the convolution
convoluted_image = convolution_with_opencv(image, kernel)
# end time after computation
end_time = time.time()
# print timing results
print('Computing the convolution of an image with a resolution of',
      image.shape[1], 'x', image.shape[0], 'and a kernel size of',
      kernel.shape[0], 'x', kernel.shape[1], 'took', end_time - start_time, 'seconds.')

# show the original and the resulting image
show_resulting_images(image, convoluted_image)
