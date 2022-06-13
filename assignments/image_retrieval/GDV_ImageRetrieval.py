from GDV_TrainingSet import Descriptor, TrainingSet
import cv2
import numpy as np


def findBestMatch(trainData, sample):
    # do the matching with FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(trainData.trainData, sample, k=1)
    # Sort by their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)
    bestMatch = matches[0][0]
    return bestMatch.queryIdx


def getMosaicImage(img, n_rows, n_cols):
    img_mosaic = np.zeros(img.shape)
    imgX = int(img.shape[1] / n_rows) * n_rows
    imgY = int(img.shape[0] / n_cols) * n_cols
    for idx1 in range (0, n_rows):
        for idx2 in range (0, n_cols):
            devided_Section = img[int(idx1 * (imgY / n_rows)): int((idx1 * imgY / n_rows + imgY / n_rows)),
                                  int(idx2 * (imgX / n_cols)): int((idx2 * imgX / n_cols + imgX / n_cols))]
            assert(isinstance(trainData.descriptor, Descriptor))
            descr = trainData.descriptor
            newcomer = np.ndarray(shape=(1, descr.getSize()),
                                buffer=np.float32(descr.compute(devided_Section)),
                                dtype=np.float32)
            match = findBestMatch(trainData, newcomer)
            matching_img = cv2.imread(trainData.getFilenameFromIndex(match), cv2.IMREAD_COLOR)
            matching_img = cv2.resize(matching_img, (int(imgX / n_cols), int(imgY / n_rows)))
            img_mosaic[int(idx1 * (imgY / n_rows)): int((idx1 * imgY / n_rows + imgY / n_rows)),
                       int(idx2 * (imgX / n_cols)): int((idx2 * imgX / n_cols + imgX / n_cols))] = matching_img
    cv2.imshow("title", img_mosaic)

    
    
''' Define and compute or load the training data '''
root_path = 'assignments/image_retrieval/data/101_ObjectCategories/' 
file_name = 'assignments/image_retrieval/data/data.npz'

trainData = TrainingSet(root_path)

# either create and save the data
# trainData.createTrainingData(Descriptor.TINY_GRAY4)
# trainData.saveTrainingData(file_name)
# or load the saved data if descriptor has not been changed.
trainData.loadTrainingData(file_name)

# exemplary test image to check the implementation. As it is part of the
# data set, the best match in the data set needs to be the same image.
newImg = cv2.imread('assignments/image_retrieval/data/101_ObjectCategories/airplanes/image_0005.jpg', cv2.IMREAD_COLOR)
# alternatively use another image and find the best match
# newImg = cv2.imread('images/butterfly.jpg', cv2.IMREAD_COLOR)
cv2.imshow('query image', newImg)

# assure that the same descriptor is used by reading it from the training data set
assert(isinstance(trainData.descriptor, Descriptor))
descr = trainData.descriptor
newcomer = np.ndarray(shape=(1, descr.getSize()),
                      buffer=np.float32(descr.compute(newImg)),
                      dtype=np.float32)
idx = findBestMatch(trainData, newcomer)
best_matching_img = cv2.imread(trainData.getFilenameFromIndex(idx), cv2.IMREAD_COLOR)
cv2.imshow('best match', best_matching_img)
img_to_compute = cv2.imread('assignments/image_retrieval/data/101_ObjectCategories/accordion/image_0010.jpg')
getMosaicImage(img_to_compute, 10, 10)
cv2.waitKey(0)