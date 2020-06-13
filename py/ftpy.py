# This module is a simply a wrapper for libftpy.so that acts as a remainder how its interface is defined
# C++ library libftpy must exist in same folder for this module to work
import libftpy


"""Get bounding boxes for FASText connected components found with given parameters

Parameters
----------
image : numpy array
    Short int (0-255) valued grayscale image of size 1024x1024x1
coun : int
    Maximum count of boxes that are returned - boxes with keypoints that have least amount of contrast are trimmed
scales : int
    How many scales are used in the scale pyramid in addition of the original scale
threshold : int
    Threshold use when defining a pixel is a FT keypoint or not
positives : bool
    Are boxes found for positive ("bright") keypoints included in the results
negatives : bool
    Are boxes found for negative ("dark") keypoints included in the results
wLimit : int
    Boxes that are wider than wLimit are trimmed from the results   
hLimit : int
    Boxes that are higher than hLimit are trimmed from the results

Returns
-------
boxes : numpy array
    Numpy array of size N * 4 representing the found boxes in format x, y, width, height (dtype is int32)
"""
def getKpBoxes(image, count, scales, threshold, positives, negatives, wLimit, hLimit):

    padding = 0
    return libftpy.getKpBoxes(image, padding, count, scales, threshold, positives, negatives, wLimit, hLimit)


"""Get FASText keypoints found with given parameters

Parameters
----------
image : numpy array
    Short int (0-255) valued grayscale image of size 1024x1024
count : int
    Maximum count of boxes that are returned - boxes with keypoints that have least amount of contrast are trimmed
scales : int
    How many scales are used in the scale pyramid in addition of the original scale
threshold : int
    Threshold use when defining a pixel is a FT keypoint or not
positives : bool
    Are boxes found for positive ("bright") keypoints included in the results
negatives : bool
    Are boxes found for negative ("dark") keypoints included in the results

        icollector[y][x][0] = y; // y
        icollector[y][x][1] = x; // x
        icollector[y][x][2] = stats[0]; // kp type (end or bend)
        icollector[y][x][3] = stats[1]; // lightess (positive or negative)
        icollector[y][x][4] = stats[2]; // max contrast for nms
        icollector[y][x][5] = stats[3]; // difference used in thresholding

Returns
-------
keypoints : numpy array
    Numpy array of size N * 4 representing the found keypoints in format x, y, kp type (end=1, bend=2), kp lightness (positive=1, negative=2), difference for thresholding
"""
def getFTKeypoints(image, count, scales, threshold, positives, negatives):
    padding = 0
    return libftpy.getFTKeypoints(image, padding, count, scales, threshold, positives, negatives)


"""Cluster CC boxes using a custom distance algorithm (which can be found in dbscan.cpp@calculateDistance)

Parameters
----------
boxes : numpy array
    int32 bounding boxes for connected components in format left, top, right, top, right, bottom, left, bottom
eps : floating point number
    Epsilon (distance) parameter for the dbscan algorithm
min_samples : integer
    How many points have be in some points neighbourhood to be a core point

Returns
-------
labels : numpy array
    One-dimensional numpy array of cluster labels for each point
    Nb! NOISE points have label -2
"""
def kpBoxDBSCAN(boxes, eps, min_samples):
    padding = 0
    boxN = len(boxes)
    return libftpy.kpBoxDBSCAN(boxes, padding, boxN, eps, min_samples)