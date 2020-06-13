import cv2
import ftpy
import numpy as np
from cc_conv_clf import CCConvClf
from sklearn.cluster import DBSCAN


# Based on a bounding box, extract sample of size 24x24x3 from image
def extract_cc_sample(img, s):
    sample = img[s[1]:(s[1] + s[3]), s[0]:(s[0] + s[2])]
    resizedSample = cv2.resize(sample, (24, 24), interpolation = cv2.INTER_AREA)
    normalized = (np.float16(resizedSample) / 255) - 0.5
    return normalized


# Create name bounding boxes from the DBSCAN clustered connected component boxes
def extract_cluster_boxes(labels, boxes):
    collector = []
    uniqlabels = np.unique(labels)
    for label in uniqlabels:
        if label != -1:
            box_indices = np.where(labels == label)

            label_boxes = boxes[box_indices]
            label_boxes = label_boxes.reshape(len(label_boxes) * 4, 2)
            min_area_rect = cv2.minAreaRect(label_boxes)
            box = cv2.boxPoints(min_area_rect)
            box = np.int64(box)

            collector.append((label, box))
    return collector


class Detector:
    ccClf = CCConvClf()
    """Detect card names bounding from the given input image

    Parameters
    ----------
    imgGray : numpy array
        Short int (0-255) valued grayscale image of size 1024x1024x1
    imgColor : numpy array
        Short int (0-255) valued BGR color image of size 1024x1024x3
    kpsCount : int
        Maximum count of FASText keypoints that are detected
    scaleCount : int
        How many scales are used in the FASText point scale pyramid in addition of the original scale
    ftThreshold : int
        Threshold use when defining a pixel is a FT keypoint or not
    ccSize : int
        Maximum width or height for a detected bounding box of connected component
    ccThreshold : float [0, 1]
        Classification threshold for the convnet classfier that classifies connected components into name text and not name text
    eps : float
        Epsilon or distance used with DBSCAN algorithm
    minPts : int
        MinPts used with DBSCAN algorithm
    positives : bool
        Are positive ("bright") FASText keypoints used
    negatives : bool
        Are negative ("dark") FASText keypoints used

    Returns
    -------
    labels_boxes : list
        List of tuples that represent the found bounding boxes in format (cluster label, bounding box [np int64 array of size 4 x 2])
    """
    def getNameBoxes(self, imgGray, imgColor, kpsCount = 1000, scaleCount = 3, ftThreshold = 28, ccSize = 40, ccThreshold = 0.41, eps=61, minPts=13, positives=False, negatives=True):

        # Get axis aligned bounding boxes for    
        segmentations = ftpy.getKpBoxes(imgGray, kpsCount, scaleCount, ftThreshold, positives, negatives, ccSize, ccSize)

        ccContainer = np.zeros((len(segmentations), 24, 24, 3))
        for i, seg in enumerate(segmentations):
            sample = extract_cc_sample(imgColor, seg)
            ccContainer[i] = sample

        assert len(segmentations) > 0

        # Classify the images of connected components are parts of name or not
        predictions = self.ccClf.predict(ccContainer).reshape(len(segmentations))
        positiveIndices = np.where(predictions >= ccThreshold)
        positiveSegs = segmentations[positiveIndices]

        lefts = positiveSegs[:,0]
        tops = positiveSegs[:,1]
        widths = positiveSegs[:,2]
        heights = positiveSegs[:,3]
        rights = lefts + widths
        bottoms = tops + heights
        boxes = np.column_stack((lefts, tops, rights, tops, rights, bottoms, lefts, bottoms)).astype("int32")

        # Cluster together positive cc boxes with customized DBSCAN
        labels = ftpy.kpBoxDBSCAN(boxes, eps=eps, min_samples=minPts)
        return extract_cluster_boxes(labels, boxes)
