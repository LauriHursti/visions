import numpy as np
import os.path as path
import cv2


def combineBoxes(a, b):
    l1 = a[0][0]
    r1 = a[1][0]
    t1 = a[0][1]
    b1 = a[2][1]
    l2 = b[0][0]
    r2 = b[1][0]
    t2 = b[0][1]
    b2 = b[2][1]
    left = min(l1, l2)
    right = max(r1, r2)
    top = min(t1, t2)
    bottom = max(b1, b2)
    return np.int64([left, top, right, top, right, bottom, left, bottom]).reshape(4, 2)


# Write box output
def drawsegmentation(image, seg, color=(0, 0, 255)):
    # [bbox.x, bbox.y, bbox.width, bbox.height, keyPoint.pt.x, keyPoint.pt.y, octave, ?, duplicate, quality, [keypointsIds]]
    left = seg[0]
    top = seg[1]
    right = left + seg[2]
    bottom = top + seg[3]
    poly = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
    drawbox(image, poly, color)


# Write box output
def drawsegmentations(image, segmentations, color=(0, 0, 255)):
    # [bbox.x, bbox.y, bbox.width, bbox.height]
    for i in range(segmentations.shape[0]):
        drawsegmentation(image, segmentations[i, :])
        rect = segmentations[i, :]
        left = rect[0]
        top = rect[1]
        right = left + rect[2]
        bottom = top + rect[3]
        poly = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
        drawbox(image, poly, color)


# Generic function for drawing quadrangles
def drawbox(image, poly, color = (255, 50, 0)):
    points = poly.reshape(1, 4, 2).astype("int32")
    return cv2.polylines(image, points, True, color=color, thickness=2, lineType=cv2.LINE_AA)


# Draw all points in a loop
def drawpoints(image, pts):
    for pt in pts:
        drawpoint(image, pt)


# Generic function for drawing a point
def drawpoint(image, pt, color = (255, 50, 0)):
    cv2.circle(image, (pt[0], pt[1]), 2, color, -1)