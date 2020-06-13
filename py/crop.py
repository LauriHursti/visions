import cv2
import numpy as np
import numpy.linalg


# Crop the segmented input from image and resize it to 32x(scaled width)
def crop(image, box):
    # Box points are in arbitrary order so some fiddling is required
    # The code assumes that start of the name is always more left than the end of the name
    xs = box[:,0]
    ys = box[:,1]

    xorder = np.argsort(xs)

    # Pick leftmost point and check if it's top left or bottom left
    left1 = box[xorder[0]]
    left2 = box[xorder[1]]
    right1 = box[xorder[3]]
    right2 = box[xorder[2]]

    # Check which point is top left and which bottom left, same for the right side
    topleft = left1 if left1[1] < left2[1] else left2
    bottomleft = left2 if left1[1] < left2[1] else left1
    topright = right1 if right1[1] < right2[1] else right2

    height = np.linalg.norm(topleft-bottomleft)
    width = np.linalg.norm(topright-topleft)
    if height == 0 or width == 0:
        raise RuntimeError("Cropped area size can't be zero")

    # Width is assumed to be larger than height
    oheight = 32
    scale = oheight / height
    newWidth = int(width * scale)

    oldpoints = np.float32([topleft, bottomleft, topright])
    newpoints = np.float32([[0, 0], [0, oheight], [newWidth, 0]])

    canvas = np.zeros((oheight, newWidth)).astype("uint8")
    canvas.fill(235)
    tmatrix = cv2.getAffineTransform(oldpoints, newpoints)
    canvas[0:oheight, 0:newWidth] = cv2.warpAffine(image, tmatrix, (newWidth, oheight))

    return canvas


# Crop a slice from image without any transformations
def simpleCrop(image, start, width):
    height, _srcw = image.shape
    slicecrop = np.zeros((height, width)).astype("uint8")
    slicecrop[0:height, 0:width] = image[0:height, start:(start+width)]
    return slicecrop
