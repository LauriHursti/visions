import cv2
from os import path, makedirs
import glob
import numpy as np
import io
import argparse

from symspell import SymspellMTGNames
from lstm_reader import LSTMClf
from detector import Detector
from crop import crop, getOrderedPoints


FT_INPUT_W = 1024
FT_INPUT_H = 1024


def drawBoxes(imgc, boxes, cards):
    for i, box in enumerate(boxes):
        card = cards[i]

        text = card if len(card) != 0 else "-not recognized-"
        _topleft, _bottomleft, topright, bottomright = getOrderedPoints(box)
        # Calculate position for text output
        origY = int((topright[1] + bottomright[1]) / 2)
        origX = int((topright[0] + bottomright[0]) / 2) + 5
        textOrig = (int(origX) + 5, int(origY))
        size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        w, h = size
        pad = 4
        top = max(0, (origY - h) - pad)
        bottom = min(FT_INPUT_H, origY + pad)
        left = max(0, origX - pad)
        right = min(FT_INPUT_W, origX + w + pad)

        # Create the text render
        textbg = imgc[top:bottom, left:right]
        whiteCanvas = np.empty_like(textbg)
        whiteCanvas.fill(255)
        merged = cv2.addWeighted(textbg, 0.3, whiteCanvas, 0.7, 1)
        cv2.putText(merged, text, (pad, pad + h), cv2.FONT_HERSHEY_DUPLEX, 1, (15, 15, 15), 1)

        # Overlay text into the image
        imgc[top:bottom, left:right] = merged
        cv2.drawContours(imgc, [box], 0, (255, 0, 255), 2, lineType=cv2.LINE_AA)
    return imgc        


def squareResize(img, grayScale=True):
    if grayScale:
        ih, iw = img.shape
    else:
        ih, iw, _d = img.shape      

    if ih > iw:
        scale = FT_INPUT_H / ih
        newH = int(scale * ih)
        newW = int(scale * iw)
        imgResized = cv2.resize(img, dsize=(newW, newH))
        imgResized = cv2.copyMakeBorder(imgResized, 0, 0, 0, FT_INPUT_W - newW, cv2.BORDER_CONSTANT)
    else:
        scale = FT_INPUT_W / iw
        newH = int(scale * ih)
        newW = int(scale * iw)
        imgResized = cv2.resize(img, dsize=(newW, newH))
        imgResized = cv2.copyMakeBorder(imgResized, 0, FT_INPUT_H - newH, 0, 0, cv2.BORDER_CONSTANT)

    if (grayScale):
        oh, ow = imgResized.shape
    else:
        oh, ow, _od = imgResized.shape

    assert oh == FT_INPUT_H
    assert ow == FT_INPUT_W
    return imgResized


def findInputs(folder):
    types = ["gif", "jpg", "jpeg", "tiff", "png", "webp", "bmp"]
    collector = []
    for ftype in types:
        globExpression = folder + "/*." + ftype
        collector.extend(glob.glob(globExpression))
    return collector


def recognizeCardNames(inputFolder, outputFolder, visualize):
    sym = SymspellMTGNames()
    lstm = LSTMClf()
    detector = Detector()
    allResults = []

    imgPaths = findInputs(inputFolder)
    for imgPath in imgPaths:
        _dir, imgName = path.split(imgPath)
        img = cv2.imread(str(imgPath), cv2.IMREAD_GRAYSCALE)
        imgc = cv2.imread(str(imgPath))
        img = squareResize(img)
        imgc = squareResize(imgc, grayScale=False)
        labels_boxes = detector.getNameBoxes(img, imgc)

        cards = []
        boxes = []

        for _label, box in labels_boxes:
            cropped = None
            try:
                cropped = crop(img, box)
            except:
                # Exception is thrown if the box has invalid size    
                continue

            boxes.append(box)
            cards.append("")

            prediction = lstm.read_image(cropped.transpose()) # Transpose to column major format
            if len(prediction) > 0:
                # Add detection if there's a valid card name left after post processing
                predStr = prediction.lstrip()
                if len(predStr) > 0:
                    corrected, _distance = sym.lookup(predStr)
                    if corrected != None:
                        cards[len(cards) - 1] = corrected

        allResults.append((imgName, imgc, cards, boxes))

    return allResults


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--input", default="inputs", help="Folder that contains input images")
    parser.add_argument("--visualize", default=0, help="Set to 1 if you want visualization outputs printed to the output folder")
    parser.add_argument("--output", default="outputs", help="Folder where image outputs are written")

    args = parser.parse_args()
    inputFolder = args.input
    outputFolder = args.output
    visualize = bool(args.visualize)

    if visualize:
        makedirs(outputFolder, exist_ok=True)

    results = recognizeCardNames(inputFolder, outputFolder, visualize)
    for imgName, imgc, cards, boxes in results:
        if visualize:
            outputPath = outputFolder + "/out_" + imgName
            imgc = drawBoxes(imgc, boxes, cards)
            cv2.imwrite(outputFolder + "/" + imgName, imgc)
            print("Output file written to", outputPath)
        print("Image: ", imgName)
        print(cards, "\n")
