import cv2
from os import path
import glob
import numpy as np
import io
import argparse

from tabulate import tabulate
from symspell import SymspellMTGNames
from lstm_reader import LSTMClf
from detector import Detector
import crop

IMGS = 500
COLORS = {
 0: (230, 25, 75),
 1: (60, 180, 75),
 2: (255, 225, 25),
 3: (0, 130, 200),
 4: (245, 130, 48),
 5: (145, 30, 180),
 6: (70, 240, 240),
 7: (240, 50, 230),
 8: (210, 245, 60),
 9: (250, 190, 190),
}


def boxes_to_res_file(imgname, labels_boxes):
    name_wo_type = imgname.replace(".jpg", "")
    lines = ""
    for i, (_label, box, _threshold) in enumerate(labels_boxes):
        res_pts = box.reshape(8).tolist()
        separator = ", "
        box_str_list = map(lambda x: str(x), res_pts)
        box_str = separator.join(box_str_list)
        newline = "\n" if i > 0 else ""
        lines = lines + newline + box_str

    output = io.open("outputs/" + name_wo_type + ".txt", "w", encoding="utf-8")
    output.write(lines)
    output.close()


def printImageOutputs(imgc, imName, labels_boxes):
    for _label, box,in boxes:
        cv2.drawContours(imgc, [box], 0, (255, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.imwrite("outputs/" + imName + ".png", imgc)


# Find all input images
def find_inputs():
    return glob.glob("inputs/*.*")


def recognize_card_names():
    sym = SymspellMTGNames()
    lstm = LSTMClf()
    detector = Detector()
    allResults = []

    imgPaths = find_inputs()
    for imgPath in imgPaths:
        _dir, imgName = path.split(imgPath)
        img = cv2.imread(str(imgPath), cv2.IMREAD_GRAYSCALE)
        imgc = cv2.imread(str(imgPath))
        labels_boxes = detector.getNameBoxes(img, imgc)

        cards = []
        boxes = []

        for _label, box in labels_boxes:
            cropped = None
            try:
                cropped = crop.crop(img, box)
                # Exception is thrown if the box has invalid size    
            except:
                continue

            boxes.append(box)
            cards.append("")

            prediction = lstm.read_image(cropped.transpose()) # Transpose to column major format
            if len(prediction) > 0:

                # Add detection if there's a valid card name left after post processing
                predStr = prediction.lower().lstrip()
                if len(predStr) > 0:
                    corrected, _distance = sym.lookup(predStr)
                    if corrected != None:
                        cards[len(cards) - 1] = corrected

        allResults.append((imgName, cards, boxes))

    return allResults


if __name__ == "__main__":
    results = recognize_card_names()
    headers = ["Card name"]
    for imgName, cards, boxes in results:
        print("Image: ", imgName)
        print(cards, "\n")
