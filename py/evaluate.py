import cv2
from os import makedirs, path, removedirs
import glob
import numpy as np
import random
import io
import crop
import statistics
import time

from symspell import SymspellMTGNames
from ctc_clf import CTCClf
from draw_boxes import drawsegmentation
from segment import getNameBoxes

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

    output = io.open("res/res_" + name_wo_type + ".txt", "w", encoding="utf-8")
    output.write(lines)
    output.close()


def draw_clusters(imgc, boxes):
    for label, box, _t in boxes:
        # (R, G, B) = COLORS[label % 10] if label != -1 else (0, 0, 0)
        cv2.drawContours(imgc, [box], 0, (255, 0, 255), 3, lineType=cv2.LINE_AA)


ctc = CTCClf()
def evaluateCTC(cropThresholdsAndLabels, sym, log=True):
    # ctc = CTCClf()
    truePositives = 0
    cards = 0
    dets = 0
    fullSuccess = 0

    ctcMsCollector = []
    symspellMsCollector = []
    totalRecCollector = []

    for cropThresholds, labels in cropThresholdsAndLabels:
        totalDistance = 0
        detections = []
        raws = []
        innerCtcMsCollector = []
        innerSymspellMsCollector = []
        totalStart = int(round(time.time() * 1000))

        for cropped, _threshold in cropThresholds:
            """wordBoxes, _eroded = getWordBoxes(cropped, threshold)

            predictions = []          
            for box in wordBoxes:
                wordCrop = crop.crop(cropped, box)
                predictions.append(ctc.read_image(wordCrop.transpose()))
            """

            millis = int(round(time.time() * 1000))
            prediction = ctc.read_image(cropped.transpose())
            millisAfter = int(round(time.time() * 1000))
            ctcMillis = millisAfter - millis
            innerCtcMsCollector.append(ctcMillis)

            millis = int(round(time.time() * 1000))
            if len(prediction) > 0:
                raws.append(prediction)
                predStr = prediction.lower().lstrip()
                
                if len(predStr) > 0:
                    corrected, distance = sym.lookup(predStr)
                    if corrected != None:
                        totalDistance += distance
                        detections.append(corrected)
            millisAfter = int(round(time.time() * 1000))
            symspellMillis = millisAfter - millis
            innerSymspellMsCollector.append(symspellMillis)

        # Calculate precision and recall for current card
        tpCurrent = 0
        detLen = len(detections)
        cardCount = len(labels)
        detCopy = detections.copy()
        for label in labels:
            labelLow = label.lower()
            if labelLow in detections:
                tpCurrent += 1
                # Remove the positive detections to make sure it can't be used again e.g. if there's multiples of same card in gt
                detections.remove(labelLow)

        truePositives += tpCurrent
        dets += detLen
        cards += cardCount
        
        totalEnd = int(round(time.time() * 1000))
        totalDur = totalEnd - totalStart
        ctcMsCollector.append(sum(innerCtcMsCollector))
        symspellMsCollector.append(sum(innerSymspellMsCollector))
        totalRecCollector.append(totalDur)

        if tpCurrent/max(1, detLen) < 0.1 or tpCurrent/cardCount < 0.1:
            if log:
                print("Bad match!")
                print("Raws:", raws)
                print("Dets:", detCopy)
                print("GT:", labels, "\n")
        else:
            if totalDistance <= 0 and cardCount >= 4:
                if log:
                    print("Good match!")
                    print("Raws:", raws)
                    print("Distance: ", totalDistance)
                    print("GT:", labels, "\n")
            fullSuccess += 1

    # Calculate precision and recall
    precision = truePositives / dets
    recall = truePositives / cards

    return round(precision, 3), round(recall, 3), round(statistics.mean(ctcMsCollector), 1), round(statistics.mean(symspellMsCollector), 1), round(statistics.mean(totalRecCollector), 1), fullSuccess/IMGS


def printImageOutputs(imgc, imName, labels_boxes, positiveSegs, cropThresholds):
    # Draw high-level segmentations and positive FASText cc's

    for seg in positiveSegs:
        # rgb(173,255,47)
        drawsegmentation(imgc, seg, (0, 255, 0))
    draw_clusters(imgc, labels_boxes)

    cv2.imwrite("imgs_out/" + imName + "_out.png", imgc)

    for i, (cropped, threshold) in enumerate(cropThresholds):

        # Draw each cropped region with cc and word segmentations
        colorCrop = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        colorCrop1 = cv2.copyMakeBorder(colorCrop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(225, 225, 225))
        h, w, _d = colorCrop1.shape
        colorCrop1 = cv2.resize(colorCrop1, dsize=(w*2, h*2))

        colorCrop2 = cv2.copyMakeBorder(colorCrop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(225, 225, 225))
        colorCrop2 = cv2.resize(colorCrop2, dsize=(w*2, h*2))
        
        cropname = imName + "_crop_" + str(i)
        cv2.imwrite("imgs_out/crops/" + cropname + ".jpg", colorCrop2)
        


# Find all file names for dataset images
def findimagepaths():
    return glob.glob("imgs/*.jpg")
    


# Get gt card names for an image
def getGTCards(id):
    path = "gt/gt_" + id + ".txt"
    gtFile = open(path, "r")

    # Collect gt shapes
    names = []
    for line in gtFile:
        parts = line.split(",")
        nameWords = parts[8:]
        sep = ","
        name = sep.join(nameWords)
        nameNoBreak = name.replace("\n", "")
        if nameNoBreak in CARD_NAMES:
            names.append(nameNoBreak)
        #else:
        #    print("Not in card names: ", nameNoBreak)    
    return names


# Get gt polys and names for an image
def getGTPolysAndCards(id):
    path = "gt/gt_" + id + ".txt"
    gtFile = open(path, "r")

    collector = []
    for line in gtFile:
        parts = line.split(",")

        p = list(map(lambda x: int(x), parts[:8]))
        gtPts = np.int64([[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]])

        nameWords = parts[8:]
        sep = ","
        name = sep.join(nameWords)
        nameNoBreak = name.replace("\n", "")

        if nameNoBreak in CARD_NAMES:
            collector.append((nameNoBreak, gtPts))
  
    return collector

sym = SymspellMTG()
def testPipe(kpsCount = 1000, scaleCount = 3, ftThreshold = 28, ccSize = 40, ccThreshold = 0.41, eps=61, min_samples=13, log=False, imgPaths=findimagepaths()):
    makedirs("imgs_out/", exist_ok=True)
    makedirs("imgs_out/crops", exist_ok=True)
    makedirs("imgs_out/chars", exist_ok=True)
    makedirs("imgs_out/erosions", exist_ok=True)

    start = int(round(time.time() * 1000))
    ftMsCollector = []
    clfMsCollector = []
    dbscanMsCollector = []
    cropThresholdsAndLabels = []
    detectionMsCollector = []
    # sym = SymspellMTG()

    for imgPath in imgPaths:
        _dir, imgName = path.split(imgPath)
        imgNameNoType = imgName.replace(".jpg", "")
        img = cv2.imread(str(imgPath), cv2.IMREAD_GRAYSCALE)
        imgc = cv2.imread(str(imgPath))

        try:
            detStart = int(round(time.time() * 1000))
            labels_boxes, positiveSegs, ftmillis, clfmillis, dbscanMillis = getNameBoxes(img, imgc, kpsCount, scaleCount, ftThreshold, ccSize, ccThreshold, eps, min_samples)
            detEnd = int(round(time.time() * 1000))
            detectionMsCollector.append(detEnd - detStart)
        except:
            continue

        clfMsCollector.append(clfmillis)
        ftMsCollector.append(ftmillis)
        dbscanMsCollector.append(dbscanMillis)

        labels = getGTCards(imgNameNoType)
        cropThresholds = []
      
        for _dbscanLabel, box, threshold in labels_boxes:
            cropped = None
            try:
                cropped = crop.crop(img, box)
                cropThresholds.append((cropped, threshold))
            # Exception is thrown if the box has invalid size    
            except:
                continue
        
        cropThresholdsAndLabels.append((cropThresholds, labels))

        if log:
            printImageOutputs(imgc, imgNameNoType, labels_boxes, positiveSegs, cropThresholds)

    precision, recall, ctcMeanDuration, symspellMeanDuration, totalRecDuration, successPercentage =  evaluateCTC(cropThresholdsAndLabels, sym, log=log)
    end = int(round(time.time() * 1000))

    ftMeanDuration = statistics.mean(ftMsCollector)
    ccClfMeanDuration = statistics.mean(clfMsCollector)
    dbscanMeanDuration = statistics.mean(dbscanMsCollector)
    totalDuration = round((end - start)/ IMGS, 1)
    detDuration = statistics.mean(detectionMsCollector)

    # if log:
    print("Recognition results:", precision, recall, successPercentage)

    # print("Tesseract", evaluateTesseract(cropThresholdsAndLabels, sym))
    # print("HOG:", evaluateHOGClf(cropThresholdsAndLabels, sym))
    print("--- Durations ---")
    print("Total duraction: ", totalDuration)
    print("---")
    print("FASText mean: ", ftMeanDuration)
    print("CC classifier mean: ", ccClfMeanDuration)
    print("Dbscan mean: ", dbscanMeanDuration)
    print("Detection total mean: ", detDuration)
    print("---")
    print("CTC mean: ", ctcMeanDuration)
    print("Symspell mean: ", symspellMeanDuration)
    print("Total recognition mean: ", totalRecDuration)

    print("Finished")

    return precision, recall, successPercentage, ftMeanDuration, ccClfMeanDuration, ctcMeanDuration, symspellMeanDuration, totalDuration


# Test the full pipe with the parameters
def test_pipe(log=False):
    scaleCount = 3
    kpsCount = 1000
    threshold = 28
    size = 40
    ccClfThreshold = 0.41
    eps = 61
    minSamples = 13
    # imgPaths=["imgs/IMG_20191130152227.jpg"]
    imgPaths=random.sample(findimagepaths(), IMGS)
    prec, rec, succPerc, ftDur, ccClfDur, ctcDur, symDur, totalDur = testPipe(kpsCount, scaleCount, threshold, size, ccClfThreshold, eps, minSamples, log, imgPaths)
    headers = ["Kps n", "Scale n", "Kp th", "Cc size", "Cc clf th", "Eps", "Min samples", "Prec", "Rec", "Succ. %", "Ft dur", "CC clf dur", "Ctc dur", "Total dur"]
    res = [kpsCount, scaleCount, threshold, size, ccClfThreshold, eps, minSamples, prec, rec, succPerc, ftDur, ccClfDur, ctcDur, totalDur]
    print(tabulate([res], headers=headers))


def test_recognition(log=False):
    imgPaths=random.sample(findimagepaths(), IMGS)
    cropThresholdsAndLabels = []

    for imgPath in imgPaths:
        _dir, imgName = path.split(imgPath)
        imgNameNoType = imgName.replace(".jpg", "")
        img = cv2.imread(str(imgPath), cv2.IMREAD_GRAYSCALE)

        labelsPolys = getGTPolysAndCards(imgNameNoType)
        cropThresholds = []
        labels = []

        for label, poly in labelsPolys:
            labels.append(label)
            try:
                cropped = crop.crop(img, poly)
                cropThresholds.append((cropped, 0))
            # Exception is thrown if the box has invalid size    
            except:
                continue
            
        cropThresholdsAndLabels.append((cropThresholds, labels))

    precision, recall, ctcMeanDuration, symspellMeanDuration, totalRecDuration, successPercentage =  evaluateCTC(cropThresholdsAndLabels, sym, log=log)

    print("Recognition results:", precision, recall, successPercentage)

    print("Tesseract", evaluateTesseract(cropThresholdsAndLabels, sym))
    # print("HOG:", evaluateHOGClf(cropThresholdsAndLabels, sym))
    print("--- Durations ---")
    print("CTC mean: ", ctcMeanDuration)
    print("Symspell mean: ", symspellMeanDuration)
    print("Total recognition mean: ", totalRecDuration)

    print("Finished")

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Success percentage: ", successPercentage)


if __name__ == "__main__":
    test_pipe(True)
    # test_recognition(True)
    # writeAllSegmentations()
    # test_detection()
    # test_distance_metrics()
