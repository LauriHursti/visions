#include <bitset>
#include <iostream>
#include <tuple>  

#include "bbox_search.h"
#include "dbscan.h"
#include "ft_api.h"
#include "constants.h"
#include "keypoint_nms.h"
#include "keypoints.h"
#include <stdint.h>

using std::array;
using std::bitset;
using std::cout;
using std::endl;
using std::vector;
using std::tuple;
using std::make_tuple;

vector<array<int, 6>> getFASTextKeypoints(uint8_t* image, int count, int scales, int threshold, bool includePositive, bool includeNegative)
{
  int* icollector = new int[FLAT_SIZE * 6]{0};
  multiScaleKps(image, icollector, DIM_SIZE, scales, threshold);
  vector<array<int, 6>> kps = getNMSKeypoints5x5(count, icollector, includePositive, includeNegative);
  delete[] icollector;
  return kps;
}

// Alternative for "getFASTextKeypoints" that returns data in a contiguous, plain in array and the amount of keypoints
tuple<int*, int> py_getFASTextKeypoints(uint8_t* image, int count, int scales, int threshold, bool includePositive, bool includeNegative)
{
  int* icollector = new int[FLAT_SIZE * 6]{0};
  multiScaleKps(image, icollector, DIM_SIZE, scales, threshold);
  vector<array<int, 6>> kps = getNMSKeypoints5x5(count, icollector, includePositive, includeNegative);
  int* kpsArr = new int[kps.size() * 5];
  for (int i = 0; i < kps.size(); i++)
  {
    array<int, 6> kp = kps[i];
    int s = i * 5;
    // Flip x and y to keep python API consistent
    // Contrast is dropped because NMS is done already in C++ side
    kpsArr[s] = kp[1]; // x
    kpsArr[s+1] = kp[0]; // y
    kpsArr[s+2] = kp[2]; // keypoint type (end = 1, bend = 2)
    kpsArr[s+3] = kp[3]; // keypoint lightness (positive = 1, negative = 2)
    kpsArr[s+4] = kp[5]; // diff for thresholding the CC
  }
  delete[] icollector;
  return make_tuple(kpsArr, kps.size());
}

// Alternative for "getFASTextBoxes" that returns data in a contiguous, plain int array and the amount of boxes
tuple<int*, int> py_getFASTextBoxes(uint8_t* image, vector<array<int, 6>> kps, int wLimit, int hLimit)
{
  vector<bbox> boxes = getCornersThreaded(image, kps, wLimit, hLimit);
  int* boxesArr = new int[boxes.size() * 5];
  for (int i = 0; i < boxes.size(); i++)
  {
    bbox box = boxes[i];
    int s = i * 5;
    boxesArr[s] = box.minx; // x
    boxesArr[s+1] = box.miny; // y
    boxesArr[s+2] = box.maxx - box.minx; // width
    boxesArr[s+3] = box.maxy - box.miny; // height
    boxesArr[s+4] = box.threshold; // threshold
  }
  return make_tuple(boxesArr, boxes.size());
}

// Cluster FASText connected component bounding boxes to (hopefully) find words
tuple<int16_t*, int> py_getCompClusters(int32_t* boxes, int boxCount, float eps, int min_samples)
{
  // Collect the raw boxes array into a vector of bboxes
  vector<ClusterBBox> vboxes = {};
  for (int i = 0; i < boxCount; i++)
  {
    int32_t* b = &boxes[8 * i];
    // Order here again as a reminder: left, top, right, top, right, bottom, left, bottom
    ClusterBBox box = {b[0], b[2], b[1], b[5], UNCLASSIFIED};
    vboxes.push_back(box);
  }
  DBSCAN scanner = DBSCAN(min_samples, eps, vboxes);
  scanner.run();
  vector<ClusterBBox> wLabels = scanner.getBoxes();

  // Collect labels into a raw format
  int n = wLabels.size();
  int16_t* labels = new int16_t[n];
  for (int i = 0; i < n; i++)
  {
    labels[i] = int16_t(wLabels[i].clusterID);
  }
  return make_tuple(labels, n);
}