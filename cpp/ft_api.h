#include <array>
#include <tuple>
#include <vector>
#include "constants.h"

using std::array;
using std::tuple;
using std::vector;

vector<array<int, 6>> getFASTextKeypoints(uint8_t* image, int count, int scales, int threshold, bool includePositive, bool includeNegative);
vector<bbox> getFASTextBoxes(uint8_t* image, vector<array<int, 6>> kps, int wLimit, int hLimit);
tuple<int*, int> py_getFASTextKeypoints(uint8_t* image, int count, int scales, int threshold, bool includePositive, bool includeNegative);
tuple<int*, int> py_getFASTextBoxes(uint8_t* image, vector<array<int, 6>> kps, int wLimit, int hLimit);
tuple<int16_t*, int> py_getCompClusters(int32_t* boxes, int boxCount, float eps, int min_samples);