#include <array>
#include <vector>
#include "constants.h"

using std::vector;
using std::array;

bool compareKp(array<int, 6> a, array<int, 6> b);
vector<array<int, 6>> getNMSKeypoints3x3(int count, int* icollector, bool positives, bool negatives);
vector<array<int, 6>> getNMSKeypoints5x5(int count, int* icollector, bool positives, bool negatives);
