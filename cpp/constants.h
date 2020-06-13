#pragma once
#include <vector>

using std::vector;

typedef vector<vector<int>> keypoints;

const char NOT_KP = -1;
const char END_KP = 1;
const char BEND_KP = 2;
const char POSITIVE = 1;
const char NEGATIVE = 2;
const float SCALING_FACTOR = 1.6;
const long DIM_SIZE = 1024;
const long FLAT_SIZE = DIM_SIZE * DIM_SIZE;
const long MAXPOINTS = 4000;
const long SIZE_LIMIT = 120;
const long BOX_MIN_DIM = 2;

struct bbox {
  int minx;
  int maxx;
  int miny;
  int maxy;
  int threshold;
};

struct lineSegment {
  int x1;
  int x2;
  int y;
};
