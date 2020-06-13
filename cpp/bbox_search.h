#include <array>
#include <bitset>
#include <vector>
#include "constants.h"

using std::bitset;
using std::array;
using std::vector;

vector<bbox> getCorners(uint8_t* im, vector<array<int, 6>>* kps, int istart, int iend, int wLimit, int hLimit);
vector<bbox> getCornersThreaded(uint8_t* im, vector<array<int, 6>> kps, int wLimit, int hLimit);

bool isVisited(bitset<FLAT_SIZE> visited, int y, int x);
void setVisited(bitset<FLAT_SIZE> visited, int y, int x);
bool boxLimit(bbox* box, int wLimit, int hLimit);

int scanLeft(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int y, int loThresh, int hiThresh);
int scanRight(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int y, int loThresh, int hiThresh);
int scanMiddle(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int x2, int y, int loThresh, int hiThresh);
void lineCornerSearch(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int x2, int y, int low, int high, int wLimit, int hLimit);
