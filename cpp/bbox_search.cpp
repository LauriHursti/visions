#include <array>
#include <bitset>
#include <chrono>
#include <future>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdint.h>

#include "constants.h"
#include "bbox_search.h"

using std::array;
using std::async;
using std::bitset;
using std::cout;
using std::endl;
using std::future;
using std::max;
using std::min;
using std::vector;

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

const int nMinX = 0;
const int nMaxX = DIM_SIZE - 1;
const int nMinY = 0;
const int nMaxY = DIM_SIZE - 1;

bool isVisited(bitset<FLAT_SIZE>* visited, int y, int x)
{
  // Safety check for indices
  if (y < nMaxY && y > nMinY && x < nMaxX && x > nMinX)
  {
    return visited->test((y * DIM_SIZE) + x);
  }
  else
  {
    return true;
  }
}

void setVisited(bitset<FLAT_SIZE>* visited, int y, int x)
{
  visited->set((y * DIM_SIZE) + x, true);
}

int scanLeft(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int y, int loThresh, int hiThresh)
{
  if (isVisited(visited, y, x1))
  {
    return x1;
  }
  else
  {
    int xL;
    uint8_t* row = &im[y * DIM_SIZE];
    for(xL = x1; xL >= 0; --xL)
    {
      setVisited(visited, y, xL);
      int pix = row[xL];
      if (!(pix >= loThresh && pix <= hiThresh))
      {
        box->minx = min(box->minx, xL+1);
        break;
      }
    }
    return xL;
  }
}

int scanRight(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int y, int loThresh, int hiThresh)
{
  if (isVisited(visited, y, x1))
  {
    return x1;
  }
  else
  {
    int xR;
    uint8_t* row = &im[y * DIM_SIZE];
    for(xR = x1; xR <= nMaxX; ++xR)
    {
      setVisited(visited, y, xR);
      int pix = row[xR];
      if (!(pix >= loThresh && pix <= hiThresh))
      {
        box->maxx = max(box->maxx, xR-1);
        break;
      }
    }
    return xR;
  }
}

int scanMiddle(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int x2, int y, int loThresh, int hiThresh)
{
  if (isVisited(visited, y, x1))
  {
    return x1;
  }
  else
  {
    int xM;
    uint8_t* row = &im[y * DIM_SIZE];
    for(xM = x1; xM <= x2 && xM <= nMaxX; ++xM)
    {
      setVisited(visited, y, xM);
      int pix = row[xM];
      if (!(pix >= loThresh && pix <= hiThresh))
      {
        break;
      }
    }
    return xM;
  }
}

// Box is looking to be too wide
bool boxLimit(bbox* box, int wLimit, int hLimit)
{
  return (box->maxy - box->miny) > hLimit || (box->maxx - box->minx) > wLimit;
}

// Box has at leas some pixels
bool boxMinLimit(bbox* box, int minSize)
{
  return (box->maxy - box->miny) >= minSize && (box->maxx - box->minx) >= minSize;
}

void lineCornerSearch(uint8_t* im, bitset<FLAT_SIZE>* visited, bbox* box, int x1, int x2, int y, int low, int high, int wLimit, int hLimit)
{
    int xL, xR, xM;

    // If upper and lower limit of the image are reached
    if( y < nMinY || y > nMaxY || boxLimit(box, wLimit, hLimit) || isVisited(visited, y, x1) || isVisited(visited, y, x2))
    {
        return;
    }

    box->miny = min(box->miny, y);
    box->maxy = max(box->maxy, y);    

    xL = scanLeft(im, visited, box, x1, y, low, high);
    // If the left scan went to new areas, add children
    if(xL < x1)
    {
        lineCornerSearch(im, visited, box, xL, x1, y-1, low, high, wLimit, hLimit);
        lineCornerSearch(im, visited, box, xL, x1, y+1, low, high, wLimit, hLimit);
        ++x1;
    }

    xR = scanRight(im, visited, box, x2, y, low, high);
    // If right scan went to new areas, add children
    if(xR > x2)
    {
        lineCornerSearch(im, visited, box, x2, xR, y-1, low, high, wLimit, hLimit);
        lineCornerSearch(im, visited, box, x2, xR, y+1, low, high, wLimit, hLimit);
        --x2;
    }

    // Scan the pixels in the middle.
    do
    {
      xM = scanMiddle(im, visited, box, x1, x2, y, low, high);
      if (xM > x1)
      {
        lineCornerSearch(im, visited, box, x1, xM, y-1, low, high, wLimit, hLimit);
        lineCornerSearch(im, visited, box, x1, xM, y+1, low, high, wLimit, hLimit);
        x1 = xM + 1;
      }
      else
      {
        x1++;
      }
    } while( x1 < x2);
}

vector<bbox> getCorners(uint8_t* im, vector<array<int, 6>>* kps, int istart, int iend, int wLimit, int hLimit)
{
  vector<bbox> boxes = {};
  bitset<FLAT_SIZE> visited;

  for (int i = istart; i < iend; i++)
  {
    int lightness = (*kps)[i][3];
    int y = (*kps)[i][0];
    int x = (*kps)[i][1];
    int intensity = int(im[(y * DIM_SIZE) + x]);
    int diff = (*kps)[i][5];
    int low = lightness == POSITIVE ? intensity - diff : 0;
    int high = lightness == POSITIVE ? 255 : intensity + diff;

    bbox box = {DIM_SIZE, 0, DIM_SIZE, 0, (lightness == POSITIVE ? low : high)};

    lineCornerSearch(im, &visited, &box, x, x, y, low, high, wLimit, hLimit);
    if (!boxLimit(&box, wLimit, hLimit) && boxMinLimit(&box, BOX_MIN_DIM))
    {
      boxes.push_back(box);
    }
    visited.reset();
  }

  return boxes;
}

vector<bbox> getCornersThreaded(uint8_t* im, vector<array<int, 6>> kps, int wLimit, int hLimit)
{
#ifdef PRINT_TIMES
  auto start = high_resolution_clock::now();
#endif

  int size = kps.size();
  int s = floor(size/4);
  future<vector<bbox>> fut1 = async(getCorners, im, &kps, 0, s*1, wLimit, hLimit);
  future<vector<bbox>> fut2 = async(getCorners, im, &kps, s*1, s*2, wLimit, hLimit);
  future<vector<bbox>> fut3 = async(getCorners, im, &kps, s*2, s*3, wLimit, hLimit);
  future<vector<bbox>> fut4 = async(getCorners, im, &kps, s*3, size, wLimit, hLimit);

  vector<bbox> b1 = fut1.get();
  vector<bbox> b2 = fut2.get();
  vector<bbox> b3 = fut3.get();
  vector<bbox> b4 = fut4.get();

  b1.insert(b1.end(), b2.begin(), b2.end());
  b1.insert(b1.end(), b3.begin(), b3.end());
  b1.insert(b1.end(), b4.begin(), b4.end());

#ifdef PRINT_TIMES
  auto stop = high_resolution_clock::now();
  auto duration2 = duration_cast<milliseconds>(stop - start);
  cout << "Boxes: " << duration2.count() << endl;
#endif  
  return b1;
}
