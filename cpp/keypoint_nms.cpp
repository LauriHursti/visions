#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <vector>

#include "constants.h"
#include "keypoint_nms.h"

using std::array;
using std::cout;
using std::endl;
using std::max;
using std::min;
using std::sort;
using std::vector;

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

// Compares keypoints with by contrast
bool compareKp(array<int, 6> a, array<int, 6> b)
{
    return (a[4] > b[4]);
}

int getContr(int* icollector, int row, int column)
{
  return icollector[(row * DIM_SIZE * 6) + (column * 6) + 4];
}

int getLightness(int* icollector, int row, int column)
{
  return icollector[(row * DIM_SIZE * 6) + (column * 6) + 3];
}

// Apply nom-maximal suppression to 3x3 neighbourhoods in the collector array
vector<array<int, 6>> getNMSKeypoints3x3(int count, int* icollector, bool positives, bool negatives)
{
#ifdef PRINT_TIMES 
  auto start = std::chrono::high_resolution_clock::now();
#endif

  vector<array<int, 6>> collector = {};
  for (int i = 2; i < DIM_SIZE - 2; i++)
  {
    for (int j = 2; j < DIM_SIZE - 2; j++)
    {
      // Get contrast and lightnes for current kp
      int c = getContr(icollector, i, j);
      int l = getLightness(icollector, i, j);
      if (c > 0 && ((positives && l == POSITIVE) || (negatives && l == NEGATIVE)))
      {
        // Check if any neighbour has higher contrast
        int n1 = getContr(icollector, i-1, j-1);
        int n2 = getContr(icollector, i-1, j);
        int n3 = getContr(icollector, i-1, j+1);

        int n4 = getContr(icollector, i, j-1);
        int n5 = getContr(icollector, i, j+1);

        int n6 = getContr(icollector, i+1, j-1);
        int n7 = getContr(icollector, i+1, j);
        int n8 = getContr(icollector, i+1, j+1);

        if (!(n1 > c || n2 > c || n3 > c || n4 > c || n5 > c || n6 > c || n7 > c || n8 > c))
        {
          int* kp = &icollector[(i * DIM_SIZE * 6) + (j * 6)];
          array<int, 6> entry = {kp[0], kp[1], kp[2], kp[3], kp[4], kp[5]};
          collector.push_back(entry);
        }
      }
    }
  }

#ifdef PRINT_TIMES
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "3x3 nms and filtering took: " << duration.count() << endl;

  auto start2 = high_resolution_clock::now();
#endif

  sort(collector.begin(), collector.end(), compareKp);
  int sliceCount = min(int(collector.size()), count);
  vector<array<int, 6>> slice = vector<array<int, 6>>(collector.begin(), collector.begin() + sliceCount);

#ifdef PRINT_TIMES
  auto stop2 = high_resolution_clock::now();
  auto duration2 = duration_cast<milliseconds>(stop2 - start2);
  cout << "Sorting and cutting took: " << duration2.count() << endl;
#endif

  return slice;
}

// Apply nom-maximal suppression to 5x5 neighbourhoods in the collector array
vector<array<int, 6>> getNMSKeypoints5x5(int count, int* icollector, bool positives, bool negatives)
{
#ifdef PRINT_TIMES 
  auto start = high_resolution_clock::now();
#endif

  vector<array<int, 6>> collector = {};
  for (int i = 2; i < DIM_SIZE - 1; i++)
  {
    for (int j = 2; j < DIM_SIZE - 1; j++)
    {
      // Get contrast and lightnes for current kp
      int c = getContr(icollector, i, j);
      int l = getLightness(icollector, i, j);
      if (c > 0 && ((positives && l == POSITIVE) || (negatives && l == NEGATIVE)))
      {
        // Check if any neighbour has higher contrast
        int n1 = getContr(icollector, i-2, j-2);
        int n2 = getContr(icollector, i-2, j-1);
        int n3 = getContr(icollector, i-2, j);
        int n4 = getContr(icollector, i-2, j+1);
        int n5 = getContr(icollector, i-2, j+2);

        int n6 = getContr(icollector, i-1, j-2);
        int n7 = getContr(icollector, i-1, j-1);
        int n8 = getContr(icollector, i-1, j);
        int n9 = getContr(icollector, i-1, j+1);
        int n10 = getContr(icollector, i-1, j+2);

        int n11 = getContr(icollector, i, j-2);
        int n12 = getContr(icollector, i, j-1);
        int n14 = getContr(icollector, i, j+1);
        int n15 = getContr(icollector, i, j+2);

        int n16 = getContr(icollector, i+1, j-2);
        int n17 = getContr(icollector, i+1, j-1);
        int n18 = getContr(icollector, i+1, j);
        int n19 = getContr(icollector, i+1, j+1);
        int n20 = getContr(icollector, i+1, j+2);

        int n21 = getContr(icollector, i+2, j-2);
        int n22 = getContr(icollector, i+2, j-1);
        int n23 = getContr(icollector, i+2, j);
        int n24 = getContr(icollector, i+2, j+1);
        int n25 = getContr(icollector, i+2, j+2);

        if (!(n1 > c || n2 > c || n3 > c || n4 > c || n5 > c || n6 > c || n7 > c || n8 > c || n9 > c || n10 > c \
            || n11 > c || n12 > c || n14 > c || n15 > c || n16 > c || n17 > c || n18 > c || n19 > c || n20 > c \
            || n21 > c || n22 > c || n23 > c || n24 > c || n25 > c))
        {
          int* kp = &icollector[(i * DIM_SIZE * 6) + (j * 6)];
          array<int, 6> entry = {kp[0], kp[1], kp[2], kp[3], kp[4], kp[5]};
          collector.push_back(entry);
        }
      }
    }
  }

#ifdef PRINT_TIMES
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "5x5 nms and filtering: " << duration.count() << endl;

  auto start2 = high_resolution_clock::now();
#endif

  sort(collector.begin(), collector.end(), compareKp);
  int sliceCount = min(int(collector.size()), count);
  vector<array<int, 6>> slice = vector<array<int, 6>>(collector.begin(), collector.begin() + sliceCount);
  
#ifdef PRINT_TIMES
  auto stop2 = high_resolution_clock::now();
  auto duration2 = duration_cast<milliseconds>(stop2 - start2);
  cout << "Sorting and cutting: " << duration2.count() << endl;
#endif

  return slice;
}
