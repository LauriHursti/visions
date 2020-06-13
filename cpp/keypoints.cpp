#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <vector>

#include "keypoints.h"
#include "constants.h"
#include "resize.h"

using std::array;
using std::async;
using std::cout;
using std::endl;
using std::future;
using std::max;
using std::min;
using std::vector;

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

bool oppositesTest(uint8_t center, uint8_t outers[], int threshold)
{
  for (int i = 0; i < 6; i++)
  {
    uint8_t pix1 = outers[i];
    uint8_t pix2 = outers[i+6];
    bool pix1dark = pix1 <= (center - threshold);
    bool pix1light = pix1 >= (center + threshold);
    bool pix2dark = pix2 <= (center - threshold);
    bool pix2light = pix2 >= (center + threshold);
    // All kind of cases where where point is not a FASText point
    bool dissimilar = (pix1dark && pix2light) || (pix1light && pix2dark);
    bool notkp = !(pix1dark || pix2dark || pix1light || pix2light);
    if (dissimilar || notkp)
    {
      return false;
    }
  }
  return true;
}

bool connectivityTest(uint8_t center, uint8_t outers[], uint8_t inners[], int threshold)
{
  for (int i = 0; i < 12; i++)
  {
    uint8_t outer = outers[i];
    uint8_t inner = inners[i];
    bool isDark = outer <= (center - threshold);
    bool isLight = outer >= (center + threshold);
    bool isInnerDarker = inner <= (center - threshold);
    bool isInnerLighter = inner >= (center + threshold);
    bool isSimilar = !isDark && !isLight;
    // Center point must have connectivity to similar outer points
    if (isSimilar && (isInnerDarker || isInnerLighter))
    {
        return false;
    }
  }
  return true;
}

int* getKpStats(uint8_t center, uint8_t outers[], int threshold)
{
  // Numbers for the first contiguous segment, it might have to be combined
  // with the ending segment if it is of same type
  int startingDark = 0;
  int startingLight = 0;
  int startingSimilar = 0;

  // Longest length trackers
  int maxDark = 0;
  int maxLight = 0;
  int maxSimilar = 0;
  int diff = 255;

  // Current length trackers
  int currentLight = 0;
  int currentDark = 0;
  int currentSimilar = 0;

  // Count trackers
  int nLight = 0;
  int nDark = 0;
  int nSimilar = 0;

  // Keep the maximum contrast in memory for the NMS algorithm
  int maxContrast = 0;

  for (int i = 0; i < 12; i++)
  {
    uint8_t outer = outers[i];
    bool isDark = outer <= (center - threshold);
    bool isLight = outer >= (center + threshold);
    bool isSimilar = !isDark && !isLight;
    maxContrast = max(abs(center - outer), maxContrast);

    // Point is similar
    if (isSimilar)
    {
      // Add count if new segment
      if (currentSimilar == 0)
      {
        nSimilar++;
      }
      currentSimilar++;
      currentLight = 0;
      currentDark = 0;
      maxSimilar = max(maxSimilar, currentSimilar);
    }
    else if (isDark)
    {
      if (currentDark == 0)
      {
        nDark++;
      }
      currentDark++;
      currentLight = 0;
      currentSimilar = 0;
      diff = min(abs(center - outer) - 1, diff);
      maxDark = max(maxDark, currentDark);
    }
    else if (isLight)
    {
      if (currentLight == 0)
      {
        nLight++;
      }
      currentLight++;
      currentDark = 0;
      currentSimilar = 0;
      diff = min(abs(center - outer) - 1, diff);
      maxLight = max(maxLight, currentLight);
    }

    // Still handling first segment
    if ((currentDark + currentLight + currentSimilar) == i+1)
    {
      startingDark = currentDark;
      startingLight = currentLight;
      startingSimilar = currentSimilar;
    }
  }

  if (currentLight > 0 && startingLight > 0)
  {
    nLight--;
    maxLight = min(12, max(maxLight, currentLight + startingLight));
  }
  else if(currentDark > 0 && startingDark > 0)
  {
    nDark--;
    maxDark = min(12, max(maxDark, currentDark + startingDark));
  }
  else if(currentSimilar > 0 && startingSimilar > 0)
  {
    nSimilar--;
    maxSimilar = min(12, max(maxSimilar, currentSimilar + startingSimilar));
  }

  // Dissimilarity is tested already in the oppositesTest so it's not checked here any more
  if (nLight == 1 && maxLight >= 9 && nSimilar == 1)
  {
    return new int[4]{END_KP, NEGATIVE, maxContrast, diff};
  }
  else if (nDark == 1 && maxDark >= 9 && nSimilar == 1)
  {
    return new int[4]{END_KP, POSITIVE, maxContrast, diff};
  }
  else if (nLight == 2 && maxLight >= 6 && nSimilar == 2)
  {
    return new int[4]{BEND_KP, NEGATIVE, maxContrast, diff};
  }
  else if (nDark == 2 && maxDark >= 6 && nSimilar == 2)
  {
    return new int[4]{BEND_KP, POSITIVE, maxContrast, diff};
  }
  else
  {
    return new int[4]{NOT_KP, NOT_KP, 0, 0};
  }
}

void parseKps(uint8_t* im, int* icollector, int size, float scale, int threshold, int istart, int iend)
{
  vector<array<int, 6>> collector = {};
  for (int i = istart; i < iend; i++)
  {
    uint8_t* r0 = &im[i*size];
    uint8_t* r1 = &im[(i+1)*size];
    uint8_t* r2 = &im[(i+2)*size];
    uint8_t* r3 = &im[(i+3)*size];
    uint8_t* r4 = &im[(i+4)*size];

    for (int j = 0; j < (size - 4); j++)
    {
      int c0 = j;
      int c1 = j + 1;
      int c2 = j + 2;
      int c3 = j + 3;
      int c4 = j + 4;

      // Center: the center pixel for this patch
      // Outer: The outer circle of 12 pixels
      // Inner: The inner circle with corner pixels duplicated
      uint8_t center = r2[c2];
      uint8_t outers[] = {r0[c1], r0[c2], r0[c3], r1[c4], r2[c4], r3[c4], r4[c3], r4[c2], r4[c1], r3[c0], r2[c0], r1[c0]};
      uint8_t inners[] = {r1[c1], r1[c2], r1[c3], r1[c3], r2[c3], r3[c3], r3[c3], r3[c2], r3[c1], r3[c1], r2[c1], r1[c1]};

      // Opposite test as specified in the paper.
      // It drops most of the points before full detection to reduce computation.
      if (oppositesTest(center, outers, threshold))
      {
        int* stats = getKpStats(center, outers, threshold);
        // stats[0] > 0 means, that the kp has a kp type i.e. it is a kp
        if (stats[0] > 0 && connectivityTest(center, outers, inners, threshold))
        {
          int y = int(floor((i + 2) * scale));
          int x = int(floor((j + 2) * scale));
          long row = y * DIM_SIZE * 6;
          long col = x * 6;
          icollector[row + col] = y; // y
          icollector[row + col + 1] = x; // x
          icollector[row + col + 2] = stats[0]; // kp type (end or bend)
          icollector[row + col + 3] = stats[1]; // lightess (positive or negative)
          icollector[row + col + 4] = stats[2]; // max contrast for nms
          icollector[row + col + 5] = stats[3]; // difference used in thresholding
        }
      }
    }
  }
}

void multiScaleKps(uint8_t* image, int* icollector, int size, int scales, int threshold)
{
#ifdef PRINT_TIMES
  auto start = high_resolution_clock::now();
#endif

  int previousSize = size;

  int remainder = size % 4;
  int slice = floor(size / 4);

  // Last slice is smaller because kps search ranges from rows 0 to (rows - 4)
  int lastRow = (slice * 4) + remainder - 4;
  future<void> fut1 = async(parseKps, image, icollector, size, 1, threshold, 0, slice * 1);
  future<void> fut2 = async(parseKps, image, icollector, size, 1, threshold, slice * 1, slice * 2);
  future<void> fut3 = async(parseKps, image, icollector, size, 1, threshold, slice * 2, slice * 3);
  future<void> fut4 = async(parseKps, image, icollector, size, 1, threshold, slice * 3, lastRow);

  fut1.get();
  fut2.get();
  fut3.get();
  fut4.get();

  for (int i = 1; i <= scales; i++)
  {
    float divider = pow(SCALING_FACTOR, float(i));
    int newSize = min(previousSize, int(round(size/divider)));

    uint8_t* resized = resize(image, newSize, size, divider);

    // Analyze the image in four threaded pieces
    int remainder = newSize % 4;
    int slice = floor(newSize / 4);

    // Last slice is smaller because kps search ranges from rows 0 to (rows - 4)
    int lastRow = (slice * 4) + remainder - 4;
    future<void> fut1 = async(parseKps, resized, icollector, newSize, divider, threshold, 0, slice * 1);
    future<void> fut2 = async(parseKps, resized, icollector, newSize, divider, threshold, slice * 1, slice * 2);
    future<void> fut3 = async(parseKps, resized, icollector, newSize, divider, threshold, slice * 2, slice * 3);
    future<void> fut4 = async(parseKps, resized, icollector, newSize, divider, threshold, slice * 3, lastRow);

    fut1.get();
    fut2.get();
    fut3.get();
    fut4.get();

    previousSize = newSize;
  }

#ifdef PRINT_TIMES
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Kps finding: " << duration.count() << endl;
#endif
}
