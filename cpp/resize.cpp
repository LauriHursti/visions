#include <chrono>
#include <future>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include "resize.h"

using std::async;
using std::cout;
using std::endl;
using std::future;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

// Simple averaging over neighbourhood area is used in resizing
void resizeThread(uint8_t* src, uint8_t* dst, int newSize, int oldSize, float scale, int startI, int endI)
{
  int flatSize = newSize * newSize;

  for (int i = startI; i < endI; i++)
  {
    int dstRow = floor(i / newSize);
    float srcY = dstRow * scale;
    int srcRow = floor(srcY);

    float srcYEnd = (dstRow + 1) * scale;
    int srcRowEnd = ceil(srcYEnd);

    int dstCol = i % newSize;
    float srcX = dstCol * scale;
    int srcCol = floor(srcX);

    float srcXEnd = (dstCol + 1) * scale;
    int srcColEnd = ceil(srcXEnd);

    float pixCount = (srcRowEnd - srcRow) * (srcColEnd - srcCol);
    float pixSum = 0;

    // Iterate through rows
    for (int j = srcRow; j < srcRowEnd; j++)
    {
      for (int k = srcCol; k < srcColEnd; k++)
      {
        pixSum += src[(j * oldSize) + k];
      }
    }
    dst[i] = round(pixSum / pixCount);
  }
}

// Threaded version of the resize function
uint8_t* resize(uint8_t* src, int newSize, int oldSize, float scale)
{        
#ifdef PRINT_TIMES
  auto start = high_resolution_clock::now();
#endif

  uint8_t* dst = (uint8_t*) malloc(sizeof (uint8_t) * newSize * newSize);
  int newFlatSize = newSize * newSize;

  // Calculate the new pixel values in four threads
  int remainder = newFlatSize % 4;
  int slice = floor(newFlatSize / 4);
  int lastRow = (slice * 4) + remainder - 4;

  future<void> fut1 = async(resizeThread, src, dst, newSize, oldSize, scale, 0, slice);
  future<void> fut2 = async(resizeThread, src, dst, newSize, oldSize, scale, slice * 1, slice * 2);
  future<void> fut3 = async(resizeThread, src, dst, newSize, oldSize, scale, slice * 2, slice * 3);
  future<void> fut4 = async(resizeThread, src, dst, newSize, oldSize, scale, slice * 3, lastRow);

  fut1.get();
  fut2.get();
  fut3.get();
  fut4.get();

#ifdef PRINT_TIMES
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Resize: " << duration.count() << endl;
#endif

  return dst;
}
