#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>
#include <string>

#include "constants.h"
#include "bbox_search.h"
#include "ft_api.h"

using std::array;
using std::cout;
using std::endl;
using std::string;
using std::vector;

cv::Mat getGrayImage(string fname)
{
  return cv::imread(fname, cv::IMREAD_GRAYSCALE);   // Read the file
}

cv::Mat getColorImage(string fname)
{
  return cv::imread(fname, cv::IMREAD_UNCHANGED);
}

uint8_t* imgToArray(cv::Mat image)
{
  uint8_t* arr = image.isContinuous() ? image.data: image.clone().data;
  return arr;
}

void test()
{
  array<string, 4> testImgs = {"test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"};

  for (int i = 0; i < testImgs.size(); i++)
  {
    string name = testImgs[i];
    cv::Mat grayImg = getGrayImage(name);
    cv::Mat colorImg = getColorImage(name);
    uint8_t* uimage = imgToArray(grayImg);
    vector<array<int, 6>> kpss = getFASTextKeypoints(uimage, 2000, 3, 32, false, true);
    vector<bbox> boxes = getCornersThreaded(uimage, kpss, 42, 42);

    cout << "Handled img" << endl;

    for (int i = 0; i < kpss.size(); i++)
    {
      int y = kpss[i][0];
      int x = kpss[i][1];
      cv::Scalar color = cv::Scalar(0, 10, 255);
      cv::circle(colorImg, cv::Point(x, y), 1, color, 1);
    }

    for (int i = 0; i < boxes.size(); i++)
    {
      bbox box = boxes[i];
      cv::Rect rect = {box.minx, box.miny, box.maxx-box.minx, box.maxy-box.miny};
      cv::rectangle(colorImg, rect, cv::Scalar(255, 10, 0));
    }

    cv::imwrite(name + "out.jpg", colorImg);
    cout << endl;
  }
}

int main()
{
  test();
  return 0;
}
