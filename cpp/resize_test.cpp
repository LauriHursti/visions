#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <stdint.h>
#include <iostream>

#include "constants.h"
#include "resize.h"

using std::cout;
using std::endl;
using std::string;
using std::to_string;

cv::Mat getGrayImage(string fname)
{
  return cv::imread(fname, cv::IMREAD_GRAYSCALE);   // Read the file
}

uint8_t* imgToArray(cv::Mat image)
{
  uint8_t* arr = image.isContinuous() ? image.ptr<uint8_t>(0): image.clone().ptr<uint8_t>(0);
  return arr;
}

void test()
{
    string name = "test1.jpg";
    cv::Mat grayImg = getGrayImage(name);
    uint8_t* uimage = imgToArray(grayImg);

    for (int i = 1; i < 4; i++)
    {
        float divider = pow(SCALING_FACTOR, float(i));
        int newSize = round(DIM_SIZE/divider);
        uint8_t* resized = resize(uimage, newSize, DIM_SIZE, divider);       
        cv::Mat cvimg(newSize, newSize, CV_8UC1, resized);
        cv::imwrite(name + "_" + to_string(i) + "_scale_out.jpg", cvimg);
    }
}

int main()
{
  test();
  return 0;
}
