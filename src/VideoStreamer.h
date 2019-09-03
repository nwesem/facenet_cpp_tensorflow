#ifndef VIDEO_INPUT_WRAPPER_VIDEOSTREAMER_H
#define VIDEO_INPUT_WRAPPER_VIDEOSTREAMER_H

#include <iostream>
#include <assert.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

class VideoStreamer {
private:
    int m_videoWidth;
    int m_videoHeight;
    VideoCapture *m_capture;

public:
    VideoStreamer(int nmbrDevice, int videoWidth, int videoHeight);
    VideoStreamer(string filename, int videoWidth, int videoHeight);
    void setResolutionDevice(int width, int height);
    void setResoltionFile(int width, int height);
    void assertResolution();
    void getFrame(Mat &frame);
};

#endif //VIDEO_INPUT_WRAPPER_VIDEOSTREAMER_H
