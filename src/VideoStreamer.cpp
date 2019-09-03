#include "VideoStreamer.h"

VideoStreamer::VideoStreamer(int nmbrDevice, int videoWidth, int videoHeight) {
    m_capture = new VideoCapture(nmbrDevice);
    if (!m_capture->isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << std::endl;
    }
    m_videoWidth = videoWidth;
    m_videoHeight = videoHeight;
    m_capture->set(CAP_PROP_FRAME_WIDTH, m_videoWidth);
    m_capture->set(CAP_PROP_FRAME_HEIGHT, m_videoHeight);
}

VideoStreamer::VideoStreamer(string filename, int videoWith, int videoHeight) {
    m_capture = new VideoCapture(filename);
    if (!m_capture->isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << std::endl;
    }
    // ToDo set filename width+height doesn't work with m_capture.set(...)
}

void VideoStreamer::setResolutionDevice(int width, int height) {
    m_videoWidth = width;
    m_videoHeight = height;
    m_capture->set(CAP_PROP_FRAME_WIDTH, m_videoWidth);
    m_capture->set(CAP_PROP_FRAME_HEIGHT, m_videoHeight);
}

void VideoStreamer::setResoltionFile(int width, int height) {
    // ToDo set resolution for input files
}

void VideoStreamer::getFrame(Mat &frame) {
    *m_capture >> frame;
}

void VideoStreamer::assertResolution() {
    // currently wrong, since m_capture->get returns max/default width, height
    // but a function like this would be good to ensure good performance
    assert(m_videoWidth == m_capture->get(CAP_PROP_FRAME_WIDTH));
    assert(m_videoHeight == m_capture->get(CAP_PROP_FRAME_HEIGHT));
}