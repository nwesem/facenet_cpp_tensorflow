#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "VideoStreamer.h"
#include "FaceNet.h"

// uncomment to show how much time inference needed
//#define LOG_TIMES

using namespace tensorflow;


void checkStatus(Status status);


/**
 * Face recognition based on Google FaceNet using OpenCV, dlib and TensorFlow.
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                    "./facenet_recognition <Path/To/Image/Directory/Structure>\n"
                    "Directory structure should be path/to/img_directory/class_names.jpg\n" << std::endl;
        return 0;
    }

    cv::Mat frame;
    int nFrames = 0;
    time_t timeStart, timeEnd;

    std::string modelPath = "../models/20180402-114759.pb";
    std::string haarCascadePath = "../models/haarcascade_frontalface_default.xml";
    std::string imagesPath = argv[1];

    VideoStreamer videoStreamer = VideoStreamer(0, 640, 480);

    float knownPersonThreshold = 1.;
    FaceNetClassifier faceNetClassifier = FaceNetClassifier(modelPath, knownPersonThreshold);

    faceNetClassifier.forwardPreprocessing(imagesPath);

    auto start = chrono::steady_clock::now();
    time(&timeStart);
    while (true) {
        videoStreamer.getFrame(frame);
        if (frame.empty()) {
            std::cout << "Empty frame! Exiting..." << std::endl;
            break;
        }
        auto startFW = chrono::steady_clock::now();
        faceNetClassifier.forward(frame);
        auto endFW = chrono::steady_clock::now();

        cv::imshow("InputFrame", frame);
        nFrames++;
        char keyboard = cv::waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;

        #ifdef LOG_TIMES
        std::cout << "Forward took " << std::chrono::duration_cast<chrono::milliseconds>(endFW - startFW).count() <<
            "ms" << std::endl;
        #endif

    }
    time(&timeEnd);
    auto end = chrono::steady_clock::now();
    cv::destroyAllWindows();
    auto milliseconds = chrono::duration_cast<chrono::milliseconds>(end-start).count();
    double seconds = double(milliseconds)/1000.;
    double fps = nFrames/seconds;

    std::cout << "Counted " << nFrames << " frames in " << double(milliseconds)/1000. << " seconds!" <<
              " This equals " << fps << "fps." << std::endl;

    return 0;

}

// helper function for tensorflow status
void checkStatus(Status status) {
    if(!status.ok()) {
        std::cout << status.ToString() << std::endl;
        exit(EXIT_FAILURE);
    }
}