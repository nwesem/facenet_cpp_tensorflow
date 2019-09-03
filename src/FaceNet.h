#ifndef FACE_RECOGNITION_FACENET_H
#define FACE_RECOGNITION_FACENET_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <dlib/image_processing.h>
#include <dlib/matrix.h>
#include <dlib/opencv.h>
#include "FaceExtractor.h"

using namespace tensorflow;

struct Paths {
    std::string absPath;
    std::string fileName;
};

struct KnownID {
    std::string className;
    int classNumber;
    cv::Mat embeddedFace;
};

class FaceNetClassifier : public FaceExtractor {
private:
    Session* session;
    GraphDef graphDef;
    std::vector<struct KnownID> knownFaces;
    Tensor inputTensor, phaseTensor;
    std::vector<cv::Mat> outputs;
    float knownPersonThresh;
public:
    FaceNetClassifier(std::string modelPath, float knownPersonThreshold);
    void checkStatus(Status status);
    void getFilePaths(std::string imagesPath, std::vector<struct Paths>& paths);
    void loadInputImage(std::string inputFilePath, cv::Mat& image);
    void preprocessInput(std::vector<cv::Mat>& croppedFaces);
    void createInputTensor(std::vector<cv::Mat> croppedFaces);
    void createPhaseTensor();
    void inference(int nmbrFaces);
    void computeEuclidDistanceAndClassify();
    void clearVariables();
    void forward(cv::Mat currentImg);
    void forwardPreprocessing(std::string imagesPath);
    void deleteSession();
};


#endif //FACE_RECOGNITION_FACENET_H
