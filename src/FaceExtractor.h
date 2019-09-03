#ifndef FACE_RECOGNITION_FACEEXTRACTOR_H
#define FACE_RECOGNITION_FACEEXTRACTOR_H

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/matrix.h>
#include <dlib/opencv.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/gui_widgets.h>


class FaceExtractor {
protected:
    int m_faceWidth;
    int m_faceHeight;
    dlib::frontal_face_detector m_ffdetector = dlib::get_frontal_face_detector();
    cv::Ptr<cv::cuda::CascadeClassifier> m_haarDetector;
public:
    FaceExtractor();
    FaceExtractor(int faceWidth, int faceHeight);
    FaceExtractor(int faceWidth, int faceHeight, std::string haarCascadePath);
    void getCroppedFaces(cv::Mat frame, std::vector<cv::Mat> &croppedFaces, bool verbose);
    void getCroppedFacesHaar(cv::Mat& frame, cv::cuda::GpuMat d_frame, std::vector<cv::Mat> &croppedFaces, bool verbose);
    void saveCroppedFaces(std::string pathToFile);
    static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
    void setCropWidthHeight(int faceWidth, int faceHeight);
};


#endif //FACE_RECOGNITION_FACEEXTRACTOR_H
