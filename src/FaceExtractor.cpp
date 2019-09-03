#include "FaceExtractor.h"

FaceExtractor::FaceExtractor() {

    this->m_faceWidth = 160;
    this->m_faceHeight = 160;
    this->m_haarDetector = cv::cuda::CascadeClassifier::create("../models/haarcascade_frontalface_default.xml");
}


FaceExtractor::FaceExtractor(int faceWidth, int faceHeight) {

    this->m_faceWidth = faceWidth;
    this->m_faceHeight = faceHeight;
}

FaceExtractor::FaceExtractor(int faceWidth, int faceHeight, std::string haarCascadePath) {
    this->m_faceWidth = faceWidth;
    this->m_faceHeight = faceHeight;
    this->m_haarDetector = cv::cuda::CascadeClassifier::create(haarCascadePath);
}

void FaceExtractor::getCroppedFaces(cv::Mat frame, std::vector<cv::Mat> &croppedFaces, bool verbose) {

    int frameHeight = frame.rows;
    int frameWidth = frame.cols;
    int faceWidth, faceHeight;
    int padLeft = 0, padTop = 0, padRight = 0, padBottom = 0;
    cv::Mat rgbFrame, paddedFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
    dlib::cv_image<dlib::rgb_pixel> inputImg(rgbFrame);
    std::vector<dlib::rectangle> faceRects = m_ffdetector(inputImg);

    if (!faceRects.empty()){
        for (auto itFace=faceRects.begin(); itFace!=faceRects.end(); itFace++) {
            // check if face was detected close to edges, if yes pad frame to include the ROI
            // and then update the position of the ROI --> face
            if (itFace->left() < 0) {
                padLeft = std::abs(itFace->left());
                itFace->left() = 0;
            }
            if (itFace->top() < 0) {
                padTop = std::abs(itFace->top());
                itFace->top() = 0;
            }
            if (itFace->right() > frameWidth) {
                padRight = (itFace->right() - frameWidth);
                itFace->right() -= 1;
            }
            if (itFace->bottom() > frameHeight) {
                padBottom = (itFace->bottom() - frameHeight);
                itFace->bottom() -= 1;
            }
            cv::Scalar blackPixel(0,0,0);
            cv::copyMakeBorder(frame, paddedFrame, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, blackPixel);

            // convert to OpenCV rectangles, crop, resize and append to croppedFaces vector for main
            cv::Rect faceRectCV = this->dlibRectangleToOpenCV(*itFace);
            cv::Mat tempCrop = paddedFrame(faceRectCV);
            cv::resize(tempCrop, tempCrop, cv::Size(m_faceWidth, m_faceHeight), 0, 0, cv::INTER_CUBIC);
            croppedFaces.push_back(tempCrop);
        }
    }

    //show gui for debugging
    if (verbose){
//        cv::imshow("frame in function", frame);
        std::cout << "Currently " << croppedFaces.size() << " face detected!" << std::endl;
        if (croppedFaces.size() == 1) cv::imshow("face1", croppedFaces[0]);
        if (croppedFaces.size() == 2) {
            cv::imshow("face1", croppedFaces[0]);
            cv::imshow("face2", croppedFaces[1]);
        }
    }
    faceRects.clear();
}


void FaceExtractor::getCroppedFacesHaar(cv::Mat& frame, cv::cuda::GpuMat d_frame, std::vector<cv::Mat> &croppedFaces, bool verbose) {
    cv::cuda::GpuMat objBuf, d_grayFrame;
    std::vector<cv::Rect> faceRects;
    cv::cuda::cvtColor(d_frame, d_grayFrame, CV_BGR2GRAY);
    this->m_haarDetector->detectMultiScale(d_grayFrame, objBuf);
    this->m_haarDetector->convert(objBuf, faceRects);
    for(auto itFace = faceRects.begin(); itFace != faceRects.end(); ++itFace) {
        cv::Mat tempCrop = frame(*itFace);
        cv::Mat finalCrop;
        cv::resize(tempCrop, finalCrop, cv::Size(m_faceWidth, m_faceHeight), 0, 0, cv::INTER_CUBIC);
        croppedFaces.push_back(finalCrop);
    }
    //show gui for debugging
    if (verbose){
        std::cout << "Currently " << croppedFaces.size() << " face detected!" << std::endl;
        if (croppedFaces.size() == 1) cv::imshow("face1", croppedFaces[0]);
        if (croppedFaces.size() == 2) {
            cv::imshow("face1", croppedFaces[0]);
            cv::imshow("face2", croppedFaces[1]);
        }
    }
    faceRects.clear();

}


cv::Rect FaceExtractor::dlibRectangleToOpenCV(dlib::rectangle r)
{
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

void FaceExtractor::saveCroppedFaces(std::string pathToFile) {
    std::vector<cv::Mat> croppedFaces;
    cv::Mat inputImg;
    inputImg = cv::imread(pathToFile);
    this->getCroppedFaces(inputImg, croppedFaces, false);
    cv::imwrite(pathToFile, croppedFaces[0]);
}

void FaceExtractor::setCropWidthHeight(int faceWidth, int faceHeight) {

    m_faceWidth = faceWidth;
    m_faceHeight = faceHeight;
}