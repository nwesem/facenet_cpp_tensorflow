#include "FaceNet.h"
/**
 * Constructor for FaceNetClassifier objects. It initializes the tensorflow session and creates the computation graph.
 * @param modelPath local path to the TensorFlow model as protobuf (.pb) file
 * @param knownPersonThreshold Threshold for the euclidean distance between face encodings to decide whether detected
 * face is known or not
 */
FaceNetClassifier::FaceNetClassifier(std::string modelPath, float knownPersonThreshold) {
    this->knownPersonThresh = knownPersonThreshold;
    Status status = NewSession(SessionOptions(), &this->session);
    this->checkStatus(status);
    status = ReadBinaryProto(Env::Default(), modelPath, &this->graphDef);
    this->checkStatus(status);
    status = this->session->Create(this->graphDef);
    this->checkStatus(status);
}


/**
 * Checks status for initializations of TensorFlow session and exits if it failed.
 * @param status a TensorFlow Status object that holds the current status message.
 */
void FaceNetClassifier::checkStatus(Status status) {
    if(!status.ok()) {
        std::cout << status.ToString() << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Checks image directory and returns all paths and filesnames. The directory structure should be imgs/class_name.jpg.
 * @param imagesPath absolute or relative path to the image directory.
 * @param paths Struct to temporary save the absolute path and filename per class.
 */
void FaceNetClassifier::getFilePaths(std::string imagesPath, std::vector<struct Paths>& paths) {
    std::cout << "Parsing Directory: " << imagesPath << std::endl;
    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir (imagesPath.c_str())) != NULL) {
        while ((entry = readdir (dir)) != NULL) {
            std::string readmeCheck(entry->d_name);
            if (entry->d_type != DT_DIR && readmeCheck != "README.md") {
                struct Paths tempPaths;
                tempPaths.fileName = string(entry->d_name);
                tempPaths.absPath = imagesPath + "/" + tempPaths.fileName;
                paths.push_back(tempPaths);
            }
        }
        closedir (dir);
    }
}


/**
 * Loads input image into a cv::Mat object.
 * @param inputFilePath absolute or relative path to file
 * @param image cv::Mat object that will hold the input image
 */
void FaceNetClassifier::loadInputImage(std::string inputFilePath, cv::Mat& image) {
    image = cv::imread(inputFilePath.c_str());
}


void FaceNetClassifier::preprocessInput(std::vector<cv::Mat>& croppedFaces) {
    for (int i = 0; i < croppedFaces.size(); i++) {
        //mean and std
        cvtColor(croppedFaces[i], croppedFaces[i], CV_RGB2BGR);
        cv::Mat temp = croppedFaces[i].reshape(1, croppedFaces[i].rows * 3);
        cv::Mat     mean3;
        cv::Mat     stddev3;
        cv::meanStdDev(temp, mean3, stddev3);

        double mean_pxl = mean3.at<double>(0);
        double stddev_pxl = stddev3.at<double>(0);
        cv::Mat image2;
        croppedFaces[i].convertTo(image2, CV_64FC1);
        croppedFaces[i] = image2;
        croppedFaces[i] = croppedFaces[i] - cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
        croppedFaces[i] = croppedFaces[i] / stddev_pxl;
    }
}

/**
 * Creates the input tensor for the feed dict for the network.
 * @param croppedFaces currently detected faces cropped from image or frame: Size: [nmbrFaces x 160 x 160 x 3]
 */
void FaceNetClassifier::createInputTensor(std::vector<cv::Mat> croppedFaces) {
    int nmbrFaces = croppedFaces.size();
    Tensor tempTensor(DT_FLOAT, TensorShape({nmbrFaces, 160, 160, 3}));
    // get pointer to memory for that Tensor
    float *p = tempTensor.flat<float>().data();
    int i;

    for (i = 0; i < nmbrFaces ; i++) {
        // create a "fake" cv::Mat from it

        cv::Mat camera_image(160, 160, CV_32FC3, p + i*160*160*3);
        croppedFaces[i].convertTo(camera_image, CV_32FC3);
    }
//    std::cout << tempTensor.DebugString() << std::endl;
    this->inputTensor = Tensor(tempTensor);
    for (i = 0; i < croppedFaces.size(); i++) {
        croppedFaces[i].release();
    }
}

/**
 * Creates the Phase tensor for the feed dict for the network.
 */
void FaceNetClassifier::createPhaseTensor() {
    Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_tensor.scalar<bool>()() = false;
    this->phaseTensor = Tensor (phase_tensor);
}

/**
 * Computes the output for the currently presented feed dict consisting of input tensor and phase tensor.
 * @param nmbrFaces the amount of currently detected faces
 */
void FaceNetClassifier::inference(int nmbrFaces) {
    std::string input_layer = "input:0";
    std::string phase_train_layer = "phase_train:0";
    std::string output_layer = "embeddings:0";
    std::vector<tensorflow::Tensor> outputTensor;
    std::vector<std::pair<string, tensorflow::Tensor>> feed_dict = {
            {input_layer, this->inputTensor},
            {phase_train_layer, this->phaseTensor},
    };

    // cout << "Input Tensor: " << inputTensor.DebugString() << endl;
    Status run_status = this->session->Run(feed_dict, {output_layer}, {} , &outputTensor);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status << "\n";
        return;
    }
    // cout << "Output: " << outputs[0].DebugString() << endl;

    float *p = outputTensor[0].flat<float>().data();
    cv::Mat output_mat;
    for (int i = 0; i < nmbrFaces; i++) {
        cv::Mat matRow(cv::Size(512, 1), CV_32F, p + i * 512);
        this->outputs.push_back(matRow);
        matRow.release();
    }
}

/**
 * Computes the Euclidean distance between the currently detected face encodings and all known encodings and classifies
 * using the distance. If there is more than one face encoding below threshold, it will use the one with the smallest
 * euclidean distance.
 */
void FaceNetClassifier::computeEuclidDistanceAndClassify() {

    for(int i = 0; i < this->outputs.size(); i++) {
        double minDistance = 100.;
        double currDistance = 0.;
        int winner;
        for (int j = 0; j < this->knownFaces.size(); j++) {
            cv::Mat tempOutput;
            this->outputs[i].copyTo(tempOutput);
            currDistance = cv::norm(tempOutput, this->knownFaces[j].embeddedFace, cv::NORM_L2);
            // for Debug distances between faces
            // std::cout << "Distance to " << knownFaces[j].className << " is " << currDistance << std::endl;
            if (currDistance < minDistance) {
                minDistance = currDistance;
                winner = j;
            }
        }
        if (minDistance < this->knownPersonThresh) {
            std::cout << this->knownFaces[winner].className << std::endl;
        }
        else {
            std::cout << "New Person?" << std::endl;
        }
    }
    std::cout << "\n";
}

/**
 * Performs a full foward pass including crop faces, preprocessing (images standardization), preparation of tensors,
 * inference using the tensorflow model, computation of euclidean distance and classification.
 * @param currentImg an input image or a current frame from a camera.
 */
void FaceNetClassifier::forward(cv::Mat currentImg) {
    std::vector<cv::Mat> croppedFaces;
    cv::cuda::GpuMat d_currentImg;
    d_currentImg.upload(currentImg);
    // You can log the times of the CropFace function here
    //auto start = std::chrono::steady_clock::now();
    this->getCroppedFacesHaar(currentImg, d_currentImg, croppedFaces, false);
    //auto end = std::chrono::steady_clock::now();
    //std::cout << "CropFace took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() <<
    //          "ms" << std::endl;
    if(!croppedFaces.empty()) {
        int nmbrFaces = croppedFaces.size();
        this->preprocessInput(croppedFaces);
        this->createInputTensor(croppedFaces);
        this->createPhaseTensor();
        this->inference(nmbrFaces);
        this->computeEuclidDistanceAndClassify();
        this->clearVariables();
    }
    else {
        std::cout << "No person found!" << std::endl;
    }
    // release all data
    for (auto face : croppedFaces) face.release();
    croppedFaces.clear();
}

/**
 * Same as forward. Performs a full foward pass including crop faces, preprocessing (images standardization),
 * preparation of tensors, inference using the tensorflow model, computation of euclidean distance and classification,
 * but here the input is a directory containing images for the classes.
 * @param imagesPath path/to/image/directory - local path to images
 */
void FaceNetClassifier::forwardPreprocessing(std::string imagesPath) {
    std::vector<cv::Mat> croppedFaces;
    std::vector<struct Paths> paths;
    cv::Mat image;

    this->getFilePaths(imagesPath, paths);
    int classCounter = 0;
    for (int i = 0; i < paths.size(); i++) {
        loadInputImage(paths[i].absPath, image);
        this->getCroppedFaces(image, croppedFaces, false);
        if(!croppedFaces.empty()) {
            int nmbrFaces = croppedFaces.size(); // should be one when data is captured
            this->preprocessInput(croppedFaces);
            this->createInputTensor(croppedFaces);
            this->createPhaseTensor();
            this->inference(nmbrFaces);

            struct KnownID person;
            std::size_t index = paths[i].fileName.find_last_of(".");
            std::string rawName = paths[i].fileName.substr(0,index);
            person.className = rawName;
            person.classNumber = classCounter;
            // ToDo optimize copy
            this->outputs[0].copyTo(person.embeddedFace);
            this->knownFaces.push_back(person);
            classCounter++;
        }
        else {
            std::cout << "No face found in this path:" << paths[i].absPath << std::endl;
        }
        for (int j = 0; j < this->outputs.size(); j++) {
            this->outputs[j].release();
        }
        this->clearVariables();
        croppedFaces.clear();
    }

    // for DEBUG
    /*
    for (auto face : this->knownFaces) {
        std::cout << face.className << "--->Class " << face.classNumber << " and Size = " <<
            face.embeddedFace.size() << "\n"; // << face.embeddedFace << "\n\n";
    }
    for(auto it = knownFaces.begin(); it != knownFaces.end(); ++it) {
        std::cout << it->className << " = " << it->embeddedFace << "\n\n";
    }
    */
}

/**
 * Clears variables in the end of a forward.
 */
void FaceNetClassifier::clearVariables() {
    for(int i = 0; i < outputs.size(); i++) {
        outputs[i].release();
    }
    outputs.clear();
}

/**
 * Deletes the created TensorFlow session.
 */
void FaceNetClassifier::deleteSession() {
    // ToDo
}

