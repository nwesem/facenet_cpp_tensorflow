# facenet_cpp_tensorflow
Fully working live face recognition using retrained 
[Google FaceNet](https://arxiv.org/abs/1503.03832) architecture.
Implementation based on
[David Sandberg's python implementation](https://github.com/davidsandberg/facenet)
and
[mndar's cpp implementation](https://github.com/mndar/facenet_classifier).
This neural network architecture was originally trained with a triplet
loss function. Reimplementations have had trouble reproducing the
original results of the paper with the triplet loss function.
Reimplementations use Softmax loss instead with good results. David
Sandberg states >99.6% accuracy with a model trained on VGGFace2. Check
[Github Wiki](https://github.com/davidsandberg/facenet/wiki) for more
info.

## Dependencies
cuda 10.0 + cudnn 7.5 <br>
bazel 0.18 <br>
protobuf 3.6.0 <br>
eigen 3.3.5 <br>
tensorflow r1.10+ <br>
opencv 3.x or opencv 4.x

## Install dependencies
You can use 
[this Medium article](https://medium.com/@fanzongshaoxing/use-tensorflow-c-api-with-opencv3-bacb83ca5683)
as a rough guideline for the tensorflow installation. See dependencies
for more information on the versions used for this project.

* Install bazel 0.18, protobuf 3.6.0, eigen 3.3.5 <br>
* tensorflow (1.10)
```bash
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
sudo pip3 install six numpy wheel
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.10      # or other release

# During configure: No to everything except cuda and jemalloc (for now)
./configure


# build tensorflow c++ with bazel (you can change the number of cores used with jobs flag)
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both \
--copt=-msse4.2 --config=monolithic --config=cuda --jobs 4 //tensorflow:libtensorflow_cc.so
# --config=monolithic is important, so tf works with opencv (for cv::imread)

# copy required files into a single path for c++ linkage
sudo mkdir /usr/local/include/tf  # make a directory ubder /usr/local/include path
sudo mkdir /usr/local/include/tf/tensorflow
sudo cp -r bazel-genfiles/ /usr/local/include/tf
sudo cp -r tensorflow/cc /usr/local/include/tf/tensorflow
sudo cp -r tensorflow/core /usr/local/include/tf/tensorflow
sudo cp -r third_party /usr/local/include/tf
sudo cp bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib
sudo cp bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib
# OPTIONAL: also copy eigen to /usr/local/include/tf
sudo cp -r /usr/local/include/eigen3 /usr/local/include/tf/third_party
```

## Download models
* **Download frozen tensorflow graph** from
  [github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet/#pre-trained-models).
  The links are in the pre-trained models section. This repo was tested
  with the VGGFace2 model. **After downloading move the .pb file to the
  models folder.**


* Download HaarCascadeClassifier to the empty models folder
```bash
cd /path/to/facenet_cpp_tensorflow/models
wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

## dlib 

* If dlib is **not** compiled on your machine, download dlib and then go
  ahead and compile this project.
```bash
# go to folder one above this project and download dlib
cd /path/to/facenet_cpp_tensorflow/..
wget http://dlib.net/files/dlib-19.17.tar.bz2
tar xfvj dlib-19.17.tar.bz2
rm dlib-19.17.tar.bz2
```
* If precompiled dlib library is used, please change the
  [CMakeLists.txt](CMakeLists.txt). Replace
  **add_subdirectory(path/to/dlib(-19.17)/ dlib_build)** with
  **find_package(dlib)**, then go ahead and compile this project

 ## Installation
```bash
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make -j${nproc}
```
Make sure to build this project as Release. If dlib is built in Debug,
it is extremely slow at detecting faces ~0.01fps.


## Usage 
Run this project with the path to your image directory of known people
with class names as file names, e.g., class_name.jpg will be classified
as class_name. Make sure you only have images in the imgs folder, no
other files. README.md will be skipped.

```bash
./face_recognition ../imgs/
```

## Documentation
Open Doxygen documentation (located in docs/html/index.html) with your 
local browser for more info about the project.

## Stats
Running with **~18fps** on Intel i7 7700HQ processor and NVIDIA GeForce
GTX 1050 using OpenCV's HaarCascadeClassifier for face detection, with a
few changes you can use dlibs face detection which is more accurate, but
slower.

## License
Please respect all licenses of OpenCV, dlib, and the data the tensorflow
model was trained on.

## Info
Niclas Wesemann <br>
[niclas.wesemann@gmail.com](mailto:niclas.wesemann@gmail.com) <br>
