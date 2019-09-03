cmake_minimum_required(VERSION 3.0)
set(PROJECT_NAME facenet_cpp_tensorflow)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

# OpenCV
find_package(OpenCV REQUIRED)

# dlib
add_subdirectory(../dlib-19.17 dlib_build)
#find_package(dlib REQUIRED)

# eigen
include_directories("/usr/local/include/eigen3")

# tensorflow include dirs
include_directories("/usr/local/include/tf/")
include_directories("/usr/local/include/tf/bazel-genfiles/")
include_directories("/usr/local/include/tf/tensorflow/")
include_directories("/usr/local/include/tf/third-party/")

# execs
AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})

# libs
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} dlib::dlib)
#target_link_libraries(face_recognition ${DLIB_LIBRARIES} dlib)
target_link_libraries(${PROJECT_NAME} "/usr/local/lib/libtensorflow_cc.so")
target_link_libraries(${PROJECT_NAME} "/usr/local/lib/libtensorflow_framework.so")
