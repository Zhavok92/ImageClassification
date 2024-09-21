cmake_minimum_required(VERSION 3.25)

project(ImageRecognition)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/dependencies)
include(get_cpm)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ImageRecognition)
add_subdirectory(dependencies)
add_subdirectory(src)