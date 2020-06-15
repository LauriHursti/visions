cmake_minimum_required(VERSION 3.12)

# Set the project name
project(visions)

# Set tests off by default
option(Tests "Tests" OFF)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable multithreading
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")

# Add files to be compiled
add_library(ftpy SHARED cpp/ft_py)
add_library(bbox_search cpp/bbox_search.cpp)
add_library(keypoint_nms cpp/keypoint_nms.cpp)
add_library(keypoints cpp/keypoints.cpp)
add_library(ft_api cpp/ft_api.cpp)
add_library(resize cpp/resize.cpp)
add_library(dbscan cpp/dbscan.cpp)

# Find dependency packages: Python and Numpy
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
message(STATUS "Using Python3 version ${Python3_VERSION}")
message(STATUS "Using Numpy version ${Python3_NumPy_VERSION}")

# Link libraries
target_link_libraries(ft_api PUBLIC bbox_search keypoint_nms keypoints resize dbscan)
target_link_libraries(ftpy PUBLIC ${Python3_LIBRARIES} ${Python3_Numpy_LIBRARIES} bbox_search ft_api keypoints keypoint_nms resize dbscan)

# Output python library to python folder
add_custom_command(TARGET ftpy 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ftpy> ../py)

# Do test setup if test flag is set to True. Requires OpenCV C++ libraries to be installed
if(Tests)
    add_executable(ft_test cpp/test.cpp)
    add_executable(resize_test cpp/resize_test.cpp)

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    target_link_libraries(ft_test PUBLIC ${OpenCV_LIBRARIES} bbox_search ft_api keypoints keypoint_nms resize)
    target_link_libraries(resize_test PUBLIC ${OpenCV_LIBRARIES} bbox_search ft_api keypoints keypoint_nms resize)

    # Cope example file for the executable
    configure_file(cpp/test_imgs/test1.jpg test1.jpg COPYONLY)
    configure_file(cpp/test_imgs/test2.jpg test2.jpg COPYONLY)
    configure_file(cpp/test_imgs/test3.jpg test3.jpg COPYONLY)
    configure_file(cpp/test_imgs/test4.jpg test4.jpg COPYONLY)
endif()    