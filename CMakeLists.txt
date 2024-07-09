cmake_minimum_required(VERSION 3.22)
project(tensorrt_inference)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# CMake Options
option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
if (ENABLE_BENCHMARKS)
    add_compile_definitions(ENABLE_BENCHMARKS)
endif()
# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



# We require CUDA, OpenCV, and TensorRT
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

include_directories(include)
# Build the TensorRT inference engine library
# Build the YoloV9 library
add_library(${PROJECT_NAME}_lib SHARED 
                src/tensorrt_api/engine.cpp
                src/yolov9.cpp
)

target_link_libraries(${PROJECT_NAME}_lib PUBLIC 
                        ${OpenCV_LIBRARIES}
                        ${CUDA_LIBRARIES} 
                        ${CMAKE_THREAD_LIBS_INIT} 
                        ${TensorRT_LIBRARIES}
                        fmt::fmt
                    )
target_include_directories(
  ${PROJECT_NAME}_lib 
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${CUDA_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}  # for private headers generated during build
)




# Build and link the executables
add_executable(detect_object_image nodes/object_detection_image.cpp)
target_link_libraries(detect_object_image ${PROJECT_NAME}_lib)


add_executable(detect_object_video nodes/object_detection_video_stream.cpp)
target_link_libraries(detect_object_video ${PROJECT_NAME}_lib)