cmake_minimum_required(VERSION 3.22)
message(STATUS "Configuring with CMake ${CMAKE_VERSION}")
project(tensorrt_inference VERSION 0.0.1
               DESCRIPTION "TensorRT Inference"
               LANGUAGES C CXX CUDA)
# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")


# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
option(BUILD_PYTHON_BINDING "Build Python binding" ON)


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
find_package(OpenCV REQUIRED NO_MODULE PATHS /usr/local NO_DEFAULT_PATH)
find_package(YAML-CPP REQUIRED)
find_package(fmt REQUIRED)

include_directories(include)
# Build the TensorRT inference engine library
# Build the YoloV9 library
add_library(${PROJECT_NAME}_lib SHARED 
                src/tensorrt_api/engine.cpp
                src/model.cpp
                src/utils.cpp
                
                src/detection.cpp
                src/yolov8.cpp
                src/yolov9.cpp
                src/retinaface.cpp
                # src/face_recognition.cpp
                src/paddle_ocr/clipper.cpp
                src/paddle_ocr/paddleocr_utils.cpp
                src/paddle_ocr/paddleocr.cpp
)

target_link_libraries(${PROJECT_NAME}_lib PUBLIC 
                        ${OpenCV_LIBRARIES}
                        ${YAML_CPP_LIBRARIES}
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

# if(BUILD_PYTHON_BINDING)
#   set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#   file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
#   file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#   find_package(pybind11 REQUIRED)
#   add_subdirectory(pybind)
# endif()


# Build and link the executables
add_executable(detection_test nodes/detection_test.cpp)
target_link_libraries(detection_test ${PROJECT_NAME}_lib)

add_executable(paddleocr_test nodes/paddleocr_test.cpp)
target_link_libraries(paddleocr_test ${PROJECT_NAME}_lib)


# add_executable(face_rec_test nodes/face_rec_test.cpp)
# target_link_libraries(face_rec_test ${PROJECT_NAME}_lib)


add_executable(detectron2_test nodes/detectron2_test.cpp)
target_link_libraries(detectron2_test ${PROJECT_NAME}_lib)
##########
# EXPORT #
##########
include(GNUInstallDirs)
install(
    TARGETS ${PROJECT_NAME}_lib       
    EXPORT tensorrt_inferenceTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

# install(
#   TARGETS face_rec_test detection_test paddleocr_test detectron2_test
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
# )

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install data files to the CMake share directory
install(
    DIRECTORY config/                      # Source folder
    DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}/config  
)


include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/tensorrt_inferenceConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
    "tensorrt_inferenceConfigVersion.cmake"
    VERSION "${tensorrt_inference_VERSION_MAJOR}.${tensorrt_inference_VERSION_MINOR}.${tensorrt_inference_VERSION_PATCH}"
    COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_inferenceConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_inferenceConfigVersion.cmake
    DESTINATION share/tensorrt_inference/cmake)

install(
    EXPORT tensorrt_inferenceTargets
    NAMESPACE tensorrt_inference::
    DESTINATION share/tensorrt_inference/cmake
)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_uninstall_target.cmake")
include(cmake/package_debian.cmake)