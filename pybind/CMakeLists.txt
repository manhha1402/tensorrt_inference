

message(STATUS "Build Python binding enable!")
set(PYSRCS ndarray_converter.cpp
           py_tensorrt_inference.cpp
           py_detection.cpp
           py_model.cpp
           )

pybind11_add_module(tensorrt_inference_py ${PYSRCS})

target_link_libraries(tensorrt_inference_py PRIVATE ${PROJECT_NAME}_lib)
target_include_directories(tensorrt_inference_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


set_target_properties(tensorrt_inference_py
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/pybind/tensorrt_inference"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pybind/tensorrt_inference"
  )
set_target_properties(tensorrt_inference_py PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(tensorrt_inference_py PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

file(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pybind/tensorrt_inference/__init__.py"
   "from .tensorrt_inference_py import *\n"
   )

install(TARGETS tensorrt_inference_py
  EXPORT tensorrt_inference_py_target 
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/tensorrt_inference/lib/pybind/tensorrt_inference")

install(FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pybind/tensorrt_inference/__init__.py"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/tensorrt_inference/lib/pybind/tensorrt_inference")
#sudo ln -s /usr/lib/python3.10/site-packages/numpy/core/include/numpy /usr/include/numpy