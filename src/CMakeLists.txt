file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/cpp/*.cpp")
file(GLOB CU_SOURCES "${PROJECT_SOURCE_DIR}/src/cpp/*.cu")


add_library(trt_plugin SHARED ${SOURCES} ${CU_SOURCES} ${DCN_SOURCES} ${DCN_CU_SOURCES})
target_link_libraries(trt_plugin stdc++fs)
target_link_libraries(trt_plugin nvinfer)
target_link_libraries(trt_plugin nvonnxparser)
target_link_libraries(trt_plugin cudart)
target_link_libraries(trt_plugin ${OpenCV_LIBS})
target_link_libraries(trt_plugin nvinfer_plugin)
