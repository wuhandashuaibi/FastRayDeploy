include_directories(
    ../src/cpp
)

add_executable(infer_sample ${SOURCES} ${CU_SOURCES} ${PROJECT_SOURCE_DIR}/testcase/main.cpp)
target_link_libraries(infer_sample nvinfer)
target_link_libraries(infer_sample nvonnxparser)
target_link_libraries(infer_sample cudart)
target_link_libraries(infer_sample nvinfer_plugin)
target_link_libraries(infer_sample ${OpenCV_LIBS})
target_link_libraries(infer_sample trt_plugin)
