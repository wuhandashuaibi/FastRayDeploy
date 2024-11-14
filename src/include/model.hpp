#ifndef __MODEL_HPP__
#define __MODEL_HPP__

// TensorRT related
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_fp16.h>
#include <string>
#include <map>
#include <memory>
#include "opencv2/opencv.hpp"

class Model{

public:
    enum precision {
        FP32,
        FP16,
        INT8
    };

public:
    Model(std::string onnxPath, precision prec);
    bool build();
    bool infer();

private:
    void init_data(nvinfer1::Dims input_dims, nvinfer1::Dims output_dims);
    void init_data(nvinfer1::Dims input_dims, nvinfer1::Dims output_dims, const std::vector<std::string>& imagePaths);
    void init_data(nvinfer1::Dims input_dims, float* camera_parm, float* device_camera_parm);
    void init_input_fish_img_data(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_input_pv_img_data(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_input_front_img_data(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_input_fish_img_data_half(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_input_pv_img_data_half(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_input_front_img_data_half(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_output_data(nvinfer1::Dims output_dims);
    void init_output_data_half(nvinfer1::Dims output_dims);
    bool build_from_onnx();
    bool build_from_weights();
    
    void init_data_cpu_fish(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_data_cpu_pv(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);
    void init_data_cpu_front(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths);

    void init_data_cpu_fish(nvinfer1::Dims input_dims);
    void init_data_cpu_pv(nvinfer1::Dims input_dims);
    void init_data_cpu_front(nvinfer1::Dims input_dims);

    bool constructNetwork();
    bool preprocess();
    void print_network(nvinfer1::INetworkDefinition &network, bool optimized);
    std::map<std::string, nvinfer1::Weights> loadWeights();
    
private:
    std::string mWtsPath = "";
    std::string mOnnxPath = "";
    std::string mEnginePath = "";
    std::map<std::string, nvinfer1::Weights> mWts;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    float* mInputHost;
    float* mInputDevice;
    // image data
    float* mInputDevice_fish_imgs;
    float* mInputDevice_pv_imgs;
    float* mInputDevice_front_imgs;

    half* mInputDevice_fish_imgs_half;
    half* mInputDevice_pv_imgs_half;
    half* mInputDevice_front_imgs_half;
    // host camera parm data
    float* host_fish_extri_parm;
    float* host_fish_intri_parm;
    float* host_pv_extri_parm;
    float* host_pv_intri_parm;
    float* host_front_extri_parm;
    float* host_front_intri_parm;

    float* device_fish_extri_parm;
    float* device_fish_intri_parm;
    float* device_pv_extri_parm;
    float* device_pv_intri_parm;
    float* device_front_extri_parm;
    float* device_front_intri_parm;


    half* mOutputHost_half;
    half* mOutputDevice_half;
    float* mOutputHost;
    float* mOutputDevice;
    int mInputSize;
    int mInputSize_fish_img;
    int mInputSize_pv_img;
    int mInputSize_front_img;

    int mInputSize_fish_intri;
    int mInputSize_fish_extri;
    int mInputSize_pv_intri;
    int mInputSize_pv_extri;
    int mInputSize_front_intri;
    int mInputSize_front_extri;
    int mOutputSize;
    nvinfer1::DataType mPrecision;
    
};

#endif // __MODEL_HPP__
