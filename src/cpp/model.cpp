#include <cstring>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <type_traits>
#include <assert.h>

#include "model.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "cuda_runtime.h"
#include "math.h"
#include "network.hpp"
#include "preprocess.hpp"
#include "timer.hpp"

float input_5x5[] = {
    0.7576, 0.2793, 0.4031, 0.7347, 0.0293,
    0.7999, 0.3971, 0.7544, 0.5695, 0.4388,
    0.6387, 0.5247, 0.6826, 0.3051, 0.4635,
    0.4550, 0.5725, 0.4980, 0.9371, 0.6556,
    0.3138, 0.1980, 0.4162, 0.2843, 0.3398};

float input_1x5[] = {
    0.7576, 0.2793, 0.4031, 0.7347, 0.0293};


using namespace std;

class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]" CLEAR;
            case Severity::kERROR:          str = RED    "[error]" CLEAR;
            case Severity::kWARNING:        str = BLUE   "[warn]"  CLEAR;
            case Severity::kINFO:           str = YELLOW "[info]"  CLEAR;
            case Severity::kVERBOSE:        str = PURPLE "[verb]"  CLEAR;
        }
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
// using make_unique_alins = std::unique_ptr<T, InferDeleter>;
using make_unique_alins = std::unique_ptr<T>;

Model::Model(string path, precision prec){
    if (getFileType(path) == ".onnx")
        mOnnxPath = path;
    else if (getFileType(path) == ".weights")
        mWtsPath = path;
    else 
        LOGE("ERROR: %s, wrong weight or model type selected. Program terminated", getFileType(path).c_str());

    if (prec == precision::FP16) {
        mPrecision = nvinfer1::DataType::kHALF;
    } else if (prec == precision::INT8) {
        mPrecision = nvinfer1::DataType::kINT8;
    } else {
        mPrecision = nvinfer1::DataType::kFLOAT;
    }

    mEnginePath = getEnginePath(path, prec);
}

// decode一个weights文件，并保存到map中
// weights的格式是:
//    count
//    [name][len][weights value in hex mode]
//    [name][len][weights value in hex mode]
//    ...
map<string, nvinfer1::Weights> Model::loadWeights(){
    ifstream f;
    if (!fileExists(mWtsPath)){ 
        LOGE("ERROR: %s not found", mWtsPath.c_str());
    }

    f.open(mWtsPath);

    int32_t size;
    map<string, nvinfer1::Weights> maps;
    f >> size;

    if (size <= 0) {
        LOGE("ERROR: no weights found in %s", mWtsPath.c_str());
    }

    while (size > 0) {
        nvinfer1::Weights weight;
        string name;
        int weight_length;

        f >> name;
        f >> std::dec >> weight_length;

        uint32_t* values = (uint32_t*)malloc(sizeof(uint32_t) * weight_length);
        for (int i = 0; i < weight_length; i ++) {
            f >> std::hex >> values[i];
        }

        weight.type = nvinfer1::DataType::kFLOAT;
        weight.count = weight_length;
        weight.values = values;

        maps[name] = weight;

        size --;
    }

    return maps;
}

bool Model::build() {
    if (mOnnxPath != "") {
        return build_from_onnx();
    } else {
        return build_from_weights();
    }
}

bool Model::build_from_weights(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }

    mWts = loadWeights();

    // 这里和之前的创建方式是一样的
    Logger logger;
    auto builder       = make_unique_alins<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto config        = make_unique_alins<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto network       = make_unique_alins<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));

    // 根据不同的网络架构创建不同的TensorRT网络，这里使用几个简单的例子
    if (mWtsPath == "models/weights/sample_cbr.weights") {
        network::build_cbr(*network, mPrecision, mWts);
    } else if (mWtsPath == "models/weights/sample_resBlock.weights") {
        network::build_resBlock(*network, mPrecision, mWts);
    } else if (mWtsPath == "models/weights/sample_convBNSiLU.weights") {
        network::build_convBNSiLU(*network, mPrecision, mWts);
    } else if (mWtsPath == "models/weights/sample_c2f.weights") {
        network::build_C2F(*network, mPrecision, mWts);
    } else {
        return false;
    }

    // 接下来的事情也是一样的
    config->setMaxWorkspaceSize(1<<28);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    builder->setMaxBatchSize(1);

    // 设置量化参数
    // 注意一点的是，kPREFER_PRECISION_CONSTRAINTS是用来保证所有的层是按照指定的精度计算
    // 如果没有的话，TensorRT会根据计算效率有可能不做转换
    // 这个是配合layer的精度指定使用的
    
    if (builder->platformHasFastFp16() && mPrecision == nvinfer1::DataType::kHALF) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    } else if (builder->platformHasFastInt8() && mPrecision == nvinfer1::DataType::kINT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    auto engine        = make_unique_alins<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique_alins<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    cout << "file size is " << plan->size() << endl;
    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();

    // 把优化前和优化后的各个层的信息打印出来
    LOG("Before TensorRT optimization");
    print_network(*network, false);
    LOG("");
    LOG("After TensorRT optimization");
    print_network(*network, true);

    // 最后把map给free掉
    for (auto& mem : mWts) {
        free((void*) (mem.second.values));
    }
    LOG("Finished building engine");
    return true;
}

bool Model::build_from_onnx(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }
    Logger logger;
    auto builder       = make_unique_alins<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique_alins<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique_alins<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique_alins<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<30);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }
    LOG("onnx path %s", mOnnxPath.c_str());
    if (builder->platformHasFastFp16() && mPrecision == nvinfer1::DataType::kHALF) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    } else if (builder->platformHasFastInt8() && mPrecision == nvinfer1::DataType::kINT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }
    network->getInput(0)->setType(nvinfer1::DataType::kFLOAT);
    network->getInput(1)->setType(nvinfer1::DataType::kFLOAT);
    network->getInput(2)->setType(nvinfer1::DataType::kFLOAT);
    network->getOutput(0)->setType(nvinfer1::DataType::kFLOAT);
    
    auto engine        = make_unique_alins<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique_alins<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);
    // LOG("debug 0");
    if (false)
    {
        mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        // mInputDims         = network->getInput(0)->getDimensions();
        // mOutputDims        = network->getOutput(0)->getDimensions();
        auto mInput0 = mEngine->getBindingDimensions(0);
        auto mInput1 = mEngine->getBindingDimensions(1);
        auto mInput2 = mEngine->getBindingDimensions(2);
        auto mOutput0 = mEngine->getBindingDimensions(9);

        LOG("Input tensor shape: ");
        LOG("input0 dim shape is:  %s", printDims(mInput0).c_str());    
        LOG("input1 dim shape is:  %s", printDims(mInput1).c_str());    
        LOG("input2 dim shape is:  %s", printDims(mInput2).c_str());    
        LOG("output0 dim shape is:  %s", printDims(mOutput0).c_str());    
        // // // // 把优化前和优化后的各个层的信息打印出来
        LOG("Before TensorRT optimization");
        print_network(*network, false);
        LOG("");
        LOG("After TensorRT optimization");
        print_network(*network, true);
        mEngine->destroy();
    }
    
    // network->destroy();
    
    // engine->destroy();
    // runtime->destroy();
    // builder->destroy();
    LOG("Finished building engine");
    return true;
};

bool Model::infer(){
    /*
        我们在infer需要做的事情
        1. 读取model => 创建runtime, engine, context
        2. 把数据进行host->device传输
        3. 使用context推理
        4. 把数据进行device->host传输
    */

    /* 1. 读取model => 创建runtime, engine, context */
    std::shared_ptr<timer::Timer> m_timer;
    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);
    
    Logger logger;
    auto runtime     = make_unique_alins<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = make_unique_alins<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = make_unique_alins<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    auto input_dims_fish_img   = context->getBindingDimensions(0);
    auto input_dims_pv_img   = context->getBindingDimensions(1);
    auto input_dims_front_img   = context->getBindingDimensions(2);
    // auto input_fish_intri_parm = context->getBindingDimensions(3);
    // auto input_fish_extri_parm = context->getBindingDimensions(4);
    // auto input_pv_intri_parm = context->getBindingDimensions(5);
    // auto input_pv_extri_parm = context->getBindingDimensions(6);
    // auto input_front_intri_parm = context->getBindingDimensions(7);
    // auto input_front_extri_parm = context->getBindingDimensions(8);
    auto output_dims  = context->getBindingDimensions(3);

    LOG("input dim fish img shape is:  %s", printDims(input_dims_fish_img).c_str());
    LOG("input dim pv img shape is:  %s", printDims(input_dims_pv_img).c_str());
    LOG("input dim front img shape is:  %s", printDims(input_dims_front_img).c_str());
    LOG("output dim shape is: %s", printDims(output_dims).c_str());

    /* 2. 创建流 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* 2. 初始化input，以及在host/device上分配空间 */
    // init_data(input_dims, output_dims);
    std::vector<std::string> fishimageFiles = 
                    {"./work_dir/vt_debug/fish_front.jpg",
                     "./work_dir/vt_debug/fish_left.jpg", 
                     "./work_dir/vt_debug/fish_back.jpg",
                     "./work_dir/vt_debug/fish_right.jpg"};
    // init_data(input_dims_fish_img, output_dims, imageFiles);
    
    std::vector<std::string> pvimageFiles = 
                    {"./work_dir/vt_debug/pv_front_left.jpg",
                     "./work_dir/vt_debug/pv_back_left.jpg", 
                     "./work_dir/vt_debug/pv_back.jpg",
                     "./work_dir/vt_debug/pv_front_right.jpg",
                     "./work_dir/vt_debug/pv_back_right.jpg"};
    
    std::vector<std::string> frontimageFiles = 
                    {"./work_dir/vt_debug/front.jpg",
                     };
    
    // gpu process chw
    // init_input_fish_img_data(input_dims_fish_img, fishimageFiles);
    // init_input_pv_img_data(input_dims_pv_img, pvimageFiles);
    // init_input_front_img_data(input_dims_front_img, frontimageFiles);

    // cpu process chw
    init_data_cpu_fish(input_dims_fish_img, fishimageFiles);
    init_data_cpu_pv(input_dims_pv_img, pvimageFiles);
    init_data_cpu_front(input_dims_front_img, frontimageFiles);

    // dummy data 64 channel
    // init_data_cpu_fish(input_dims_fish_img);
    // init_data_cpu_pv(input_dims_pv_img);
    // init_data_cpu_front(input_dims_front_img);

    // test img
    // float* front_img;
    // cudaMallocHost(&front_img, mInputSize_fish_img);
    // cudaMemcpyAsync(front_img, mInputDevice_fish_imgs, mInputSize_fish_img, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    // cudaStreamSynchronize(stream);
    // std::string bin_file = "./work_dir/vt_debug/inputs.bin";
    // auto f = fopen(bin_file.c_str(), "wb");

    // fwrite(front_img, sizeof(float), 3*80*128*4, f);
    // fclose(f);
    // float* front_img;
    // cudaMallocHost(&front_img, mInputSize_front_img);
    // cudaMemcpyAsync(front_img, mInputDevice_front_imgs, mInputSize_front_img, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    // cudaStreamSynchronize(stream);
    // std::string bin_file = "./work_dir/vt_debug/inputs.bin";
    // auto f = fopen(bin_file.c_str(), "wb");

    // fwrite(front_img, sizeof(float), 3*96*192*1, f);
    // fclose(f);
    
    // init_data(input_fish_intri_parm, host_fish_intri_parm, device_fish_intri_parm);
    // init_data(input_fish_extri_parm, host_fish_extri_parm, device_fish_extri_parm);
    // init_data(input_pv_intri_parm, host_pv_intri_parm, device_pv_intri_parm);
    // init_data(input_pv_extri_parm, host_pv_extri_parm, device_pv_extri_parm);
    // init_data(input_front_intri_parm, host_front_intri_parm, device_front_intri_parm);
    // init_data(input_front_extri_parm, host_front_extri_parm, device_front_extri_parm);

    //float
    init_output_data(output_dims);
    /* 2. host->device的数据传递*/
    // cudaMemcpyAsync(device_fish_intri_parm, host_fish_intri_parm, mInputSize_fish_intri, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(device_fish_extri_parm, host_fish_extri_parm, mInputSize_fish_extri, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(device_pv_intri_parm, host_pv_intri_parm, mInputSize_pv_intri, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(device_pv_extri_parm, host_pv_extri_parm, mInputSize_pv_extri, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(device_front_intri_parm, host_front_intri_parm, mInputSize_front_intri, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(device_front_extri_parm, host_front_extri_parm, mInputSize_front_extri, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
    /* 3. 模型推理, 最后做同步处理 */
    // float* bindings[] = {mInputDevice_fish_imgs, mInputDevice_pv_imgs, mInputDevice_front_imgs, 
    //                     device_fish_intri_parm, device_fish_extri_parm,device_pv_intri_parm, 
    //                     device_pv_extri_parm,device_front_intri_parm,device_front_extri_parm,
    //                     mOutputDevice};
    
    // float 
    float* bindings[] = {mInputDevice_fish_imgs, mInputDevice_pv_imgs, mInputDevice_front_imgs, 
                        mOutputDevice};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start, 0);
    bool success = context->enqueueV2((void**)bindings, stream, nullptr);
    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Total time: %f ms\n", milliseconds);
    /* 4. device->host的数据传递 */
    cudaMemcpyAsync(mOutputHost, mOutputDevice, mOutputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // LOG("input data is:  %s", printTensor(mInputHost, mInputSize / sizeof(float), input_dims).c_str());
    // LOG("output data is: %s", printTensor(mOutputHost, mOutputSize / sizeof(float), output_dims).c_str());
    // for (size_t i = 0; i < 18*240*120; i++)
    // {
    //     if (mOutputHost[i]>1)
    //     {
    //         printf("%f ", mOutputHost[i]);
    //     }
        
    // }

    // float
    std::string out_bin_file = "./work_dir/vt_debug/test.bin";
    auto d = fopen(out_bin_file.c_str(), "wb");

    fwrite(mOutputHost, sizeof(float), int32_t(mOutputSize/sizeof(float)), d);
    fclose(d);
    mEngine->destroy();

    LOG("finished inference");
    return true;
}

void Model::print_network(nvinfer1::INetworkDefinition &network, bool optimized) {

    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    string layer_info;

    for (int i = 0; i < inputCount; i++) {
        auto input = network.getInput(i);
        LOG("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        auto output = network.getOutput(i);
        LOG("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }
    
    int layerCount = optimized ? mEngine->getNbLayers() : network.getNbLayers();
    LOG("network has %d layers", layerCount);

    if (!optimized) {
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            auto layer   = network.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOG("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        auto inspector = make_unique_alins<nvinfer1::IEngineInspector>(mEngine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            string info = inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON);
            info = info.substr(0, info.size() - 1);
            LOG("layer_info: %s", info.c_str());
        }
    }
}


void Model::init_data(nvinfer1::Dims input_dims, nvinfer1::Dims output_dims){
    mInputSize  = getDimSize(input_dims) * sizeof(float);
    mOutputSize = getDimSize(output_dims) * sizeof(float);

    cudaMallocHost(&mInputHost, mInputSize);
    cudaMallocHost(&mOutputHost, mOutputSize);
    for (size_t i = 0; i < mInputSize / sizeof(float); ++i) {
        mInputHost[i] = 1.0f;
    }
    cudaMalloc(&mInputDevice, mInputSize);
    cudaMalloc(&mOutputDevice, mOutputSize);
    
}

void Model::init_data(nvinfer1::Dims input_dims, float* camera_parm, float* device_camera_parm){
    int32_t camera_parm_size  = getDimSize(input_dims) * sizeof(float);

    cudaMallocHost(&camera_parm, camera_parm_size);
    for (size_t i = 0; i < camera_parm_size / sizeof(float); ++i) {
        camera_parm[i] = 1.0f;
    }
    cudaMalloc(&device_camera_parm, camera_parm_size);
}

void Model::init_output_data(nvinfer1::Dims output_dims){
    mOutputSize = getDimSize(output_dims) * sizeof(float);
    cudaMallocHost(&mOutputHost, mOutputSize);
    cudaMalloc(&mOutputDevice, mOutputSize);
}

void Model::init_output_data_half(nvinfer1::Dims output_dims){
    mOutputSize = getDimSize(output_dims) * sizeof(half);
    cudaMallocHost(&mOutputHost_half, mOutputSize);
    cudaMalloc(&mOutputDevice_half, mOutputSize);
}

void Model::init_data(nvinfer1::Dims input_dims, nvinfer1::Dims output_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize  = getDimSize(input_dims) * sizeof(float);
    mOutputSize = getDimSize(output_dims) * sizeof(float);
    // CUDA_CHECK(cudaMallocHost(&mInputHost, mInputSize));
    CUDA_CHECK(cudaMallocHost(&mOutputHost, mOutputSize));

    CUDA_CHECK(cudaMalloc(&mInputDevice, mInputSize));
    CUDA_CHECK(cudaMalloc(&mOutputDevice, mOutputSize));

    float mean=0;
    float std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            // cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            float* imagePtr = mInputDevice + i * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
            preprocess_gpu<float>(image, imagePtr, IMAGE_HEIGHT, IMAGE_WIDTH, 1, mean, std);
        }
    }
}

void Model::init_data_cpu_fish(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_fish_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_fish_imgs, mInputSize_fish_img));

    float mean=0;
    float std=0.003921568;
    
    for (int i = 0; i < NUM_IMAGES; ++i) {
        size_t offset = i * IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS;
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        cv::Mat float_img;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            float data[NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
            cv::resize(image, resizedInput_gpu, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            resizedInput_gpu.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);
            std::vector<float> chw_data(IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS);
    
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                for (int h = 0; h < IMAGE_HEIGHT; ++h) {
                    for (int w = 0; w < IMAGE_WIDTH; ++w) {
                        chw_data[c * IMAGE_HEIGHT * IMAGE_WIDTH + h * IMAGE_WIDTH + w] = 
                            float_img.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            // std::ofstream file("./work_dir/vt_debug/front1.bin", std::ios::binary);
            // // 直接写入数据
            // file.write(reinterpret_cast<const char*>(chw_data.data()), 
            //         chw_data.size() * sizeof(float));
            
            // file.close();
            CUDA_CHECK(cudaMemcpy(mInputDevice_fish_imgs + offset, chw_data.data(), IMAGE_HEIGHT*IMAGE_WIDTH*3*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

void Model::init_data_cpu_pv(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_pv_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_pv_imgs, mInputSize_pv_img));

    float mean=0;
    float std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        size_t offset = i * IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS;
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        cv::Mat float_img;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            float data[NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
            cv::resize(image, resizedInput_gpu, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            resizedInput_gpu.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);
            std::vector<float> chw_data(IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS);
    
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                for (int h = 0; h < IMAGE_HEIGHT; ++h) {
                    for (int w = 0; w < IMAGE_WIDTH; ++w) {
                        chw_data[c * IMAGE_HEIGHT * IMAGE_WIDTH + h * IMAGE_WIDTH + w] = 
                            float_img.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            CUDA_CHECK(cudaMemcpy(mInputDevice_pv_imgs + offset, chw_data.data(), IMAGE_HEIGHT*IMAGE_WIDTH*3*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

void Model::init_data_cpu_front(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_front_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_front_imgs, mInputSize_front_img));

    float mean=0;
    float std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        size_t offset = i * IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS;
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        cv::Mat float_img;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            float data[NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
            cv::resize(image, resizedInput_gpu, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            resizedInput_gpu.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);
            std::vector<float> chw_data(IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS);
    
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                for (int h = 0; h < IMAGE_HEIGHT; ++h) {
                    for (int w = 0; w < IMAGE_WIDTH; ++w) {
                        chw_data[c * IMAGE_HEIGHT * IMAGE_WIDTH + h * IMAGE_WIDTH + w] = 
                            float_img.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            CUDA_CHECK(cudaMemcpy(mInputDevice_front_imgs + offset, chw_data.data(), IMAGE_HEIGHT*IMAGE_WIDTH*3*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

void Model::init_data_cpu_fish(nvinfer1::Dims input_dims){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_fish_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_fish_imgs, mInputSize_fish_img));
    CUDA_CHECK(cudaMemset(mInputDevice_fish_imgs, 1, mInputSize_fish_img));
}

void Model::init_data_cpu_pv(nvinfer1::Dims input_dims){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_front_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_pv_imgs, mInputSize_front_img));
    CUDA_CHECK(cudaMemset(mInputDevice_pv_imgs, 1, mInputSize_front_img));
}

void Model::init_data_cpu_front(nvinfer1::Dims input_dims){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_front_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_front_imgs, mInputSize_front_img));
    CUDA_CHECK(cudaMemset(mInputDevice_front_imgs, 1, mInputSize_front_img));
}


void Model::init_input_fish_img_data_half(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_fish_img  = getDimSize(input_dims) * sizeof(half);
    CUDA_CHECK(cudaMalloc(&mInputDevice_fish_imgs_half, mInputSize_fish_img));

    half mean=0.;
    half std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            // cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            half* imagePtr = mInputDevice_fish_imgs_half + i * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
            preprocess_gpu<half>(image, imagePtr, IMAGE_HEIGHT, IMAGE_WIDTH, 1, mean, std);
        }
    }
}
void Model::init_input_fish_img_data(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    // float
    mInputSize_fish_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_fish_imgs, mInputSize_fish_img));

    float mean=0;
    float std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            // cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            float* imagePtr = mInputDevice_fish_imgs + i * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
            preprocess_gpu<float>(image, imagePtr, IMAGE_HEIGHT, IMAGE_WIDTH, 1, mean, std);
        }
    }
}

void Model::init_input_pv_img_data_half(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_pv_img  = getDimSize(input_dims) * sizeof(half);
    CUDA_CHECK(cudaMalloc(&mInputDevice_pv_imgs_half, mInputSize_pv_img));

    half mean=0.;
    half std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            // cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            half* imagePtr = mInputDevice_pv_imgs_half + i * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
            preprocess_gpu<half>(image, imagePtr, IMAGE_HEIGHT, IMAGE_WIDTH, 1, mean, std);
        }
    }
}
void Model::init_input_pv_img_data(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];

    // float
    mInputSize_pv_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_pv_imgs, mInputSize_pv_img));

    float mean=0;
    float std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            // cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            float* imagePtr = mInputDevice_pv_imgs + i * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
            preprocess_gpu<float>(image, imagePtr, IMAGE_HEIGHT, IMAGE_WIDTH, 1, mean, std);
        }
    }
}

void Model::init_input_front_img_data_half(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];
    mInputSize_front_img  = getDimSize(input_dims) * sizeof(half);
    CUDA_CHECK(cudaMalloc(&mInputDevice_front_imgs_half, mInputSize_front_img));

    half mean=0.;
    half std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            // cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            half* imagePtr = mInputDevice_front_imgs_half + i * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
            preprocess_gpu<half>(image, imagePtr, IMAGE_HEIGHT, IMAGE_WIDTH, 1, mean, std);
        }
    }
}
void Model::init_input_front_img_data(nvinfer1::Dims input_dims, const std::vector<std::string>& imagePaths){
    const int32_t IMAGE_WIDTH = input_dims.d[3];
    const int32_t IMAGE_HEIGHT = input_dims.d[2];
    const int32_t NUM_CHANNELS = input_dims.d[1];
    const int32_t NUM_IMAGES = input_dims.d[0];

    // float
    mInputSize_front_img  = getDimSize(input_dims) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mInputDevice_front_imgs, mInputSize_front_img));

    float mean=0;
    float std=0.003921568;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << imagePaths[i] << std::endl;
            exit(1);
        }
        cv::Mat resizedInput_gpu;
        // Resize image if necessary
        if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT) {
            // cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
            float* imagePtr = mInputDevice_front_imgs + i * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
            preprocess_gpu<float>(image, imagePtr, IMAGE_HEIGHT, IMAGE_WIDTH, 1, mean, std);
        }
    }
}

