#include <iostream>
#include <memory>

#include "utils.hpp"
#include "model.hpp"
#include <iostream>
#include <fstream>
using namespace std;

inline void readbinFile(const std::string& fileName, float* buffer, int32_t dataSize)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    infile.read(reinterpret_cast<char*>(buffer), dataSize*sizeof(float));
}


int main(int argc, char const *argv[])
{
    // Model model("models/onnx/sample_customScalar.onnx", Model::precision::FP32);
    // Model model("models/onnx/sample_customLeakyReLU.onnx", Model::precision::FP16);
    // Model model("models/onnx/sample_customclip.onnx", Model::precision::FP16);
    // Model model("models/onnx/sample_customLeakyReLU.onnx", Model::FP32);
    // Model model("models/onnx/sample_big_solution_plugindcnv2.onnx", Model::FP32);
    // Model model("/home/wuhan/mmlab/custom_plugin/work_dir/demo_sim.onnx", Model::FP32);
    // Model model("models/onnx/voxel_projection.onnx", Model::FP16);
    // Model model("models/onnx/voxel_projection_img.onnx", Model::FP32);
    // Model model("models/onnx/voxel_projection_512.onnx", Model::FP32);
    Model model("models/onnx/voxel_projection_conv_64_permute_int.onnx", Model::FP16);
    // Model model("models/onnx/voxel_projection_conv_img_v1.onnx", Model::FP16);
    // Model model("models/onnx/fast_bev_det.onnx", Model::FP32);
    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }
    if(!model.infer()){
        LOGE("fail in infering model");
        return 0;
    }
    return 0;
    // const std::string fileName = "data.bin";
    // const int dataSize = 75; // 假设要读取的数据大小为 100 个 float
    // float buffer[dataSize];

    // readbinFile(fileName, buffer, dataSize);

    // // 打印读取的数据
    // for (int i = 0; i < dataSize; ++i) {
    //     std::cout << buffer[i] << " ";
    // }
    // std::cout << std::endl;

    // return 0;
}
