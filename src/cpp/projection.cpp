/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 **************************************************************************
 * Modified from mmcv (https://github.com/open-mmlab/mmcv/tree/master/mmcv)
 * Copyright (c) OpenMMLab. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/open-mmlab/mmcv/blob/master/LICENSE
 **************************************************************************
 */

#include "projection.hpp"
#include "utils.hpp"
#include <assert.h>
#include <chrono>
#include "common/checkMacrosPlugin.h"


using namespace nvinfer1;

void ProjectionImpl(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);
void ProjectionImpl_1(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);
void ProjectionImpl_half(const half* inputs, half* outputs, const int32_t* uu, const int32_t* vv, 
                    const int32_t* valid, const half* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);
void ProjectionImpl_2(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

void ProjectionImpl_3(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

void ProjectionImpl_4(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

void ProjectionImpl_5(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

void ProjectionImpl_6(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

void ProjectionImpl_7(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

void ProjectionImpl_7_half(const half* inputs, half* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

void ProjectionImpl_8(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

namespace custom
{

REGISTER_TENSORRT_PLUGIN(CustomProjectCreator);


nvinfer1::PluginFieldCollection CustomProjectCreator::mFC{};
std::vector<nvinfer1::PluginField> CustomProjectCreator::mPluginAttributes;

CustomProject::CustomProject(std::string const& name,
    int32_t const voxelnum, int32_t const outputsizeb, int32_t const outputsizec, int32_t const outputsizeh, int32_t const outputsizew)
    : mLayerName(name)
    , mVoxel_num(voxelnum)
    , mOutput_size_b(outputsizeb)
    , mOutput_size_c(outputsizec)
    , mOutput_size_h(outputsizeh)
    , mOutput_size_w(outputsizew)
{
    
}

CustomProject::CustomProject(
    const std::string name, void const* data, size_t length)
    : mLayerName(name)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    mVoxel_num = nvinfer1::plugin::read<int32_t>(d);
    mOutput_size_b = nvinfer1::plugin::read<int32_t>(d);
    mOutput_size_c = nvinfer1::plugin::read<int32_t>(d);
    mOutput_size_h = nvinfer1::plugin::read<int32_t>(d);
    mOutput_size_w = nvinfer1::plugin::read<int32_t>(d);
    PLUGIN_VALIDATE(d == a + length);
}

CustomProject::~CustomProject() {}

nvinfer1::IPluginV2DynamicExt* CustomProject::clone() const noexcept
{
    try
    {
        CustomProject* plugin = new CustomProject(
            mLayerName, mVoxel_num, mOutput_size_b, mOutput_size_c, mOutput_size_h, mOutput_size_w);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        nvinfer1::plugin::caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs CustomProject::getOutputDimensions(int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // TODO
        nvinfer1::DimsExprs ret;
        ret.nbDims = 4;
        // ret.d[0] = exprBuilder.constant(1);
        // ret.d[1] = exprBuilder.constant(2016);

        // ret.d[2] = exprBuilder.constant(240);
        // ret.d[3] = exprBuilder.constant(120);
        // NCHW
        ret.d[0] = exprBuilder.constant(mOutput_size_b);
        ret.d[1] = exprBuilder.constant(mOutput_size_c);
        ret.d[2] = exprBuilder.constant(mOutput_size_h);
        ret.d[3] = exprBuilder.constant(mOutput_size_w);
        //NHWC
        // ret.d[0] = exprBuilder.constant(mOutput_size_b);
        // ret.d[3] = exprBuilder.constant(mOutput_size_c);
        // ret.d[1] = exprBuilder.constant(mOutput_size_h);
        // ret.d[2] = exprBuilder.constant(mOutput_size_w);

        // ret.d[0] = exprBuilder.constant(1);
        // ret.d[1] = exprBuilder.constant(6);
        // ret.d[2] = exprBuilder.constant(240);
        // ret.d[3] = exprBuilder.constant(120);
        // ret.d[4] = exprBuilder.constant(64);
        return ret;
    }
    catch (std::exception const& e)
    {
        nvinfer1::plugin::caughtError(e);
    }
    return DimsExprs{};
}

bool CustomProject::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_VALIDATE(inOut != nullptr);
    PLUGIN_VALIDATE(nbInputs == 7);
    PLUGIN_VALIDATE(nbOutputs == 1);
    PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));
    if (pos < 3)
    {
        // 设置输入的X为指定精度类型
        // return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
        // fp16
        return ((inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
    else if (pos == nbInputs)
    {
        // 设置输出的Output为指定精度类型，并和输入的精度保持一致
        // return ((inOut[0].type == inOut[nbInputs].type) && (inOut[1].type == inOut[nbInputs].type) && (inOut[2].type == inOut[nbInputs].type) && (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
        return ((inOut[0].type == inOut[nbInputs].type && inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[0].type == inOut[nbInputs].type && inOut[pos].type == nvinfer1::DataType::kHALF)) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
    else if (pos == nbInputs-1)
    {
        // 设置权重为FP32
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
    else
    {
        // 设置其他的输入为int32
        return (inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
}

// bool CustomProject::supportsFormatCombination(
//     int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
// {
//     PLUGIN_VALIDATE(inOut != nullptr);
//     PLUGIN_VALIDATE(nbInputs == 5);
//     PLUGIN_VALIDATE(nbOutputs == 1);
//     PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));
//     if (pos == 0)
//     {
//         // 设置输入的X为指定精度类型
//         return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
//     }
//     else if (pos == nbInputs)
//     {
//         // 设置输出的Output为指定精度类型，并和输入的精度保持一致
//         return ((inOut[0].type == inOut[nbInputs].type) && (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
//     }
//     else
//     {
//         // 设置其他的输入为FP32（本质上数据也是FP32）
//         return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
//     }
// }

void CustomProject::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs==7);
    PLUGIN_ASSERT(nbOutputs==1);

    return;
}

size_t CustomProject::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t sizeofDtype = nvinfer1::plugin::bert::getElementSize(outputs[0].type);

    int32_t nInputBatch = outputs[0].dims.d[0];
    int32_t nInputChannel = outputs[0].dims.d[1];
    int32_t outputHeight = outputs[0].dims.d[2];
    int32_t outputWidth = outputs[0].dims.d[3];

    int64_t colSize = divUp(nInputBatch * nInputChannel * outputHeight * outputWidth * sizeofDtype, 16) * 16;

    return colSize;
}

int32_t CustomProject::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workSpace,
    cudaStream_t stream) noexcept
{
    try
    {
        // NHWC
        // int32_t batch    = inputDesc[0].dims.d[0];
        // int32_t channels = inputDesc[0].dims.d[3];
        // int32_t height   = inputDesc[0].dims.d[1];
        // int32_t width    = inputDesc[0].dims.d[2];
        // NCHW
        parm_feature fish_parm;
        fish_parm.cam_num = inputDesc[0].dims.d[0];
        fish_parm.channel = inputDesc[0].dims.d[3];
        fish_parm.height  = inputDesc[0].dims.d[1];
        fish_parm.width   = inputDesc[0].dims.d[2];
        
        parm_feature pv_parm;
        pv_parm.cam_num   = inputDesc[1].dims.d[0];
        pv_parm.channel   = inputDesc[1].dims.d[3];
        pv_parm.height    = inputDesc[1].dims.d[1];
        pv_parm.width     = inputDesc[1].dims.d[2];

        parm_feature front_parm;
        front_parm.cam_num = inputDesc[2].dims.d[0];
        front_parm.channel = inputDesc[2].dims.d[3];
        front_parm.height  = inputDesc[2].dims.d[1];
        front_parm.width   = inputDesc[2].dims.d[2];


        void const* x_fish       = inputs[0];
        void const* x_pv         = inputs[1];
        void const* x_front      = inputs[2];
        void const* uu      = inputs[3];
        void const* vv      = inputs[4];
        void const* valid   = inputs[5];
        void const* density = inputs[6];

        void* output = outputs[0];
        // int32_t im2colStep = std::min(batch, 32);
        
        auto data_type = inputDesc[0].type;
        LOG("%d ", data_type);
        switch (data_type)
        {
        case nvinfer1::DataType::kFLOAT:
            LOG("data float");
            ProjectionImpl_Temp_V3<float>((float*) x_fish, (float*) x_pv, (float*) x_front, (float*) output, (int32_t*) uu,
                (int32_t*) vv, (int32_t*) valid, (float*) density, fish_parm, pv_parm, front_parm, mVoxel_num, stream);
            break;
        case nvinfer1::DataType::kHALF:
            LOG("data half");
            ProjectionImpl_Temp_V3<half>((half*) x_fish, (half*) x_pv, (half*) x_front, (half*) output, (int32_t*) uu,
                (int32_t*) vv, (int32_t*) valid, (float*) density, fish_parm, pv_parm, front_parm, mVoxel_num, stream);
            break;
        default: return 1;
        }
    }
    catch (std::exception const& e)
    {
        nvinfer1::plugin::caughtError(e);
    }

    return 0;
}


// int32_t CustomProject::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
//     nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workSpace,
//     cudaStream_t stream) noexcept
// {
//     try
//     {
//         // NHWC
//         // int32_t batch    = inputDesc[0].dims.d[0];
//         // int32_t channels = inputDesc[0].dims.d[3];
//         // int32_t height   = inputDesc[0].dims.d[1];
//         // int32_t width    = inputDesc[0].dims.d[2];
//         // NCHW
//         int32_t batch    = inputDesc[0].dims.d[0];
//         int32_t channels = inputDesc[0].dims.d[1];
//         int32_t height   = inputDesc[0].dims.d[2];
//         int32_t width    = inputDesc[0].dims.d[3];
        
//         // int32_t channelsOut = outputDesc[0].dims.d[1];
//         // int32_t kernelH = inputDesc[3].dims.d[2];
//         // int32_t kernelW = inputDesc[3].dims.d[3];
//         int32_t nElements = 1;
//         nElements = batch * channels * height * width;

//         void const* x       = inputs[0];
//         void const* uu      = inputs[1];
//         void const* vv      = inputs[2];
//         void const* valid   = inputs[3];
//         void const* density = inputs[4];

//         void* output = outputs[0];
//         int32_t im2colStep = std::min(batch, 32);
        
//         auto data_type = inputDesc[0].type;
//         LOG("%d ", data_type);
//         switch (data_type)
//         {
//         case nvinfer1::DataType::kFLOAT:
//             LOG("data float");
//             ProjectionImpl_Temp<float>((float*) x, (float*) output, (float*) uu,
//                 (float*) vv, (float*) valid, (float*) density, batch, mVoxel_num, nElements, channels, height, width, stream);
//             break;
//         case nvinfer1::DataType::kHALF:
//             LOG("data half");
//             ProjectionImpl_Temp<half>((half*) x, (half*) output, (float*) uu,
//                 (float*) vv, (float*) valid, (float*) density, batch, mVoxel_num, nElements, channels, height, width, stream);
//             break;
//         default: return 1;
//         }
//     }
//     catch (std::exception const& e)
//     {
//         nvinfer1::plugin::caughtError(e);
//     }

//     return 0;
// }

nvinfer1::DataType CustomProject::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

// IPluginV2 Methods
char const* CustomProject::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

char const* CustomProject::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t CustomProject::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CustomProject::initialize() noexcept
{
    return 0;
}

void CustomProject::terminate() noexcept {}

size_t CustomProject::getSerializationSize() const noexcept
{
    return sizeof(mVoxel_num) + sizeof(mOutput_size_b) + sizeof(mOutput_size_c) + sizeof(mOutput_size_h) + sizeof(mOutput_size_w);
}

void CustomProject::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    nvinfer1::plugin::write(d, mVoxel_num);
    nvinfer1::plugin::write(d, mOutput_size_b);
    nvinfer1::plugin::write(d, mOutput_size_c);
    nvinfer1::plugin::write(d, mOutput_size_h);
    nvinfer1::plugin::write(d, mOutput_size_w);
}

void CustomProject::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void CustomProject::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept
{
    mCublasHandle = cublasContext;
}

void CustomProject::detachFromContext() noexcept {}

void CustomProject::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        nvinfer1::plugin::caughtError(e);
    }
}

char const* CustomProject::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

CustomProjectCreator::CustomProjectCreator()
{
    mPluginAttributes.emplace_back(nvinfer1::PluginField("voxelnum"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("outputsizeb"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("outputsizec"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("outputsizeh"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("outputsizew"));
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CustomProjectCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const* CustomProjectCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* CustomProjectCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* CustomProjectCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {   
        int32_t voxelnum = 1;
        int32_t outputsizeb = 1;
        int32_t outputsizec = 1;
        int32_t outputsizeh = 1;
        int32_t outputsizew = 1;
        std::map<std::string, int32_t*> paramMap = {{"voxelnum", &voxelnum}, {"outputsizeb", &outputsizeb}, {"outputsizec", &outputsizec}, {"outputsizeh", &outputsizeh}, {"outputsizew", &outputsizew}};
        LOG("createplugin");
        for (int i = 0; i < fc->nbFields; i++) {
            if (paramMap.find(fc->fields[i].name) != paramMap.end()){
                *paramMap[fc->fields[i].name] = *reinterpret_cast<const int32_t*>(fc->fields[i].data);  // must match datatype with top
            }
        }
        CustomProject* plugin
            = new CustomProject(name, voxelnum, outputsizeb, outputsizec, outputsizeh, outputsizew);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        nvinfer1::plugin::caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2* CustomProjectCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto plugin = new CustomProject(name, serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        nvinfer1::plugin::caughtError(e);
    }
    return nullptr;
}

void CustomProjectCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        nvinfer1::plugin::caughtError(e);
    }
}

char const* CustomProjectCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
} // namespace