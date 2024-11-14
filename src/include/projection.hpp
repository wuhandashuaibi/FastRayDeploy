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

#ifndef TRT_CUSTOM_CLIP_H
#define TRT_CUSTOM_CLIP_H
#include <cstdint>
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include "helper.h"
#include "common/bertCommon.h"
#include "common/checkMacrosPlugin.h"
#include "common/plugin.h"
#include "common/serialize.hpp"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"

using namespace nvinfer1;

struct parm_feature
{
    int32_t cam_num; 
    int32_t channel;
    int32_t height;
    int32_t width;
};


template <typename T> void ProjectionImpl_Temp(const T* inputs, T* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);
template <typename T> void ProjectionImpl_Temp_V2(const T* inputs_fish, const T* inputs_pv, const T* inputs_front, T* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, 
                    parm_feature fish_parm, parm_feature pv_parm, parm_feature front_parm, int32_t length,
                    cudaStream_t stream);
template <typename T> void ProjectionImpl_Temp_V3(const T* inputs_fish, const T* inputs_pv, const T* inputs_front, T* outputs, const int32_t* uu, const int32_t* vv, 
                    const int32_t* valid, const float* density, 
                    parm_feature fish_parm, parm_feature pv_parm, parm_feature front_parm, int32_t length,
                    cudaStream_t stream);
namespace custom
{
static const char* PLUGIN_NAME {"CustomProject"};
static const char* PLUGIN_VERSION {"1"};


class CustomProject : public nvinfer1::IPluginV2DynamicExt
{
public:
    CustomProject(std::string const& name, int32_t const voxelnum, int32_t const outputsizeb, int32_t const outputsizec, int32_t const outputsizeh, int32_t const outputsizew);

    CustomProject(const std::string name, void const* data, size_t length);

    CustomProject() = delete;

    ~CustomProject() override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    int32_t mVoxel_num;

    int32_t mOutput_size_b;
    int32_t mOutput_size_c;
    int32_t mOutput_size_h;
    int32_t mOutput_size_w;

    // std::vector<int32_t> uu;
    // std::vector<int32_t> vv;
    // std::vector<int32_t> valid;
    // std::vector<float> density;

    cublasHandle_t mCublasHandle;
};

class CustomProjectCreator : public nvinfer1::IPluginCreator
{
public:
    CustomProjectCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace custom

#endif // TRT_CUSTOM_CLIP_H

