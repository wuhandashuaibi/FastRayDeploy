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

#include <cstdint>
#include <cuda.h>
#include <algorithm>

constexpr int32_t THREADS_PER_BLOCK{512};

inline int32_t get_blocks(int32_t const N, int32_t const numThreads = THREADS_PER_BLOCK)
{
    int32_t optimalBlockNum = (N + numThreads - 1) / numThreads;
    int32_t maxBlockNum = 4096;

    return std::min<int32_t>(optimalBlockNum, maxBlockNum);
}

struct TensorDesc
{
    int32_t shape[10];
    int32_t stride[10];
    int32_t dim;
};

inline int64_t divUp(int64_t m, int32_t n)
{
    return (m + n - 1) / n;
}

template <class TScalar>
void memcpyPermute(
    TScalar* dst, TScalar const* src, int32_t* src_size, int32_t* permute, int32_t src_dim, cudaStream_t stream = 0);

template <typename TScalar>
cublasStatus_t cublasGemmWrap(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int32_t m,
    int32_t n, int32_t k, TScalar const* alpha, TScalar const* A, int32_t lda, TScalar const* B, int32_t ldb,
    TScalar const* beta, TScalar* C, int32_t ldc);
