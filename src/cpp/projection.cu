#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <cuda_fp16.h>
#include "utils.hpp"
#include "projection.hpp"


template <typename T>
__global__ void ProjectionKernel_Temp(
    const T* input, T* output, 
    const float* uu, const float* vv, const float* valid,
    const float* density, const int32_t cam_num, int32_t length, int32_t channel, int32_t height, int32_t width, const int32_t nElements)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) 
        return;
    const int32_t offset = channel * width * height;
    int32_t output_index = index * channel;
    for (size_t c_id = 0; c_id < channel; c_id++)
    {
        T tmp = 0.;
        for (size_t cam_id = 0; cam_id < cam_num; cam_id++)
        {
            T valid_ = static_cast<T>(valid[length * cam_id + index]);            
            if (valid_==T{1})
            {   
                int32_t uu_ = static_cast<int32_t>(uu[length * cam_id + index]);
                int32_t vv_ = static_cast<int32_t>(vv[length * cam_id + index]);
                assert(uu_ >= 0 && "uu must be greater than or equal to zero");
                assert(vv_ >= 0 && "vv must be greater than or equal to zero");
                T density_  = static_cast<T>(density[length * cam_id + index]);  
                // printf("|%f %d %d %f|", __half2float(valid_), uu_, vv_, __half2float(density_));
                // HWC
                // tmp += input[vv_ * width * channel + uu_ * channel + c_id + cam_id * offset] * density_;
                // CHW
                tmp += input[vv_ * width + uu_ + c_id*width*height + cam_id * offset] * density_;
                // printf("%f ", __half2float(density_));
            }
        }
        // printf("%f ", tmp);
        // C*6*HW
        output[index + length * c_id] = tmp;
        // 6*HW*C
        // output[output_index + c_id] = tmp;
    }
}
template __global__ void ProjectionKernel_Temp<float>(const float* input, float* output, 
            const float* uu, const float* vv, const float* valid,
            const float* density, const int32_t cam_num, int32_t length, int32_t channel, int32_t height, int32_t width, const int32_t nElements);
template __global__ void ProjectionKernel_Temp<half>(const half* input, half* output, 
            const float* uu, const float* vv, const float* valid,
            const float* density, const int32_t cam_num, int32_t length, int32_t channel, int32_t height, int32_t width, const int32_t nElements);



template <typename T>
__global__ void ProjectionKernel_Temp_V2(
    const T* input_fish, const T* input_pv, const T* input_front, T* output, 
    const float* uu, const float* vv, const float* valid,
    const float* density, parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front, int32_t length)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) 
        return;
    const int32_t fish_offset = parm_fish.channel * parm_fish.height * parm_fish.width;    
    const int32_t pv_offset = parm_pv.channel * parm_pv.height * parm_pv.width;    
    const int32_t front_offset = parm_front.channel * parm_front.height * parm_front.width;    

    // for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
    // {
    //     T tmp = 0.;
    //     for (size_t cam_id = 0; cam_id < 10; cam_id++)
    //     {
    //         T valid_ = static_cast<T>(valid[length * cam_id + index]);            
    //         if (valid_==T{1})
    //         {   
    //             int32_t uu_ = static_cast<int32_t>(uu[length * cam_id + index]);
    //             int32_t vv_ = static_cast<int32_t>(vv[length * cam_id + index]);
    //             assert(uu_ >= 0 && "uu must be greater than or equal to zero");
    //             assert(vv_ >= 0 && "vv must be greater than or equal to zero");
    //             T density_  = static_cast<T>(density[length * cam_id + index]);  
    //             if (cam_id < 4)
    //             {
    //                 // CHW
    //                 // tmp += input_fish[vv_ * parm_fish.width + uu_ + c_id * parm_fish.width * parm_fish.height + cam_id * fish_offset] * density_;
    //                 // HWC
    //                 tmp += input_fish[vv_ * parm_fish.width * parm_fish.channel + uu_ * parm_fish.channel + c_id + cam_id * fish_offset] * density_;
    //             }
    //             else if(cam_id <= 8)
    //             {
    //                 // tmp += input_pv[vv_ * parm_pv.width + uu_ + c_id * parm_pv.width * parm_pv.height + (cam_id - 4) * pv_offset] * density_;
    //                 tmp += input_pv[vv_ * parm_pv.width * parm_pv.channel + uu_ * parm_pv.channel + c_id + (cam_id - 4) * pv_offset] * density_;
    //             }
    //             else
    //             {
    //                 // tmp += input_front[vv_ * parm_front.width + uu_ + c_id * parm_front.width * parm_front.height] * density_;  // only one img
    //                 tmp += input_front[vv_ * parm_front.width * parm_front.channel + uu_ * parm_front.channel + c_id] * density_;
    //             }
    //         }
    //     }
    //     // printf("%f ", tmp);
    //     // C*6*HW
    //     output[index + length * c_id] = tmp;
    //     // 6*HW*C
    //     // output[output_index + c_id] = tmp;

    T reg_cache[3];  

    for (size_t cam_id = 0; cam_id < 10; cam_id++)
    {
        T valid_ = static_cast<T>(valid[length * cam_id + index]);            
        if (valid_==T{1})
        {   
            int32_t uu_ = static_cast<int32_t>(uu[length * cam_id + index]);
            int32_t vv_ = static_cast<int32_t>(vv[length * cam_id + index]);
            assert(uu_ >= 0 && "uu must be greater than or equal to zero");
            assert(vv_ >= 0 && "vv must be greater than or equal to zero");
            T density_  = static_cast<T>(density[length * cam_id + index]);  
            if (cam_id < 4)
            {
                for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
                {
                    // HWC
                    reg_cache[c_id] += input_fish[vv_ * parm_fish.width * parm_fish.channel + uu_ * parm_fish.channel + c_id + cam_id * fish_offset] * density_;
                    // CHW
                    // tmp += input[vv_ * width + uu_ + c_id*width*height + cam_id * offset] * density_;
                }
            }
            else if(cam_id <= 8)
            {
                for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
                {
                    // HWC
                    reg_cache[c_id] += input_pv[vv_ * parm_pv.width * parm_pv.channel + uu_ * parm_pv.channel + c_id + (cam_id - 4) * pv_offset] * density_;
                    // CHW
                    // tmp += input[vv_ * width + uu_ + c_id*width*height + cam_id * offset] * density_;
                }
            }
            else
            {
                for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
                {
                    // HWC
                    reg_cache[c_id] += input_front[vv_ * parm_front.width * parm_front.channel + uu_ * parm_front.channel + c_id] * density_;
                    // CHW
                    // tmp += input[vv_ * width + uu_ + c_id*width*height + cam_id * offset] * density_;
                }
            }
        }
    }
    int32_t output_index = index * parm_fish.channel;
    for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
    {   
        // printf("%f ", tmp);
        // C*6*HW
        output[index + length * c_id] = reg_cache[c_id];
        // 6*HW*C
        // output[output_index + c_id] = reg_cache[c_id];
    }
}
template __global__ void ProjectionKernel_Temp_V2<float>(const float* input_fish, const float* input_pv, const float* input_front, float* output, 
            const float* uu, const float* vv, const float* valid,
            const float* density, parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front, int32_t length);
template __global__ void ProjectionKernel_Temp_V2<half>(const half* input_fish, const half* input_pv, const half* input_front, half* output, 
            const float* uu, const float* vv, const float* valid,
            const float* density, parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front, int32_t length);


template <typename T>
__global__ void ProjectionKernel_Temp_V3(
    const T* input_fish, const T* input_pv, const T* input_front, T* output, 
    const int32_t* uu, const int32_t* vv, const int32_t* valid,
    const float* density, parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front, int32_t length)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) 
        return;
    const int32_t fish_offset = parm_fish.channel * parm_fish.height * parm_fish.width;    
    const int32_t pv_offset = parm_pv.channel * parm_pv.height * parm_pv.width;    
    const int32_t front_offset = parm_front.channel * parm_front.height * parm_front.width;    

    // for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
    // {
    //     T tmp = 0.;
    //     for (size_t cam_id = 0; cam_id < 10; cam_id++)
    //     {
    //         T valid_ = static_cast<T>(valid[length * cam_id + index]);            
    //         if (valid_==T{1})
    //         {   
    //             int32_t uu_ = static_cast<int32_t>(uu[length * cam_id + index]);
    //             int32_t vv_ = static_cast<int32_t>(vv[length * cam_id + index]);
    //             assert(uu_ >= 0 && "uu must be greater than or equal to zero");
    //             assert(vv_ >= 0 && "vv must be greater than or equal to zero");
    //             T density_  = static_cast<T>(density[length * cam_id + index]);  
    //             if (cam_id < 4)
    //             {
    //                 // CHW
    //                 // tmp += input_fish[vv_ * parm_fish.width + uu_ + c_id * parm_fish.width * parm_fish.height + cam_id * fish_offset] * density_;
    //                 // HWC
    //                 tmp += input_fish[vv_ * parm_fish.width * parm_fish.channel + uu_ * parm_fish.channel + c_id + cam_id * fish_offset] * density_;
    //             }
    //             else if(cam_id <= 8)
    //             {
    //                 // tmp += input_pv[vv_ * parm_pv.width + uu_ + c_id * parm_pv.width * parm_pv.height + (cam_id - 4) * pv_offset] * density_;
    //                 tmp += input_pv[vv_ * parm_pv.width * parm_pv.channel + uu_ * parm_pv.channel + c_id + (cam_id - 4) * pv_offset] * density_;
    //             }
    //             else
    //             {
    //                 // tmp += input_front[vv_ * parm_front.width + uu_ + c_id * parm_front.width * parm_front.height] * density_;  // only one img
    //                 tmp += input_front[vv_ * parm_front.width * parm_front.channel + uu_ * parm_front.channel + c_id] * density_;
    //             }
    //         }
    //     }
    //     // printf("%f ", tmp);
    //     // C*6*HW
    //     output[index + length * c_id] = tmp;
    //     // 6*HW*C
    //     // output[output_index + c_id] = tmp;

    T reg_cache[3];  

    for (size_t cam_id = 0; cam_id < 10; cam_id++)
    {
        int32_t valid_ = static_cast<int32_t>(valid[length * cam_id + index]);            
        if (valid_ == 1)
        {   
            int32_t uu_ = static_cast<int32_t>(uu[length * cam_id + index]);
            int32_t vv_ = static_cast<int32_t>(vv[length * cam_id + index]);
            assert(uu_ >= 0 && "uu must be greater than or equal to zero");
            assert(vv_ >= 0 && "vv must be greater than or equal to zero");
            T density_  = static_cast<T>(density[length * cam_id + index]);  
            if (cam_id < 4)
            {
                for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
                {
                    // HWC
                    reg_cache[c_id] += input_fish[vv_ * parm_fish.width * parm_fish.channel + uu_ * parm_fish.channel + c_id + cam_id * fish_offset] * density_;
                    // CHW
                    // tmp += input[vv_ * width + uu_ + c_id*width*height + cam_id * offset] * density_;
                }
            }
            else if(cam_id <= 8)
            {
                for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
                {
                    // HWC
                    reg_cache[c_id] += input_pv[vv_ * parm_pv.width * parm_pv.channel + uu_ * parm_pv.channel + c_id + (cam_id - 4) * pv_offset] * density_;
                    // CHW
                    // tmp += input[vv_ * width + uu_ + c_id*width*height + cam_id * offset] * density_;
                }
            }
            else
            {
                for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
                {
                    // HWC
                    reg_cache[c_id] += input_front[vv_ * parm_front.width * parm_front.channel + uu_ * parm_front.channel + c_id] * density_;
                    // CHW
                    // tmp += input[vv_ * width + uu_ + c_id*width*height + cam_id * offset] * density_;
                }
            }
        }
    }
    int32_t output_index = index * parm_fish.channel;
    for (size_t c_id = 0; c_id < parm_fish.channel; c_id++)
    {   
        // printf("%f ", tmp);
        // C*6*HW
        output[index + length * c_id] = reg_cache[c_id];
        // 6*HW*C
        // output[output_index + c_id] = reg_cache[c_id];
    }
}
template __global__ void ProjectionKernel_Temp_V3<float>(const float* input_fish, const float* input_pv, const float* input_front, float* output, 
            const int32_t* uu, const int32_t* vv, const int32_t* valid,
            const float* density, parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front, int32_t length);
template __global__ void ProjectionKernel_Temp_V3<half>(const half* input_fish, const half* input_pv, const half* input_front, half* output, 
            const int32_t* uu, const int32_t* vv, const int32_t* valid,
            const float* density, parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front, int32_t length);


template <typename T>
void ProjectionImpl_Temp(const T* inputs, T* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize(((length + blockSize.x - 1) / blockSize.x), 1, 1);
    
    cudaEvent_t start, stop;
    if (cam_num==4)
    {
        float* front_img;
        cudaMallocHost(&front_img, 4*80*128*3*sizeof(T));
        cudaMemcpyAsync(front_img, inputs, 4*80*128*3*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::string bin_file = "./work_dir/vt_debug/fish_img.bin";
        auto f = fopen(bin_file.c_str(), "wb");

        fwrite(front_img, sizeof(T), 4*80*128*3, f);
        fclose(f);
    }
    if (cam_num==5)
    {
        float* front_img;
        cudaMallocHost(&front_img, 5*96*128*3*sizeof(T));
        cudaMemcpyAsync(front_img, inputs, 5*96*128*3*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::string bin_file = "./work_dir/vt_debug/pv_img.bin";
        auto f = fopen(bin_file.c_str(), "wb");

        fwrite(front_img, sizeof(T), 5*96*128*3, f);
        fclose(f);
    }
    if (cam_num==1)
    {
        float* front_img;
        cudaMallocHost(&front_img, 1*96*192*3*sizeof(T));
        cudaMemcpyAsync(front_img, inputs, 1*96*192*3*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::string bin_file = "./work_dir/vt_debug/front1.bin";
        auto f = fopen(bin_file.c_str(), "wb");

        fwrite(front_img, sizeof(T), 1*96*192*3, f);
        fclose(f);
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start, 0);
    ProjectionKernel_Temp<T><<<gridSize, blockSize, 0, stream>>>(inputs, outputs, uu, vv, valid, density, cam_num, length, channel, height, width, nElements);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    LOG("Plugin total time: %f ms", milliseconds);
}

template void ProjectionImpl_Temp<float>(const float* inputs, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length, 
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);

template void ProjectionImpl_Temp<half>(const half* inputs, half* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, int32_t cam_num, int32_t length,
                    const int32_t nElements, int32_t channel, int32_t height, int32_t width, cudaStream_t stream);


template <typename T>
void ProjectionImpl_Temp_V2(const T* inputs_fish, const T* inputs_pv, const T* inputs_front, T* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, 
                    parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front,
                    int32_t length, cudaStream_t stream)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize(((length + blockSize.x - 1) / blockSize.x), 1, 1);
    
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start, 0);
    ProjectionKernel_Temp_V2<T><<<gridSize, blockSize, 0, stream>>>(inputs_fish, inputs_pv, inputs_front, outputs, uu, vv, valid, density, parm_fish, parm_pv, parm_front, length);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    LOG("Plugin total time: %f ms", milliseconds);
}

template void ProjectionImpl_Temp_V2<float>(const float* inputs_fish, const float* inputs_pv,const float* inputs_front, float* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, 
                    parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front,
                    int32_t length, cudaStream_t stream);

template void ProjectionImpl_Temp_V2<half>(const half* inputs_fish, const half* inputs_pv, const half* inputs_front, half* outputs, const float* uu, const float* vv, 
                    const float* valid, const float* density, 
                    parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front,
                    int32_t length, cudaStream_t stream);


template <typename T>
void ProjectionImpl_Temp_V3(const T* inputs_fish, const T* inputs_pv, const T* inputs_front, T* outputs, const int32_t* uu, const int32_t* vv, 
                    const int32_t* valid, const float* density, 
                    parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front,
                    int32_t length, cudaStream_t stream)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize(((length + blockSize.x - 1) / blockSize.x), 1, 1);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start, 0);
    ProjectionKernel_Temp_V3<T><<<gridSize, blockSize, 0, stream>>>(inputs_fish, inputs_pv, inputs_front, outputs, uu, vv, valid, density, parm_fish, parm_pv, parm_front, length);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    LOG("Plugin total time: %f ms", milliseconds);
}

template void ProjectionImpl_Temp_V3<float>(const float* inputs_fish, const float* inputs_pv,const float* inputs_front, float* outputs, const int32_t* uu, const int32_t* vv, 
                    const int32_t* valid, const float* density, 
                    parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front,
                    int32_t length, cudaStream_t stream);

template void ProjectionImpl_Temp_V3<half>(const half* inputs_fish, const half* inputs_pv, const half* inputs_front, half* outputs, const int32_t* uu, const int32_t* vv, 
                    const int32_t* valid, const float* density, 
                    parm_feature parm_fish, parm_feature parm_pv, parm_feature parm_front,
                    int32_t length, cudaStream_t stream);
