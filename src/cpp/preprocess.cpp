#include "preprocess.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#include "timer.hpp"

// 根据比例进行缩放 (CPU版本)
cv::Mat preprocess_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, timer::Timer timer, int tactis) {
    cv::Mat tar;

    int height  = src.rows;
    int width   = src.cols;
    float dim   = std::max(height, width);
    int resizeH = ((height / dim) * tar_h);
    int resizeW = ((width / dim) * tar_w);

    int xOffSet = (tar_w - resizeW) / 2;
    int yOffSet = (tar_h - resizeH) / 2;

    resizeW    = tar_w;
    resizeH    = tar_h;

    timer.start_cpu();

    /*BGR2RGB*/
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

    /*Resize*/
    cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);

    timer.stop_cpu();
    timer.duration_cpu<timer::Timer::ms>("Resize(bilinear) in cpu takes:");

    return tar;
}

// 根据比例进行缩放 (GPU版本)
template<typename T>
void preprocess_gpu(
    cv::Mat &h_src, T* d_tar, const int& tar_h, const int& tar_w, int tacti, T mean, T std) 
{
    // T*       d_tar = nullptr;
    uint8_t* d_src = nullptr;

    // 这里针对传入进来的type进行判断
    int type = (std::is_same<T, uint8_t>::value) ? CV_8UC3 : CV_32FC3;
    cv::Mat h_tar(cv::Size(tar_w, tar_h), type);

    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;

    int src_size  = height * width * chan * sizeof(uint8_t);
    int tar_size  = tar_h * tar_w * chan * sizeof(T);

    // 分配device上的src和tar的内存
    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    // CUDA_CHECK(cudaMalloc(&d_tar, tar_size));

    // 将数据拷贝到device上
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));

    // device上处理resize, BGR2RGB的核函数
    resize_bilinear_gpu<T>(d_tar, d_src, tar_w, tar_h, width, height, tacti, mean, std);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将结果返回给host上
    // CUDA_CHECK(cudaMemcpy(h_tar.data, d_tar, tar_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_src));
    // CUDA_CHECK(cudaFree(d_tar));

    // return h_tar;
}


template void preprocess_gpu<int8_t>(cv::Mat &h_src, int8_t* d_tar, const int& tar_h, const int& tar_w, int tacti, int8_t mean, int8_t std);
template void preprocess_gpu<half>(cv::Mat &h_src, half* d_tar, const int& tar_h, const int& tar_w, int tacti, half mean, half std);
template void preprocess_gpu<float>(cv::Mat &h_src, float* d_tar, const int& tar_h, const int& tar_w, int tacti, float mean, float std) ;


