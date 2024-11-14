#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__
#include <cuda_fp16.h>
#include "opencv2/opencv.hpp"
#include "timer.hpp"

cv::Mat preprocess_cpu(cv::Mat &src, const int& tarH, const int& tarW, timer::Timer timer, int tactis);
template<typename T> void preprocess_gpu(cv::Mat &h_src, T* d_tar, const int& tar_h, const int& tar_w, int tactis, T mean, T std);
template<typename T> void resize_bilinear_gpu(T* d_tar, uint8_t* d_src, int tarW, int tarH, int srcH, int srcW, int tactis, T mean, T std);

#endif //__PREPROCESS_HPP__
