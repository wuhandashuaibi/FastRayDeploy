## 该项目包含CUDA的VT Plugin以及C++模型转换工具
### How to use
- requirements: cuda, cudnn, tensorrt 新一点的任意版本应该都行（推荐CUDA11.8、CUDNN8.6.0.163、TensorRT8.6.1.7）
- 根据CMakeLists中的变量路径配置到对应的相关库
- 编译：mkdir build && cd build && cmake . && make -j8
- 运行：./build/testcase/infer_sample
- 上述运行脚本会自动根据给定ONNX模型生成对应的TensorRT engine，并执行推理（如果存在指定的engine文件则直接加载推理）

### Code Structure
- src 库的路径以及模型构建推理的源代码以及头文件
- testcase 测试代码

### 代码具体细节
- main.cpp中具体指定ONNX模型路径以及指定模型转换精度FP32 or FP16
- 包含两个CUDA Kernel
  1. ProjectionKernel_Temp
  2. resize_bilinear_hwc2chw_kernel
  以上两个CUDA Kernel都使用模板函数进行包装FP32以及FP16
- FastRay的VT转换module包含两种类型相机（共三组不同图像）共享同一Plugin

### Plugin 实现具体细节
- 做好一个输入输出能够表达出该VT module的一个dummy ONNX模型以便能够正常导出整个模型（具体参考prefusion中的contrib/fastbev_det/models/necks/view_transform.py）ONNX模型中要将LUT的uu vv 等参数作为模型的一种参数写入到ONNX模型中，减少VT module直接输入的麻烦
- CUDA Plugin的实现主要包含几个实现方法：
  1.确认ONNX中本身attribute中包含的参数 
    '''
    int32_t mVoxel_num;
    int32_t mOutput_size_b;
    int32_t mOutput_size_c;
    int32_t mOutput_size_h;
    int32_t mOutput_size_w;
    '''
  2.getOutputDimensions（该Plugin应该输出的Tensor的大小，有形状的）
  3.supportsFormatCombination（该Plugin支持的数据类型FP32 or FP16）
  4.enqueue（真正执行推理的函数，这里主要完成CUDA Kernel的调用）
  5.getWorkspaceSize（获取该Plugin所需要的额外内存空间大小）
  6.destroy（释放插件中申请的内存）
- enqueue中主要完成CUDA Kernel的调用，包括：
  1.因为ONNX模型的LUT表包含七个Tensor，所以该Plugin包括5个输入Tensor
  '''
  void const* x_fish       = inputs[0];
  void const* x_pv         = inputs[1];
  void const* x_front      = inputs[2];
  void const* uu      = inputs[1];
  void const* vv      = inputs[2];
  void const* valid   = inputs[3];
  void const* density = inputs[4];
  '''
  2.然后通过LUT在每个图像上进行查询然后投影搬运到VOXEL空间中
  3.得到最终的output Tensor
