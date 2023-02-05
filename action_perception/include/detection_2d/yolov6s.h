#ifndef _YOLOV6S_H
#define _YOLOV6S_H
//头文件内容
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

namespace Action {
namespace Perception {

    static bool ifFileExists(const char* FileName);

    struct Configuration {
        float confThreshold; // Confidence threshold
        float nmsThreshold; // Non-maximum suppression threshold
        float objThreshold; // Object Confidence threshold
        std::string modelpath;
    };

    typedef struct BoxInfo {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    } BoxInfo;

    class YoloV6 {
    public:
        YoloV6(Configuration config);

        ~YoloV6();

        void Inference(cv::Mat& frame);

    private:
        void normalize(cv::Mat img); // 归一化函数

        void nms(std::vector<BoxInfo>& input_boxes);

        cv::Mat resizeImage(cv::Mat srcimg, int* newh, int* neww, int* top, int* left);

        void loadTrt(const std::string strName);

        void loadOnnx(const std::string strName);

        float confThreshold_;
        float nmsThreshold_;
        float objThreshold_;
        int inpWidth_;
        int inpHeight_;
        std::vector<std::string> classes_ = {
            "person", "bicycle", "car", "motorbike",
            "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake",
            "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop",
            "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        };
        const bool keep_ratio_ = true;

        nvinfer1::ICudaEngine* m_CudaEngine;
        nvinfer1::IRuntime* m_CudaRuntime;
        nvinfer1::IExecutionContext* m_CudaContext;
        cudaStream_t m_CudaStream; // //初始化流,CUDA流的类型为cudaStream_t
        int m_iInputIndex;
        int m_iOutputIndex;
        int m_iClassNums;
        int m_iBoxNums;
        cv::Size m_InputSize;
        void* m_ArrayDevMemory[2] { 0 };
        void* m_ArrayHostMemory[2] { 0 };
        int m_ArraySize[2] { 0 };
        std::vector<cv::Mat> m_InputWrappers {};
    };
} // namespace Perception
} // namespace Action

#endif