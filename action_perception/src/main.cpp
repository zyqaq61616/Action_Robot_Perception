#include "action_base/base/path.hpp"
#include "detection_2d/yolov6s.h"
#include <unistd.h>

int main()
{
    char* buf;
    buf = get_current_dir_name();
    std::string path = buf;
    clock_t startTime, endTime; //计算时间
    Action::Perception::Configuration yolo_nets = { 0.3, 0.5, 0.3, "yolov5s.engine" };
    Action::Perception::YoloV6 yolo_model(yolo_nets);
    std::string imgpath = path + "/img.jpg";
    cv::Mat srcimg = cv::imread(imgpath);

    double timeStart = (double)cv::getTickCount();
    startTime = clock(); //计时开始
    yolo_model.Inference(srcimg);
    endTime = clock(); //计时结束
    double nTime = ((double)cv::getTickCount() - timeStart) / cv::getTickFrequency();
    std::cout << "clock_running time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
    std::cout << "The run time is:" << (double)clock() / CLOCKS_PER_SEC << "s" << std::endl;
    std::cout << "getTickCount_running time :" << nTime << "sec\n"
              << std::endl;
    // static const string kWinName = "Deep learning object detection in ONNXRuntime";
    // namedWindow(kWinName, WINDOW_NORMAL);
    // imshow(kWinName, srcimg);
    imwrite(path + "/restult_trt.jpg", srcimg);
    // waitKey(0);
    // destroyAllWindows();
    return 0;
}