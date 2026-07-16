#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>      // 引入 OpenCV DNN 模块
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

/**
 * @brief 解析后的纯净 YOLO 目标结构体
 * @note  这是一个经过后处理(NMS)后，直接可供 PnP 解算的最终物理对象
 */
struct YoloObject 
{
    int color;                    // 颜色: 0->红, 1->蓝, 2->灰(熄灭), 3->紫
    int number;                   // 编号: 1(英雄) 2(工程) 3/4/5(步兵) 6(哨兵) 7(前哨站) 8(基地)
    bool is_big;                  // 尺寸: true->大装甲板(225mm), false->小装甲板(135mm)
    
    float prob;                   // 综合置信度 (代表模型对该目标的确信程度)
    cv::Rect box;                 // 生成的外接矩形框 (仅用于 NMS 非极大值抑制，不参与解算)
    std::vector<cv::Point2f> pts; // 模型直接回归出的4个物理角点 (左上, 左下, 右下, 右上)
};

/**
 * @brief YOLO 目标检测器类 (基于 OpenCV DNN)
 * @note  支持一键切换 CPU / GPU(OpenVINO) 加速推理
 */
class YoloDetector 
{
public:
    /**
     * @brief 构造函数，初始化网络模型
     * @param model_path 模型(.onnx / .bin)的绝对路径
     * @param use_gpu    是否开启 GPU/OpenVINO 核显加速 (默认 false 使用 CPU)
     */
    YoloDetector(const std::string& model_path, bool use_gpu = true);
    
    /**
     * @brief 执行一帧图像的目标检测
     * @param raw_img     相机传来的 BGR 原图
     * @param enemy_color 我们要打击的敌方颜色 (用于在底层直接抛弃友方目标，节省算力)
     * @return std::vector<YoloObject> 检测到的所有敌方装甲板集合
     */
    std::vector<YoloObject> Detect(const cv::Mat& raw_img, int enemy_color);
 
private:
    bool is_ready_ = false;              // 模型加载状态标志位，防止空指针崩溃
    cv::dnn::Net net_;                   // OpenCV DNN 网络核心对象

    // --- 深大模型的超参数 ---
    const float score_threshold_ = 0.65; // 置信度阈值 (低于该值的候选框直接被抛弃)
    const float nms_threshold_ = 0.45;   // NMS重叠阈值 (去除重叠度大于 45% 的重复框)

    // ======================================================================
    // 【内存池优化】将所有会频繁 push_back 的容器作为类成员，避免每帧重复 new 内存
    // ======================================================================
    std::vector<int> class_ids;
    std::vector<int> color_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> all_keypoints;
    std::vector<int> indices;
    
    std::vector<cv::Mat> net_outputs_;   // 接收网络输出的容器
    cv::Mat transposed_buffer_;          // 用于存储转置后的矩阵，避免每帧重新分配显存
};

#endif // YOLO_DETECTOR_HPP