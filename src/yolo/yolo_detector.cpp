#include "yolo/yolo_detector.hpp"
#include "tools/logger.hpp"
#include "tools/math_utils.hpp"

/**
 * @brief Sigmoid 激活函数
 * @note  将网络输出的原始 logits 值映射到 (0, 1) 区间，代表概率。
 * inline 关键字确保该函数在编译时被直接展开，消灭函数调用压栈开销。
 */
inline float sigmoid(float x) 
{
    return 1.0f / (1.0f + std::exp(-x));
}

YoloDetector::YoloDetector(const std::string& model_path, bool use_gpu) 
{
    // 强制开启底层数学指令集优化 (唤醒 OpenBLAS 和 AVX2 等指令集)
    cv::setUseOptimized(true);

    LOG_DEBUG("[YOLO] 正在初始化 OpenCV DNN 模型...");

    // 检查模型文件在物理磁盘上是否存在，防止 cv::dnn 加载时产生段错误
    std::ifstream f(model_path.c_str());
    if (!f.good()) 
    {
        LOG_ERROR("[YOLO] 模型文件不存在: {}", model_path);
        return; 
    }

    try 
    {
        // 加载 ONNX 模型结构与权重
        net_ = cv::dnn::readNetFromONNX(model_path);

        // =======================================================
        // 【核心修改】：通过传参安全地切换 CPU 与 GPU 推理模式
        // =======================================================
        if (use_gpu) 
        {
            // 召唤 Intel OpenVINO 引擎，使用核显(OpenCL)并开启半精度(FP16)极速模式
            try 
            {
                // 尝试召唤 Intel OpenVINO 引擎
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE); 
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);        
                LOG_DEBUG("[YOLO] 引擎就绪: OpenVINO GPU (FP16 模式)");
            } 
            catch (const cv::Exception& e) 
            {
                // 如果当前环境缺失 OpenVINO 插件，警告并自动降级到 CPU
                LOG_WARN("[YOLO] OpenVINO GPU 插件不可用，自动降级为 CPU 模式! 原因: {}", e.what());
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); 
            }
        }
        else 
        {
            // 锁定纯 CPU 运算 (稳定，适合没有 OpenVINO 环境的设备)
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); 
            LOG_DEBUG("[YOLO] 引擎就绪: OpenCV CPU 模式");
        }

        is_ready_ = true; 

        LOG_DEBUG("[YOLO] 当前分配给 OpenCV 的线程数: {}", cv::getNumThreads());
    } 
    catch (const cv::Exception& e) 
    {
        LOG_ERROR("[YOLO] DNN 初始化异常: {}", e.what());
    }
}

std::vector<YoloObject> YoloDetector::Detect(const cv::Mat& raw_img, int enemy_color) 
{
    if (!is_ready_ || raw_img.empty()) return {};

    // 1. 图像预处理 (Blob 转换)
    // 将图像缩放至 640x640，并且做归一化(乘以 1/255)，同时完成 HWC 到 CHW 的通道重排
    cv::Mat blob = cv::dnn::blobFromImage(raw_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    
    // 计算缩放比例：用于将模型在 640x640 下预测的坐标，还原回相机真实分辨率（如 1280x1024）
    float scale_x = static_cast<float>(raw_img.cols) / 640.0f; 
    float scale_y = static_cast<float>(raw_img.rows) / 640.0f;

    // 2. 网络前向传播 (Forward)
    net_.setInput(blob);
    try 
    {
        net_.forward(net_outputs_, net_.getUnconnectedOutLayersNames());
    } 
    catch (const cv::Exception& e) 
    { 
        LOG_ERROR("[YOLO] 推理异常 (掉帧): {}", e.what());
        return {};
    }

    if (net_outputs_.empty()) return {};

    // 3. 维度处理与转置 (YOLOv8/11 输出格式兼容)
    cv::Mat output = net_outputs_[0];
    int rows = 0, cols = 0;
    float* data_ptr = (float*)output.data;

    // 解析网络输出的维度 (通常为 1 x 22 x 8400 或是 1 x 8400 x 22)
    if (output.dims == 3) 
    {
        rows = output.size[1];
        cols = output.size[2];
    } 
    else if (output.dims == 2) 
    {
        rows = output.size[0];
        cols = output.size[1];
    } 
    else 
    {
        return {};
    }

    cv::Mat output_buffer; 
    // 大多数新版 YOLO 模型输出格式为 [1, 通道数(22), 预测框数(8400)]，即 rows=22, cols=8400
    // 我们需要将其转置为 [8400, 22]，方便通过行指针连续遍历每个预测框
    if (rows < cols) 
    {
        cv::Mat temp(rows, cols, CV_32F, data_ptr); // 创建 Mat 头，不产生拷贝
        // 【极致优化】：复用 transposed_buffer_ 内存池，避免每帧 new 740KB 的内存
        cv::transpose(temp, transposed_buffer_); 
        output_buffer = transposed_buffer_; 
    } 
    else 
    {
        // 如果已经是 [8400, 22]，直接引用原始数据
        output_buffer = cv::Mat(rows, cols, CV_32F, data_ptr);
    }

    // 4. 清空上一帧的数据 (vector 的 clear 只移动指针，不释放内存，极快)
    class_ids.clear(); 
    color_ids.clear(); 
    confidences.clear(); 
    boxes.clear(); 
    all_keypoints.clear(); 

    // 获取内存首地址与单行的跨度
    float* data = (float*)output_buffer.data;
    int stride = output_buffer.cols; // 通常为 22

    // =========================================================================
    // 5. 【核心优化区】利用行指针遍历 8400 个预测框，拒绝一切 OpenCV 包装函数
    // =========================================================================
    for (int r = 0; r < output_buffer.rows; r++) 
    {
        float* row_ptr = data + r * stride; // 定位到第 r 个候选框的内存起始地址

        // A. 提取并校验置信度 (位于索引 8)
        float conf = sigmoid(row_ptr[8]);
        if (conf < score_threshold_) continue; // 快速淘汰大量背景框

        // B. 极速提取颜色类别 (位于索引 9-12)
        // 废弃原先耗时极大的 cv::minMaxLoc，改用循环查表，耗时减少 90%
        int raw_color = 0;
        float max_color_score = -1e6f;
        for (int i = 0; i < 4; ++i) 
        {
            if (row_ptr[9 + i] > max_color_score) 
            {
                max_color_score = row_ptr[9 + i];
                raw_color = i;
            }
        }

        // C. 底层硬过滤：如果是灰板(2)、紫板(3) 或者 不是当前指定的敌方颜色，直接抛弃！
        if (raw_color == 2 || raw_color == 3 || raw_color != enemy_color) continue;

        // D. 极速提取装甲板类别 (位于索引 13-21)
        int raw_class = 0;
        float max_class_score = -1e6f;
        for (int i = 0; i < 9; ++i) 
        {
            if (row_ptr[13 + i] > max_class_score)
            {
                max_class_score = row_ptr[13 + i];
                raw_class = i;
            }
        }

        // E. 提取 4 个物理角点 (位于索引 0-7)
        // 顺便计算角点的外接矩形 (min_x, max_x, min_y, max_y)，以供 NMS 剔除重叠框使用
        std::vector<cv::Point2f> armor_kpts(4);
        float min_x = 1e6f, max_x = -1e6f, min_y = 1e6f, max_y = -1e6f;
        for (int i = 0; i < 4; i++) 
        {
            float kx = row_ptr[i * 2] * scale_x;     // 乘以 scale_x 还原至原图尺度
            float ky = row_ptr[i * 2 + 1] * scale_y; // 乘以 scale_y 还原至原图尺度
            
            armor_kpts[i] = cv::Point2f(kx, ky);

            if (kx < min_x) min_x = kx;
            if (kx > max_x) max_x = kx;
            if (ky < min_y) min_y = ky;
            if (ky > max_y) max_y = ky;
        }

        cv::Rect box(static_cast<int>(min_x), static_cast<int>(min_y), 
                     static_cast<int>(max_x - min_x), static_cast<int>(max_y - min_y));

        // 保存优质的候选数据
        class_ids.push_back(raw_class);
        color_ids.push_back(raw_color);
        confidences.push_back(conf);
        boxes.push_back(box);
        all_keypoints.push_back(armor_kpts);
    }

    // 6. NMS (非极大值抑制)
    // 同一个装甲板可能被模型输出多个位置相近的预测框，NMS 会保留置信度最高的那一个
    indices.clear();
    if (!boxes.empty()) 
    {
        cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices); 
    }

    // 7. 组装最终输出 (根据 NMS 筛选出的有效索引)
    std::vector<YoloObject> final_objects;
    for (int idx : indices) 
    {
        YoloObject obj;
        obj.prob = confidences[idx];
        obj.box = boxes[idx];
        obj.pts = all_keypoints[idx]; // PnP 解算真正依赖的核心数据！
        obj.color = color_ids[idx];

        int raw_class = class_ids[idx];
        // 深大模型特定的映射表：将模型 class id 映射到现实装甲板的装甲号
        int number_map[] = {6, 1, 2, 3, 4, 5, 7, 8, 8}; 
        
        if(raw_class >= 0 && raw_class <= 8) 
        {
            obj.number = number_map[raw_class];
            // 英雄(1) 和 基地(8) 采用 225mm 大装甲板，其余均为 135mm 小板
            obj.is_big = (raw_class == 1 || raw_class == 8); 
        } 
        else 
        {
            obj.number = 0;
            obj.is_big = false;
        }
        
        final_objects.push_back(obj);
    }

    return final_objects;
}