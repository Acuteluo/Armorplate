#ifndef ARMOR_CORNER_REFINER_HPP
#define ARMOR_CORNER_REFINER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class ArmorCornerRefiner 
{
public:
    ArmorCornerRefiner();
    ~ArmorCornerRefiner() = default;

    /**
     * @brief 核心接口：利用图像一维梯度极值与 RANSAC 将 YOLO 角点精化至亚像素级
     * @param bgr_img     原始 BGR 彩色图像
     * @param corners     YOLO 传出的 4 个粗糙角点 (将在函数内被原地修改为精化后的坐标)
     * @param armor_color 敌方装甲板颜色 (用于通道分离，提取最锐利边缘)
     * @return true 精化成功, false 精化失败(退化为使用原 YOLO 角点)
     */
    bool Refine(const cv::Mat& bgr_img, std::vector<cv::Point2f>& corners, int armor_color);

private:
    // --- 算法核心子步骤 ---

    // 1. 边缘中点提取与 RANSAC 拟合
    // 原理：在两个粗端点连线上均匀取样，沿法线扫描提取左右边缘的亚像素位置，
    //       求得一系列高精度“灯条中点”，最后 RANSAC 拟合出精确的中轴线。
    bool GetSubPixelCenterLine(const cv::Mat& gray_img, 
                               const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                               cv::Vec4f& exact_line, float dynamic_W);

    // 2. 端点一维滑动搜索
    // 原理：沿着求得的精确中轴线滑动窗口，构造一维横向平均亮度信号，寻找梯度跳变极值点。
    cv::Point2f SearchExactEndpoint(const cv::Mat& gray_img, 
                                    const cv::Vec4f& exact_line, 
                                    const cv::Point2f& rough_pt, 
                                    bool is_top_point);

    // --- 纯数学工具函数 ---

    // 双线性插值：提取浮点数坐标的亚像素灰度值 (消除整型坐标带来的阶梯误差)
    float BilinearInterpolation(const cv::Mat& img, float x, float y);

    // 三点抛物线插值：基于离散梯度的最大值和左右两点，解析求出二次函数的极值亚像素偏移
    float ParabolicInterpolation(float y_minus_1, float y_0, float y_plus_1);

private:
    // 算法超参数配置
    int sample_points_K_;     // 扫描线均匀取样点数 K (用于提取边缘)
    float scan_half_width_W_; // 法线扫描默认半宽 W (像素)
    int endpoint_search_R_;   // 端点沿中轴线滑动搜索范围 R (像素)
};

#endif // ARMOR_CORNER_REFINER_HPP