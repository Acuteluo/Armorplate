#ifndef ARMOR_CORNER_REFINER_HPP
#define ARMOR_CORNER_REFINER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class ArmorCornerRefiner 
{
public:
    ArmorCornerRefiner();
    ~ArmorCornerRefiner() = default;

    // ====================================================================
    // 【新增】：用于向外部可视化提供每一层的提取点 (左边缘、右边缘、配对中点)
    // ====================================================================
    std::vector<cv::Point2f> debug_l_left_edges_, debug_l_right_edges_, debug_l_centers_;
    std::vector<cv::Point2f> debug_r_left_edges_, debug_r_right_edges_, debug_r_centers_;

    /**
     * @brief 核心接口：利用图像一维梯度极值与 RANSAC 将 YOLO 角点精化至亚像素级
     * @param bgr_img     原始 BGR 彩色图像
     * @param corners     YOLO 传出的 4 个粗糙角点 (将在函数内被原地修改为精化后的坐标)
     * @param enemy_armor_color  敌方装甲板颜色 (用于通道分离，提取最锐利边缘)
     * @param left_middle_line   左灯条中轴线拟合的直线参数
     * @param right_middle_line  右灯条中轴线拟合的直线参数
     * @return true 精化成功, false 精化失败(退化为使用原 YOLO 角点)
     */
    bool Refine(const cv::Mat& bgr_img, std::vector<cv::Point2f>& corners, int enemy_armor_color, cv::Vec4f& left_middle_line, cv::Vec4f& right_middle_line);


    /**
     * @brief 获取迭代后存储的 6 组 debug 提取点
     * @return std::vector<std::vector<cv::Point2f>> 迭代后左灯条左、右、中点集合 + 右灯条左、右、中集合
     */
    std::vector<std::vector<cv::Point2f>> GetDebugPoints();

private:
    // --- 算法核心子步骤 ---

    // 对于某个灯条，获取最终优化后灯条中轴线的过程函数，内部调用两次 ScanAndFit 实现迭代纠正
    bool GetSubPixelCenterLine(const cv::Mat& gray_img, 
                               const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                               cv::Vec4f& exact_line, float dynamic_W,
                               std::vector<cv::Point2f>& out_left_edges,
                               std::vector<cv::Point2f>& out_right_edges,
                               std::vector<cv::Point2f>& out_centers);
    
    
    // 对于某个灯条，执行具体扫描线 寻找双峰 和 RANSAC+PCA 优化得到中轴线的函数
    bool ScanAndFit(const cv::Mat& gray_img, 
                    const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                    cv::Vec4f& exact_line, float dynamic_W,
                    std::vector<cv::Point2f>& out_left_edges,
                    std::vector<cv::Point2f>& out_right_edges,
                    std::vector<cv::Point2f>& out_centers);


    // 对于某个灯条 上 / 下 端点，优化端点位置的函数
    cv::Point2f SearchExactEndpoint(const cv::Mat& gray_img,
                                    const cv::Vec4f& exact_line, 
                                    const cv::Point2f& rough_pt, 
                                    bool is_top_point,
                                    float bar_length);

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