#ifndef YOLO_ARMOR_HPP
#define YOLO_ARMOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// YOLO 专属装甲板类，负责 3D 姿态解算 (PnP) 与坐标系转换
class YoloArmor 
{
public:

    YoloArmor();

    YoloArmor(int armor_id, int color, bool is_big, float confidence, const cv::Rect& box, const std::vector<cv::Point2f>& corners, const cv::Mat& K, const cv::Mat& D);

    ~YoloArmor() = default;

    // --- Yolo 得到的装甲板的基础属性 ---
    int armor_id_;                     // 编号 (1~8)
    int color_;                        // 颜色 (0:红, 1:蓝, 2:灰, 3:紫)
    bool is_big_;                      // 大小板标志 (true: 大板, false: 小板)
    float confidence_;                 // 置信度
    cv::Rect box_;                     // 2D 边界框
    std::vector<cv::Point2f> corners_; // 2D 像素角点 (左上, 左下, 右下, 右上)

    // --- 姿态与位置 (FLU 统一右手系！) ---
    bool pnp_success_ = false;         // PnP 解算是否成功
    Eigen::Vector3d t_vec_;            // 最终给 EKF 的平移向量，在 FLU 系下 (x向前, y向左, z向上)
    Eigen::Matrix3d R_;                // 最终给 EKF 的旋转矩阵，在 FLU 系下 经过优化后的 
    
    double t_distance_;                // 距离相机中心距离 (米)
    double obs_yaw_angle_;             // 视线偏差角 目标相对于相机的偏航角 (度)
    double obs_pitch_angle_;           // 视线偏差角 目标相对于相机的俯仰角 (度)

    double euler_yaw_angle_ = 0.00;    // 优化得到的最好的 欧拉角 yaw（度）

    // --- 相机内参 ---
    cv::Mat K_;                        // 内参矩阵
    cv::Mat D_;                        // 畸变系数

    // --- 核心方法 ---
    // 根据 YOLO 判断的大/小板设置装甲板四个角点的坐标，并将 FLU 系的点转换到 OpenCV 系下供 PnP 使用
    void SetArmorplateSize();

    // 执行 PnP 解算，并完成逆向基变换 (OpenCV -> FLU)，自动调用 OptimizeEulerYaw()
    void PNP();

    // 在图像上画框并打印信息，模式 "simple" / "complex" 仅输出框 / 打印所有信息
    void DrawAndPrintInfo(cv::Mat& img_show, std::string mode_select);

    // 打印调试日志
    void PrintDebugLog(bool is_debug);

private:

    // 优化 PnP 解算的欧拉角
    void OptimizeEulerYaw();

    // 进行 区间搜索 以找到最优的欧拉角 yaw
    double SearchOptimalEulerYaw(double initial_euler_yaw, double search_range, double search_step);

    // 计算给定的 欧拉角yaw（弧度） 和 pitch=+15°、roll=0°时，在图上的重投影误差
    double CalculateReprojectionError(double test_euler_yaw);

    std::vector<cv::Point3f> vertice_world_; // 装甲板在 FLU 系下的 3D 物理点（定义点）
    std::vector<cv::Point3f> vertice_cv_;    // 转换到 OpenCV 相机系下的 3D 物理点（定义点）
    cv::Mat r_cv_, t_cv_;                    // OpenCV 算出来的原始旋转平移向量
    Eigen::Matrix3d P_;                      // 坐标系转换矩阵 (FLU -> OpenCV)

    bool show_logger_pnp_ = false;           // 是否打印 调试日志
};

#endif // YOLO_ARMOR_HPP