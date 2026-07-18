#ifndef TOOLS_MATH_UTILS_HPP
#define TOOLS_MATH_UTILS_HPP

#include <cmath>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace tools
{

/**
 * @brief 弧度转角度
 */
inline double rad2deg(double rad) 
{ 
    return rad * 180.0 / M_PI; 
}

/**
 * @brief 角度转弧度
 */
inline double deg2rad(double deg) 
{ 
    return deg * M_PI / 180.0; 
}

/**
 * @brief 将弧度限制在 [-PI, PI] 之间 (就地修改)
 */
inline void limit_rad_inplace(double& rad) 
{
    while (rad > M_PI) rad -= 2.0 * M_PI;
    while (rad < -M_PI) rad += 2.0 * M_PI;
}

/**
 * @brief 将弧度限制在 [-PI, PI] 之间 (返回副本)
 */
inline double limit_rad(double rad) 
{
    limit_rad_inplace(rad);
    return rad;
}

/**
 * @brief 手写 RANSAC 鲁棒直线拟合 (结合 PCA 最小二乘精化)
 * @param points           输入的 2D 亚像素点集
 * @param line             输出的直线参数，格式为 [vx, vy, x0, y0] (vx vy 单位方向向量，x0 y0 线上一点)
 * @param iterations       RANSAC 最大迭代次数
 * @param distance_thresh  判定为内点的最大距离阈值 (像素)
 * @return true 拟合成功, false 拟合失败(有效点数过少)
 */
bool fitLineRANSAC(const std::vector<cv::Point2f>& points, cv::Vec4f& line, 
                   int iterations = 100, float distance_thresh = 1.0f);

} // namespace tools

#endif // TOOLS_MATH_UTILS_HPP