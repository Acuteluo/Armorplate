#include "tools/math_utils.hpp"
#include <random>
#include <cmath>
#include <algorithm>

namespace tools
{
int fitLineRANSAC(const std::vector<cv::Point2f>& points, cv::Vec4f& line, int iterations, float distance_thresh)
{
    // ransac 确保点数足够
    if (points.size() < 2) 
    {
        return 0; // 返回 0 表示失败
    }

    // 如果只有两个点，直接连线即可
    if (points.size() == 2) 
    {
        cv::Point2f v = points[1] - points[0];
        float length = (float)cv::norm(v);
        if (length < 1e-5) // 线段太短
        {
            return 0;
        }
        line = cv::Vec4f(v.x / length, v.y / length, points[0].x, points[0].y);
        return 2; // 两个点都是内点，权重为 2
    }

    // 1. 初始化随机数发生器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    int best_inlier_count = 0;
    std::vector<cv::Point2f> best_inliers;

    // 2. RANSAC 迭代寻找最优内点集合
    for (int i = 0; i < iterations; ++i) 
    {
        // 随机抽取两个不同的点的索引值
        int index1 = dis(gen);
        int index2 = dis(gen);
        if (index1 == index2) continue;

        cv::Point2f p1 = points[index1];
        cv::Point2f p2 = points[index2];

        // 构造两点确定的直线方程 Ax + By + C = 0
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float length = std::hypot(dx, dy);
        if (length < 1e-5) continue; // 防止两点重合

        // 归一化方向向量的法向量作为 A 和 B
        float A = -dy / length;
        float B =  dx / length;
        float C = -(A * p1.x + B * p1.y);

        // 统计内点
        int inlier_count = 0;
        std::vector<cv::Point2f> current_inliers;
        for (int j = 0; j < points.size(); j++) 
        {
            // 计算该点到直线的垂直距离 d = |Ax + By + C|
            float dist = std::abs(A * points[j].x + B * points[j].y + C);

            // 小于阈值，属于内点
            if (dist < distance_thresh) 
            {
                ++inlier_count; 
                current_inliers.push_back(points[j]); // 放入内点集合
            }
        }

        // 更新最佳拟合结果
        if (inlier_count > best_inlier_count) 
        {
            best_inlier_count = inlier_count;
            best_inliers = current_inliers;
        }
    }

    // 内点太少，说明拟合失败
    if (best_inlier_count < 2) 
    {
        return 0;
    }


    // =======================================================================
    // 3. 终极精化：手撕 PCA (主成分分析) 对最纯净的内点集合进行最小二乘拟合
    // =======================================================================
    
    // (A) 计算内点几何中心 (质心)
    cv::Point2f centroid(0.0f, 0.0f);
    for (int i = 0; i < best_inliers.size(); i++) 
    {
        centroid.x += best_inliers[i].x;
        centroid.y += best_inliers[i].y;
    }
    centroid.x /= best_inlier_count;
    centroid.y /= best_inlier_count;

    // (B) 构建 2x2 协方差矩阵 
    float Cxx = 0.0f, Cxy = 0.0f, Cyy = 0.0f;
    for (int i = 0; i < best_inliers.size(); i++) 
    {
        float dx = best_inliers[i].x - centroid.x;
        float dy = best_inliers[i].y - centroid.y;
        Cxx += dx * dx;
        Cxy += dx * dy;
        Cyy += dy * dy;
    }
    
    // (C) 求解协方差矩阵的最大特征向量 (即主成分方向)
    // 根据瑞利商定理，2x2 矩阵最大特征向量的角度可由 atan2 直接求出
    // theta = 0.5 * atan2(2 * Cxy, Cxx - Cyy)
    float theta = 0.5f * std::atan2(2.0f * Cxy, Cxx - Cyy);
    
    // 生成单位方向向量 (vx, vy)
    float vx = std::cos(theta);
    float vy = std::sin(theta);

    // 将直线结果打包输出 [vx, vy, x0, y0]
    line = cv::Vec4f(vx, vy, centroid.x, centroid.y);

    // 返回寻找到的绝对内点数量，作为后续加权的物理置信度
    return best_inlier_count;
}

} // namespace tools