#include "armor_corner_refiner.hpp"
#include "tools/logger.hpp"
#include "tools/math_utils.hpp" // 引入手写的 fitLineRANSAC
#include <cmath>
#include <algorithm>

ArmorCornerRefiner::ArmorCornerRefiner() 
{
    // 配置超参数
    sample_points_K_ = 15;     // 扫点数：均匀取 15 个扫描截面
    scan_half_width_W_ = 8.0f; // 扫描线：默认扫描半宽 8 像素
    endpoint_search_R_ = 6;    // 断点处：端点搜索范围 ±6 像素
}


bool ArmorCornerRefiner::Refine(const cv::Mat& bgr_img, std::vector<cv::Point2f>& corners, int armor_color)
{
    // 检测到了装甲板，而且装甲板是四个角点
    if (bgr_img.empty() || corners.size() != 4) 
    {
        return false;
    }

    // 1. 颜色通道分离抗过曝
    cv::Mat single_color_img;
    std::vector<cv::Mat> channels;
    cv::split(bgr_img, channels);

    // 目标是蓝方
    if (armor_color == 1) 
    {
        // 蓝灯条，使用蓝色通道提取
        single_color_img = channels[0]; 
    }
    // 目标是红方
    else 
    {
        // 红灯条，使用红色通道提取
        single_color_img = channels[2]; 
    }

    // 2.【自适应计算扫描线左右半宽 W】
    // 为了防止扫到另一根灯条，又为了防止扫不到边界，动态限制扫描宽度
    float top_width = (float)cv::norm(corners[3] - corners[0]);    // 右上 - 左上
    float bottom_width = (float)cv::norm(corners[2] - corners[1]); // 右下 - 左下
    float armor_average_width = (top_width + bottom_width) / 2.0f;
    // 严格限制扫描宽度，绝不超过装甲板宽度的 75%
    float dynamic_W = std::min(scan_half_width_W_, armor_average_width * 0.75f);


    // 3. 对【左灯条】进行精化流水线处理
    cv::Vec4f left_line;

    // 先进行扫描线，找到左右边缘亚像素点，然后合成中轴线，毕竟是等效的
    bool is_left_ok = GetSubPixelCenterLine(single_color_img, corners[0], corners[1], left_line, dynamic_W);
    if (is_left_ok) 
    {
        // 沿中轴线滑窗，寻找上、下明暗跳变亚像素点
        corners[0] = SearchExactEndpoint(single_color_img, left_line, corners[0], true);  // 左上
        corners[1] = SearchExactEndpoint(single_color_img, left_line, corners[1], false); // 左下
    }

    // 4. 对【右灯条】进行精化流水线处理
    cv::Vec4f right_line;
    bool is_right_ok = GetSubPixelCenterLine(single_color_img, corners[3], corners[2], right_line, dynamic_W);
    if (is_right_ok) 
    {
        // 沿中轴线滑窗，寻找上、下明暗跳变亚像素点
        corners[3] = SearchExactEndpoint(single_color_img, right_line, corners[3], true);  // 右上
        corners[2] = SearchExactEndpoint(single_color_img, right_line, corners[2], false); // 右下
    }

    // 只要有任何一边精化成功，都算优化有价值
    if(is_left_ok || is_right_ok)
    {
        return true;
    }
    else return false;
}



bool ArmorCornerRefiner::GetSubPixelCenterLine(const cv::Mat& gray_img, 
                                               const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                                               cv::Vec4f& exact_line, float dynamic_W)
{
    // 计算 Yolo 初值的粗略方向向量 v（从上指向下）和法线向量 n（从左指向右）
    cv::Point2f v = pt_bottom - pt_top;
    float length = (float)cv::norm(v);
    if (length < 1.0f) // 上下端点距离过小，无法进行精化
    {
        return false;
    }

    cv::Point2f s = v / (sample_points_K_ - 1); // 把上下端点的连线分成k-1段，每一段的小向量（从上指向下）
    cv::Point2f u = v / length;                 // 单位像素长度的轴向量（从上指向下）
    cv::Point2f n = cv::Point2f(-u.y, u.x);     // 逆时针旋转 90 度得到单位像素法向量（从左指向右）

    // 存储中轴线的点序列
    std::vector<cv::Point2f> pts_center;

    // 沿长轴均匀取样的第 k 个点 (从 1 开始算。去掉首尾端点，所以是从第 2 个点到第 K - 1 个点)
    for (int k = 2; k < sample_points_K_; k++) 
    {
        cv::Point2f k_initial_center = pt_top + (k - 1) * s; // 初始连线的第 k 个中心点

        // 1. 沿法向生成一维离散信号，宽度为 2W，每个间隔 1 像素
        std::vector<float> profile(int(dynamic_W * 2 + 1), 0.0f);
        
        for (int offset = -int(dynamic_W); offset <= int(dynamic_W); offset++) 
        {
            cv::Point2f p = k_initial_center + offset * n;
            // 必须使用双线性插值，否则直接取整会破坏亚像素梯度的平滑性！
            profile[offset + dynamic_W] = BilinearInterpolation(gray_img, p.x, p.y);
        }

        // 2. 使用中心差分 [-1, 0, 1] 计算一维梯度序列
        // f'(x) ≈ (f(x+h) - f(x-h)) / (2h)，这里 /2 可以省略，毕竟都一样
        std::vector<float> grad(int(dynamic_W * 2 + 1), 0.0f);
        float max_pos_grad = 0.0f; int pos_idx = -1; // 寻找暗->亮跳变(左边缘)，最大正梯度
        float max_neg_grad = 0.0f; int neg_idx = -1; // 寻找亮->暗跳变(右边缘)，最大负梯度

        for (int i = 1; i < int(dynamic_W * 2); i++) 
        {
            grad[i] = profile[i+1] - profile[i-1]; // 中心差分
            
            if (grad[i] > max_pos_grad) 
            { 
                max_pos_grad = grad[i]; 
                pos_idx = i; 
            }
            if (grad[i] < max_neg_grad) 
            { 
                max_neg_grad = grad[i]; 
                neg_idx = i; 
            }
        }

        // 3. 双峰逻辑校验：必须同时找到明显的正峰和负峰
        // 动态阈值：梯度必须足够大才认为是真实的边界，并且正负梯度都是
        float thresh = std::max(std::abs(max_pos_grad), std::abs(max_neg_grad)) * 0.2f;
        if (pos_idx != -1 && neg_idx != -1 && max_pos_grad > thresh && std::abs(max_neg_grad) > thresh)
        {
            // 校验两峰间距 (防止过宽扫到外侧噪声，过窄提取到局部噪点)
            int dist = std::abs(pos_idx - neg_idx);
            if (dist >= 2 && dist <= dynamic_W * 1.5)
            {
                // 4. 核心数学：三点抛物线插值寻找亚像素真实极值点
                float sub_pos_i = pos_idx + ParabolicInterpolation(grad[pos_idx-1], grad[pos_idx], grad[pos_idx+1]);
                float sub_neg_i = neg_idx + ParabolicInterpolation(grad[neg_idx-1], grad[neg_idx], grad[neg_idx+1]);

                // 左右亚像素边缘的中点，即为灯条中轴线在该截面上的极高精度落点
                float sub_center_i = (sub_pos_i + sub_neg_i) / 2.0f;
                float center_offset = sub_center_i - dynamic_W;

                // 映射回 2D 全局图像坐标系
                cv::Point2f center_pt = k_initial_center + center_offset * n;
                pts_center.push_back(center_pt);
            }
        }
    }

    // 如果有效中点太少，说明被严重遮挡或光照摧毁，拒绝拟合
    if (pts_center.size() < 4) 
    {
        return false;
    }

    // 5. 将精挑细选的亚像素中点阵列，喂给手写的 RANSAC+PCA 最小二乘算法，这一步彻底免疫了残缺发光点导致的质心偏移！
    return tools::fitLineRANSAC(pts_center, exact_line, 80, 1.2f);
}

cv::Point2f ArmorCornerRefiner::SearchExactEndpoint(const cv::Mat& gray_img, 
                                                    const cv::Vec4f& exact_line, 
                                                    const cv::Point2f& rough_pt, 
                                                    bool is_top_point)
{
    // 1. 将粗端点投影到已经拟合好的纯净中轴线上，作为搜索基准
    cv::Point2f v(exact_line[0], exact_line[1]); // 直线单位方向向量
    cv::Point2f p0(exact_line[2], exact_line[3]);// 直线上基准点
    
    // 向量点乘求投影参数 t
    float t_proj = (rough_pt.x - p0.x) * v.x + (rough_pt.y - p0.y) * v.y;
    cv::Point2f proj_pt = p0 + t_proj * v;

    // 规范方向：强行让 v 指向下方 (Y增大方向)
    if (v.y < 0) v = -v; 
    cv::Point2f n(-v.y, v.x); // 法向，用于横向提取小带状区域

    // 2. 构建一维滑动窗口平均亮度信号
    int search_len = endpoint_search_R_ * 2 + 1;
    std::vector<float> signal(search_len, 0.0f);
    
    for (int i = 0; i < search_len; ++i)
    {
        // 沿中轴线方向步进，步长 1 像素 (从 -R 到 +R)
        float step = (i - endpoint_search_R_) * 1.0f; 
        cv::Point2f curr_p = proj_pt + step * v;

        // 横向平均提取法 (Width=3)：大幅抑制字符涂装带来的噪点，保留端点阶跃性
        float sum_val = 0.0f;
        sum_val += BilinearInterpolation(gray_img, curr_p.x - 1.0f * n.x, curr_p.y - 1.0f * n.y);
        sum_val += BilinearInterpolation(gray_img, curr_p.x,              curr_p.y);
        sum_val += BilinearInterpolation(gray_img, curr_p.x + 1.0f * n.x, curr_p.y + 1.0f * n.y);
        
        signal[i] = sum_val / 3.0f;
    }


    // 3：计算跳变梯度并获取全局最大幅度，用于计算动态阈值
    std::vector<float> grad(search_len, 0.0f);
    float max_abs_grad = 0.0f;

    for (int i = 1; i < search_len - 1; i++)
    {
        grad[i] = signal[i+1] - signal[i-1];
        max_abs_grad = std::max(max_abs_grad, std::abs(grad[i]));
    }

    // 动态阈值：必须大于最大梯度的 25%，且保底为 8.0f 过滤底噪
    float thresh = std::max(8.0f, max_abs_grad * 0.25f);
    int target_idx = -1;

    // 由外向内寻找第一峰, 解决过曝导致的“白-黄-黑”中心偏移问题！
    if (is_top_point) 
    {
        // 上端点：从外(上方, 小索引)向内(下方, 大索引)找，寻找第一个符合条件的【正梯度峰值】
        for (int i = 1; i < search_len - 1; i++) 
        {
            if (grad[i] > thresh && grad[i] >= grad[i-1] && grad[i] >= grad[i+1]) 
            {
                target_idx = i;
                break; // 找到最外侧的真实边缘，直接退出！无视内部可能更大的白斑梯度！
            }
        }
    }
    else 
    {
        // 下端点：从外(下方, 大索引)向内(上方, 小索引)找，寻找第一个符合条件的【负梯度峰值】
        // 注意：因为外侧在大索引，所以必须【倒序遍历】！
        for (int i = search_len - 2; i >= 1; i--) 
        {
            if (grad[i] < -thresh && grad[i] <= grad[i-1] && grad[i] <= grad[i+1]) 
            {
                target_idx = i;
                break; // 找到最外侧的真实边缘，直接退出！
            }
        }
    }


    // 4. 重复利用数学工具：抛物线亚像素插值
    // 虽然物理意义从“扫描宽度截面”变成了“沿轴线长度切面”，
    // 但数学本质都是求解一维离散信号梯度极值的精确位置！
    float sub_offset = 0.0f;
    // 设置跳变强度的最小容忍阈值，防止把微弱的噪点波动当成端点
    if (target_idx != -1) 
    {
        sub_offset = ParabolicInterpolation(grad[target_idx-1], grad[target_idx], grad[target_idx+1]);
    }
    else
    {
        // 如果梯度太弱 (不到 10)，说明扫到了噪点，或者灯条被截断了。
        // 此时必须强制把索引复位！否则点会带着错误的 target_idx 飞走 6 个像素！
        target_idx = endpoint_search_R_; 
        sub_offset = 0.0f;
    }

    // 将亚像素步进映射回最终坐标系
    float final_step = (target_idx + sub_offset - endpoint_search_R_) * 1.0f;
    return proj_pt + final_step * v;
}

float ArmorCornerRefiner::BilinearInterpolation(const cv::Mat& img, float x, float y)
{
    // 防止访问越界引发段错误
    if (x < 0.0f || x >= (img.cols - 1) || y < 0.0f || y >= (img.rows - 1)) 
    {
        return 0.0f;
    }

    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix; // 小数部分
    float dy = y - iy;

    // 提取周围四个整型像素块的值
    float v00 = img.at<uchar>(iy, ix);
    float v10 = img.at<uchar>(iy, ix + 1);
    float v01 = img.at<uchar>(iy + 1, ix);
    float v11 = img.at<uchar>(iy + 1, ix + 1);

    // 标准双线性面积权重求和
    return (1.0f - dx) * (1.0f - dy) * v00 + 
           dx * (1.0f - dy) * v10 + 
           (1.0f - dx) * dy * v01 + 
           dx * dy * v11;
}

float ArmorCornerRefiner::ParabolicInterpolation(float y_minus_1, float y_0, float y_plus_1)
{
    // 设抛物线经过 (-1, y_minus_1), (0, y_0), (1, y_plus_1)
    // 根据顶点公式 x = -b/2a 推导出的亚像素偏移量
    float denominator = y_minus_1 - 2.0f * y_0 + y_plus_1;
    if (std::abs(denominator) < 1e-5f) return 0.0f; // 规避极平缓导致的除零崩溃

    return 0.5f * (y_minus_1 - y_plus_1) / denominator;
}