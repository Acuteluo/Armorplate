#include "armor_corner_refiner.hpp"
#include "tools/logger.hpp"
#include "tools/math_utils.hpp" // 引入手写的 fitLineRANSAC
#include <cmath>
#include <algorithm>

ArmorCornerRefiner::ArmorCornerRefiner() 
{
    // 配置超参数
    sample_points_K_ = 20;         // 扫点数：均匀取 15 个扫描截面
    scan_half_width_W_ = 8.0f;     // 扫描线：默认扫描半宽 8 像素
    endpoint_search_R_ = 12.0f;    // 断点处：端点搜索范围 ±12 像素，实际会在函数内动态防重叠
}


// 优化两侧灯条的端点的 主逻辑 过程函数
bool ArmorCornerRefiner::Refine(const cv::Mat& bgr_img, std::vector<cv::Point2f>& corners, int enemy_armor_color, cv::Vec4f& left_middle_line, cv::Vec4f& right_middle_line)
{
    // 检测到了装甲板，而且装甲板是四个角点
    if (bgr_img.empty() || corners.size() != 4) 
    {
        return false;
    }
    // 修改 

    // 1. 颜色通道分离抗过曝
    cv::Mat single_color_img;
    std::vector<cv::Mat> channels;
    cv::split(bgr_img, channels);

    // 目标是蓝方
    if (enemy_armor_color == 1) 
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


    // 2. 自适应计算扫描线左右半宽 W
    // 为了防止扫到另一根灯条，又为了防止扫不到边界，动态限制扫描宽度
    float top_width = (float)cv::norm(corners[3] - corners[0]);    // 右上 - 左上
    float bottom_width = (float)cv::norm(corners[2] - corners[1]); // 右下 - 左下
    float armor_average_width = (top_width + bottom_width) / 2.0f;
    // 严格限制扫描宽度，绝不超过装甲板宽度的 75%
    float dynamic_W = std::min(scan_half_width_W_, armor_average_width * 0.75f);

    // 清空用于可视化的调试容器
    debug_l_left_edges_.clear(); debug_l_right_edges_.clear(); debug_l_centers_.clear();
    debug_r_left_edges_.clear(); debug_r_right_edges_.clear(); debug_r_centers_.clear();


    // 3. 对【左灯条】进行精化流水线处理
    // 先进行扫描线，找到左右边缘亚像素点，然后合成中轴线，毕竟中轴线是灯条的最优平分线
    bool is_left_ok = GetSubPixelCenterLine(single_color_img, corners[0], corners[1], left_middle_line, dynamic_W,
                                            debug_l_left_edges_, debug_l_right_edges_, debug_l_centers_);
    if (is_left_ok) 
    {
        float left_length = (float)cv::norm(corners[1] - corners[0]);
        // 沿中轴线滑窗，寻找上、下明暗跳变亚像素点
        corners[0] = SearchExactEndpoint(single_color_img, left_middle_line, corners[0], true, left_length);  // 左上
        corners[1] = SearchExactEndpoint(single_color_img, left_middle_line, corners[1], false, left_length); // 左下
    }

    // 4. 对【右灯条】进行精化流水线处理
    bool is_right_ok = GetSubPixelCenterLine(single_color_img, corners[3], corners[2], right_middle_line, dynamic_W,
                                             debug_r_left_edges_, debug_r_right_edges_, debug_r_centers_);
    if (is_right_ok) 
    {
        float right_length = (float)cv::norm(corners[2] - corners[3]);
        // 沿中轴线滑窗，寻找上、下明暗跳变亚像素点
        corners[3] = SearchExactEndpoint(single_color_img, right_middle_line, corners[3], true, right_length);  // 右上
        corners[2] = SearchExactEndpoint(single_color_img, right_middle_line, corners[2], false, right_length); // 右下
    }

    // 只要有任何一边精化成功，都算优化有价值
    if(is_left_ok || is_right_ok) return true;
    else return false;
}


// 对于某个灯条，获取最终优化后灯条中轴线的过程函数
bool ArmorCornerRefiner::GetSubPixelCenterLine(const cv::Mat& gray_img, 
                                               const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                                               cv::Vec4f& exact_line, float dynamic_W,
                                               std::vector<cv::Point2f>& out_left_edges,
                                               std::vector<cv::Point2f>& out_right_edges,
                                               std::vector<cv::Point2f>& out_centers)
{
    cv::Vec4f L0;
    // 第一次扫描所需的哑容器 (不显示第一次扫描的歪点)
    std::vector<cv::Point2f> dummy_l, dummy_r, dummy_c;

    // 初始化：使用 YOLO 粗糙端点连线作为法向，进行初步扫描
    if (!ScanAndFit(gray_img, pt_top, pt_bottom, L0, dynamic_W, dummy_l, dummy_r, dummy_c)) 
    {
        return false;
    }

    // 用第一次拟合出的 L0 提取绝对纯净的方向向量，修正扫描截面！
    cv::Point2f direction(L0[0], L0[1]);
    if (direction.y < 0) direction = -direction; // 保证方向从上到下
    cv::Point2f p0(L0[2], L0[3]); // 灯条中心点

    // 将原始的 YOLO 端点投影到修正后的中轴线上，界定新的扫描起止区间
    float t_top = (pt_top - p0).dot(direction); // 投影
    float t_bottom = (pt_bottom - p0).dot(direction);
    cv::Point2f new_top = p0 + t_top * direction;
    cv::Point2f new_bottom = p0 + t_bottom * direction;

    // 【迭代 1】：利用绝对垂直的法向重新扫描，彻底抵消左右不对称位移
    // 此时将扫描出的极高精度点送入外部可视化容器 out_*
    return ScanAndFit(gray_img, new_top, new_bottom, exact_line, dynamic_W, 
                      out_left_edges, out_right_edges, out_centers);
}



// 对于某个灯条，执行具体扫描线 寻找双峰 和 RANSAC+PCA 优化得到中轴线的函数
bool ArmorCornerRefiner::ScanAndFit(const cv::Mat& gray_img, 
                                    const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                                    cv::Vec4f& exact_line, float dynamic_W,
                                    std::vector<cv::Point2f>& out_left_edges,
                                    std::vector<cv::Point2f>& out_right_edges,
                                    std::vector<cv::Point2f>& out_centers)
{
    // 计算 Yolo 初值的粗略方向向量 v（从上指向下）和法线向量 n（从左指向右）
    cv::Point2f v = pt_bottom - pt_top;
    float length = (float)cv::norm(v);
    if (length < 1.0f) // 上下端点距离过小，无法进行精化
    {
        return false;
    }

    cv::Point2f s = v / (sample_points_K_ - 1); // 把上下端点的连线分成 k - 1 段，每一段的小向量（从上指向下）
    cv::Point2f u = v / length;                 // 单位像素长度的轴向量（从上指向下）
    cv::Point2f n = cv::Point2f(u.y, -u.x);     // 单位像素法向量（从左指向右）

    // 对 W 取整，确认扫描宽度 profile_length
    int W_int = (int)dynamic_W;
    int profile_length = W_int * 2 + 1;
    std::vector<cv::Point2f> center_pts; // 存放所有中轴线中点

    // 沿长轴均匀取样的第 k 个点 (从 1 开始算。去掉首尾端点，所以是从第 2 个点到第 K - 1 个点)
    for (int k = 2; k < sample_points_K_; k++) 
    {
        cv::Point2f k_initial_center = pt_top + (k - 1) * s; // 初始连线的第 k 个扫描起点（扫描线中点）
        
        // 1. 沿法向生成一维离散信号，宽度为 2W，每个间隔 1 像素
        std::vector<float> profile(profile_length, 0.0f);

        // 1. 找到剖面的最小、最大值，也要记录极值索引
        float min_val = 255.0f;
        float max_val = 0.0f;
        int max_idx = -1;
        
        std::cout << std::endl << "第 " << k << " 个点" << std::endl << "一维离散信号：" << std::endl;
        for (int offset = -W_int; offset <= W_int; offset++) 
        {
            cv::Point2f p = k_initial_center + offset * n;
            // 必须使用双线性插值，否则直接取整会破坏亚像素梯度的平滑性！
            float val = BilinearInterpolation(gray_img, p.x, p.y);
            profile[offset + W_int] = val;

            // 找到全线的绝对最大值，锁定发光中心！
            if (val > max_val) 
            { 
                max_val = val; 
                max_idx = offset + W_int; // 锁定最大值索引
            }
            if (val < min_val) 
            { 
                min_val = val; 
            }

            std::cout << profile[offset + W_int] << " ";
        }
        std::cout << "极值索引：" << max_idx << " 极值：" << max_val << " 极小值：" << min_val << std::endl;
        

        // 2. 如果对比度足够 (保底 25.0f)，极值索引有效，才进行边缘提取，过滤纯黑底噪
        if (max_val - min_val > 25.0f && (max_idx > 0 && max_idx < profile_length - 1))
        {
            // 半高宽阈值：位于最暗和最亮正中间的能量等高线
            float thresh = min_val + (max_val - min_val) * 0.5f;

            float left_idx = -1.0f;
            float right_idx = -1.0f;

            // 找左边缘：从顶峰开始向左侧(外侧)搜索，寻找第一次跌破阈值的位置
            for (int i = max_idx; i > 0; i--)
            {
                if (profile[i] >= thresh && profile[i-1] < thresh)
                {
                    // 线性插值计算亚像素偏移 t
                    float t = (profile[i] - thresh) / (profile[i] - profile[i-1]);
                    left_idx = i - t;   // 亚像素索引
                    break;
                }
            }

            // 找右边缘：从顶峰开始向右侧(外侧)搜索，寻找第一次跌破阈值的位置
            for (int i = max_idx; i < profile_length - 1; i++)
            {
                if (profile[i] >= thresh && profile[i+1] < thresh)
                {
                    // 线性插值计算亚像素偏移 t
                    float t = (profile[i] - thresh) / (profile[i] - profile[i+1]);
                    right_idx = i + t;
                    break;
                }
            }
            std::cout << "左边缘索引: " << left_idx << " 右边缘索引: " << right_idx << std::endl;
        
            // 严格配对中点，只有左右双峰同时存在且有效时，才配对求取中点！天然抵消对称膨胀误差！
            if (left_idx >= 0.0f && right_idx >= 0.0f)
            {
                // 寻找左右边缘点，看看离扫描线中点究竟差多远，再乘以法向量
                cv::Point2f pt_l = k_initial_center + (left_idx - W_int) * n;
                cv::Point2f pt_r = k_initial_center + (right_idx - W_int) * n;
                cv::Point2f pt_c = (pt_l + pt_r) * 0.5f; // 两个边缘点的中点视为中轴线点
                std::cout << "左偏移修正left_idx - W_int: " << left_idx - W_int << " 右偏移修正right_idx - W_int: " << right_idx - W_int << std::endl; 

                // 存储每一对左边缘点、右边缘点、中轴线点
                out_left_edges.push_back(pt_l);
                out_right_edges.push_back(pt_r);
                out_centers.push_back(pt_c);
                center_pts.push_back(pt_c);
            }
        }
    }

    // 如果有效的中轴线点太少，不满足 RANSAC 需要的样本数，则返回失败
    if (center_pts.size() < 4)
    {
        return false;
    }

    // 对严格配对出来的完美中点集合，执行 RANSAC+PCA 拟合，输出到 exact_line 里
    // 阈值设置为 0.50f，因为此时的点应该排在一条极其干净的直线上
    int inliers = tools::fitLineRANSAC(center_pts, exact_line, 100, 0.50f);

    // 如果是第一次迭代扫描，这个日志会打印。由于有哑容器的存在，不会污染调试图。
    if (!out_centers.empty()) 
    {
        LOG_INFO("RANSAC+PCA 拟合成功，有效中轴线点数：{}", center_pts.size());
    }

    return inliers >= 4;
}


// 对于某个灯条 上 / 下 端点，优化端点位置的函数
cv::Point2f ArmorCornerRefiner::SearchExactEndpoint(const cv::Mat& gray_img, 
                                                    const cv::Vec4f& exact_line, 
                                                    const cv::Point2f& rough_pt, 
                                                    bool is_top_point,
                                                    float bar_length)
{
    // 1. 将粗端点投影到已经拟合好的纯净中轴线上，作为搜索基准
    cv::Point2f v(exact_line[0], exact_line[1]);    // 直线单位方向向量
    cv::Point2f p0(exact_line[2], exact_line[3]);   // 直线上基准点
    
    // 向量点乘求投影参数 t
    float t_proj = (rough_pt.x - p0.x) * v.x + (rough_pt.y - p0.y) * v.y;
    cv::Point2f project_pt = p0 + t_proj * v; // 投影点

    // 规范方向，v 从上到下，n 从左到右 
    if (v.y < 0) v = -v; 
    cv::Point2f n(v.y, -v.x); // 法向，用于横向提取小带状区域

    // 动态防重叠搜索半径（以端点为中心点的沿轴线搜索半径），最大不超过设定值，且绝不超过灯条长度的 50%！
    int R = std::min((int)endpoint_search_R_, std::max(4, (int)(bar_length * 0.50f)));

    // 2. 构建一维以端点为中心，横向宽度为 3，从上往下的垂直搜索的滑动窗口平均亮度信号，并记录极值索引
    int search_len = R * 2 + 1; // 搜索窗口长度 
    std::vector<float> signal(search_len, 0.0f); // 存储亮度信号，从上到下

    // 依旧找到剖面的最小、最大值，也要记录极值索引
    float min_val = 255.0f;
    float max_val = 0.0f;
    int max_idx = -1;
    
    for (int i = 0; i < search_len; i++)
    {
        // 从投影点开始，沿修正后的中轴线方向步进，步长 1 像素 (从 -R 到 +R)
        float step = (i - R) * 1.0f; 
        cv::Point2f current_p = project_pt + step * v; // 当前点所在位置

        // 横向平均提取法 (Width=3)
        float sum_value = 0.0f;
        sum_value += BilinearInterpolation(gray_img, current_p.x - 1.0f * n.x, current_p.y - 1.0f * n.y); // 当前点沿法线方向，朝左一步
        sum_value += BilinearInterpolation(gray_img, current_p.x,              current_p.y);              // 当前点
        sum_value += BilinearInterpolation(gray_img, current_p.x + 1.0f * n.x, current_p.y + 1.0f * n.y); // 当前点沿法线方向，朝右一步
        
        signal[i] = sum_value / 3.0f; // 取平均作为该点的亮度

        // 锁定局部最高峰(发光核心)和最低谷(背景底噪)
        if (signal[i] > max_val) 
        {
            max_val = signal[i];
            max_idx = i;
        }
        if (signal[i] < min_val) 
        {
            min_val = signal[i];
        }
    }

    float sub_offset = 0.0f; // 相对于窗口中心的亚像素偏移
    bool found = false;

    // 3. 由内向外扫描，寻找首次跌破阈值的下降沿
    // 保底：如果明暗对比太弱，说明灯条完全不可见，退回投影点
    if (max_val - min_val > 25.0f && max_idx > 0 && max_idx < search_len - 1)
    {
        // 0.50 是纯物理半高宽；0.60 偏内侧(主色调)
        // 相对阈值计算，完美免疫底噪环境光！
        float thresh = min_val + (max_val - min_val) * 0.6f;

        if (is_top_point) 
        {
            // 上端点：图像上方是外侧(小索引)。
            // 【由外向内推】：从 0 往中间走，寻找首次跨越阈值的点
            for (int i = 0; i < search_len - 1; i++)
            {
                // 当前点还在阈值之下，但下一个点（更内侧）已经超过阈值
                if (signal[i] <= thresh && signal[i+1] > thresh) 
                {
                    // 线性插值亚像素偏移，精确算出穿越 thresh 的浮点位置
                    float t = (thresh - signal[i]) / (signal[i+1] - signal[i]);
                    sub_offset = i + t - R; 
                    found = true;
                    break;
                }
            }
        }
        else 
        {
            // 下端点：图像下方是外侧(大索引)。
            // 【由外向内推】：从 search_len-1 往中间走(倒序)，寻找首次跨越阈值的点
            for (int i = search_len - 1; i > 0; i--)
            {
                if (signal[i] <= thresh && signal[i-1] > thresh)
                {
                    float t = (thresh - signal[i]) / (signal[i-1] - signal[i]);
                    sub_offset = i - t - R; 
                    found = true;
                    break;
                }
            }
        }
    }

    // 安全校验：如果没有找到，或者偏移量离谱，退回投影中心
    if (!found) 
    {
        sub_offset = 0.0f; 
    }

    // 将亚像素步进映射回去，得到最终优化端点位置
    float final_step = sub_offset * 1.0f;
    return project_pt + final_step * v;
}


// 双线性插值
float ArmorCornerRefiner::BilinearInterpolation(const cv::Mat& img, float x, float y)
{
    // 防止访问越界引发段错误
    if (x < 0.0f || x >= (img.cols - 1) || y < 0.0f || y >= (img.rows - 1)) 
    {
        return 0.0f;
    }

    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix; // x 小数部分
    float dy = y - iy; // y 小数部分

    // 提取周围四个整型像素块的值
    float v00 = img.at<uchar>(iy, ix);          // 当前点所在位置
    float v10 = img.at<uchar>(iy, ix + 1);      // 当前点所在位置的右侧
    float v01 = img.at<uchar>(iy + 1, ix);      // 当前点所在位置的下侧
    float v11 = img.at<uchar>(iy + 1, ix + 1);  // 当前点所在位置的右下侧

    // 标准双线性面积权重求和
    return (1.0f - dx) * (1.0f - dy) * v00 + 
           dx * (1.0f - dy) * v10 + 
           (1.0f - dx) * dy * v01 + 
           dx * dy * v11;
}

// 抛物线插值
float ArmorCornerRefiner::ParabolicInterpolation(float y_minus_1, float y_0, float y_plus_1)
{
    // 设抛物线经过 (-1, y_minus_1), (0, y_0), (1, y_plus_1)
    // 根据顶点公式 x = -b/2a 推导出的亚像素偏移量
    float denominator = y_minus_1 - 2.0f * y_0 + y_plus_1;
    if (std::abs(denominator) < 1e-5f)
    {
        return 0.0f; // 规避极平缓导致的除零崩溃
    }

    return 0.5f * (y_minus_1 - y_plus_1) / denominator;
}


// 获取迭代后存储的 6 组 debug 提取点
std::vector<std::vector<cv::Point2f>> ArmorCornerRefiner::GetDebugPoints()
{
    std::vector<std::vector<cv::Point2f>> debug_pts_bundle = {
        debug_l_left_edges_, debug_l_right_edges_, debug_l_centers_,
        debug_r_left_edges_, debug_r_right_edges_, debug_r_centers_
    };

    return debug_pts_bundle;
}