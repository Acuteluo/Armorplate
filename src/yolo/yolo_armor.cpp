#include "yolo/yolo_armor.hpp"
#include "tools/logger.hpp"
#include "tools/math_utils.hpp"


YoloArmor::YoloArmor() 
{
    // 初始化转换矩阵 P (FLU -> OpenCV，也就是在 OpenCV 坐标系下，FLU 三个轴基向量 分别的坐标)
    /*
        FLU: x前, y左, z上
        OpenCV: x右(-y), y下(-z), z前(x)
    */
    P_ << 0, -1,  0,
          0,  0, -1,
          1,  0,  0;
}


YoloArmor::YoloArmor(int armor_id, int color, bool is_big, float confidence, const cv::Rect& box, 
                     const std::vector<cv::Point2f>& corners, const std::vector<cv::Point2f>& fixed_corners,
                     const cv::Mat& K, const cv::Mat& D)
    : armor_id_(armor_id),
      color_(color),
      is_big_(is_big),
      confidence_(confidence),
      box_(box),
      corners_(corners),
      fixed_corners_(fixed_corners),
      pnp_success_(false),  
      t_distance_(0.0),
      K_(K),        
      D_(D)
{
    // 初始化转换矩阵 P (FLU -> OpenCV，也就是在 OpenCV 坐标系下，FLU 三个轴基向量 分别的坐标)
    P_ << 0, -1,  0,
          0,  0, -1,
          1,  0,  0;
}


void YoloArmor::SetArmorplateSize() 
{
    vertice_world_.clear();
    vertice_cv_.clear();
    
    // 1. 根据 YOLO 的判断动态获取真实物理尺寸
    double armorplate_width = is_big_ ? 0.225 : 0.135;
    double armorplate_height = 0.055;

    // 2. 严格按你旧代码的 FLU 坐标系定义装甲板四个角 (左上, 左下, 右下, 右上)
    // 注意是站在【有数字那一面】向车心看！x=0
    vertice_world_.push_back(cv::Point3f(0.00, +armorplate_width/2, +armorplate_height/2)); // 左上
    vertice_world_.push_back(cv::Point3f(0.00, +armorplate_width/2, -armorplate_height/2)); // 左下
    vertice_world_.push_back(cv::Point3f(0.00, -armorplate_width/2, -armorplate_height/2)); // 右下
    vertice_world_.push_back(cv::Point3f(0.00, -armorplate_width/2, +armorplate_height/2)); // 右上

    // 3. 将 FLU 系的点，乘上 P 矩阵，转换到 OpenCV 系下供 PnP 使用
    for (const auto& pt : vertice_world_) 
    {
        Eigen::Vector3d pt_flu(pt.x, pt.y, pt.z);
        Eigen::Vector3d pt_cv = P_ * pt_flu; // P_cv = P * P_flu
        vertice_cv_.push_back(cv::Point3f(pt_cv(0), pt_cv(1), pt_cv(2)));
    }
}



void YoloArmor::PNP(bool use_fixed_corners) 
{
    std::vector<cv::Point2f> corners_for_pnp;
    corners_for_pnp = use_fixed_corners ? fixed_corners_ : corners_;

    // 如果传入的角点数量不对，直接标记失败并返回
    if (corners_for_pnp.size() != 4 || vertice_cv_.size() != 4) 
    {
        pnp_success_ = false;
        return;
    }

    // 1. 执行 OpenCV 的 solvePnP，用 IPPE 方法
    pnp_success_ = cv::solvePnP(vertice_cv_, corners_for_pnp, K_, D_, r_cv_, t_cv_, false, cv::SOLVEPNP_IPPE);

    if (!pnp_success_) 
    {
        LOG_ERROR("[PNP] PNP 解算失败! ");
        return;
    }

    // ==========================================================
    // 2. 核心魔法：将 OpenCV 坐标系还原回 FLU 右手系！
    // ==========================================================
    
    // (A) 处理平移向量 t
    Eigen::Vector3d t_cv_eigen(t_cv_.at<double>(0, 0), t_cv_.at<double>(1, 0), t_cv_.at<double>(2, 0));
    t_flu_ = P_.transpose() * t_cv_eigen; // t_flu_ = P^(-1) * t_cv_
    t_distance_ = t_flu_.norm();

    // (B) 处理旋转矩阵 R
    cv::Mat R_cv;
    cv::Rodrigues(r_cv_, R_cv); // 先把 OpenCV系 下的 旋转向量 r_cv 转成 旋转矩阵 R_cv
    
    Eigen::Matrix3d R_cv_eigen;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R_cv_eigen(i, j) = R_cv.at<double>(i, j);
        }
    }

    // 还原回 FLU 右手系，保持纯原始的 6-DoF 状态作为 Baseline
    R_flu_ = P_.transpose() * R_cv_eigen * P_; // R_flu_ = P^(-1) * R_cv * P

    if (use_fixed_corners)
    {
        // =========================================================================
        // 【核心架构加入】：执行 4-DoF Levenberg-Marquardt 非线性优化
        // 目的：打破 Z 和 Yaw 的耦合，并约束 Roll=0, Pitch=15
        // =========================================================================
        OptimizePoseLM();

        // 优化完成后，重新提取最终的距离和角度供外层使用
        t_distance_ = t_flu_.norm();
    }

    // 从旋转矩阵中提取相对 Yaw 角 (仅供参考和打印)
    yaw_angle_ = tools::rad2deg(std::atan2(R_flu_(1, 0), R_flu_(0, 0)));

    if (show_logger_pnp_) 
    {
        if (use_fixed_corners) LOG_DEBUG("[PNP] 修正后的 PnP计算完毕, X: {:.2f}, Y: {:.2f}, Z: {:.2f}, Yaw: {:.2f}°", t_flu_(0), t_flu_(1), t_flu_(2), yaw_angle_);
        else LOG_DEBUG("[PNP] 原始的 PnP计算完毕, X: {:.2f}, Y: {:.2f}, Z: {:.2f}, Yaw: {:.2f}°", t_flu_(0), t_flu_(1), t_flu_(2), yaw_angle_);
    }
}




// =============================================================================
// 以下是为时序 BA 铺垫的核心：4-DoF 非线性优化器 (Levenberg-Marquardt)
// =============================================================================

// 计算残差 (观测 2D 点 - 投影 2D 点)
Eigen::VectorXd YoloArmor::CalculateResiduals(const Eigen::VectorXd& state)
{
    // 状态向量 state: [X, Y, Z, Yaw]
    double tx = state(0);
    double ty = state(1);
    double tz = state(2);
    double yaw = state(3);
    
    // 物理约束先验：装甲板不可侧倾，且固定有 15度 仰角
    double fixed_pitch = tools::deg2rad(15.0); 
    double fixed_roll = 0.0;

    // 构建 FLU 旋转矩阵
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(fixed_pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rollAngle(fixed_roll, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d R_flu = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
    Eigen::Vector3d t_flu(tx, ty, tz);

    // 转换到相机系
    Eigen::Matrix3d R_cv = P_ * R_flu * P_.transpose();
    Eigen::Vector3d t_cv = P_ * t_flu;

    cv::Mat R_cv_mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_cv_mat.at<double>(i, j) = R_cv(i, j);

    cv::Mat rvec_test, tvec_test;
    cv::Rodrigues(R_cv_mat, rvec_test);
    tvec_test = (cv::Mat_<double>(3, 1) << t_cv(0), t_cv(1), t_cv(2));

    // 执行重投影
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(vertice_cv_, rvec_test, tvec_test, K_, D_, projected_points);

    // 构建 8 维残差向量 [u0_err, v0_err, u1_err, v1_err ...]
    Eigen::VectorXd residuals(8);
    for (int i = 0; i < 4; i++) 
    {
        // 必须使用精化后的高精度角点 fixed_corners_
        residuals(i * 2)     = fixed_corners_[i].x - projected_points[i].x; 
        residuals(i * 2 + 1) = fixed_corners_[i].y - projected_points[i].y; 
    }

    return residuals;
}

// 数值微分计算雅可比矩阵
Eigen::MatrixXd YoloArmor::CalculateJacobian(const Eigen::VectorXd& state)
{
    Eigen::MatrixXd J(8, 4);
    
    // 扰动量 (平移 1cm，角度 0.001 弧度)
    double delta_t = 0.01; 
    double delta_r = 0.001; 
    Eigen::VectorXd deltas(4);
    deltas << delta_t, delta_t, delta_t, delta_r;

    Eigen::VectorXd res_base = CalculateResiduals(state);

    for (int j = 0; j < 4; j++)
    {
        Eigen::VectorXd state_plus = state;
        state_plus(j) += deltas(j);
        Eigen::VectorXd res_plus = CalculateResiduals(state_plus);
        J.col(j) = (res_plus - res_base) / deltas(j);
    }
    return J;
}

void YoloArmor::OptimizePoseLM()
{
    // 1. 初始化状态量 [X, Y, Z, Yaw]
    Eigen::VectorXd state(4);
    double initial_yaw = std::atan2(R_flu_(1, 0), R_flu_(0, 0)); 
    state << t_flu_(0), t_flu_(1), t_flu_(2), initial_yaw;

    int max_iterations = 20;
    double lambda = 1.0;      // LM 阻尼系数初始值
    double nu = 2.0;          // Nielsen 策略：拒绝惩罚倍数因子
    double epsilon = 1e-6;    // 收敛精度提升到 1e-6

    Eigen::VectorXd current_res = CalculateResiduals(state);
    double current_error = current_res.squaredNorm();

    // 2. LM 迭代主循环
    for (int iter = 0; iter < max_iterations; iter++)
    {
        Eigen::MatrixXd J = CalculateJacobian(state);
        Eigen::MatrixXd H = J.transpose() * J;        // 算 Hessian 矩阵
        Eigen::VectorXd g = -J.transpose() * current_res; // 算梯度

        // 梯度无穷小判定：已经陷入极其平坦的谷底，直接收敛
        if (g.lpNorm<Eigen::Infinity>() < 1e-8) 
        {
            break;
        }

        // Levenberg-Marquardt 阻尼方程： (H + lambda * I) * dx = g
        // 这一步等价于你在周报中提到的“监测 SVD 并自适应阻尼”。
        // 当 H 接近奇异（Z轴难以观测）时，lambda*I 强制让其可逆，防止步长爆炸。
        Eigen::MatrixXd H_lm = H + lambda * Eigen::MatrixXd::Identity(4, 4);
        
        // 使用 LDLT 分解求解，安全稳定
        Eigen::VectorXd dx = H_lm.ldlt().solve(g); 

        if (std::isnan(dx(0))) 
        {
            break; // 矩阵病态保护
        }

        if (dx.norm() < epsilon) 
        {
            break; // 已收敛
        }

        // -------------------------------------------------------------
        // 步长防爆裁剪 (Step Clipping)
        // 绝对不允许优化器单步走火入魔，限制最大步长
        // -------------------------------------------------------------
        double trans_step = dx.head<3>().norm();
        if (trans_step > 0.1) // 单步平移不得超过 10 厘米
        {
            dx.head<3>() *= 0.1 / trans_step;
        }
        if (std::abs(dx(3)) > 0.1) // 单步 Yaw角 旋转最大 5.7度 (0.1 rad)
        {
            dx(3) = (dx(3) > 0) ? 0.1 : -0.1;
        }

        // 步长尝试
        Eigen::VectorXd new_state = state + dx;
        tools::limit_rad_inplace(new_state(3)); // 约束 Yaw 在 -PI 到 PI

        Eigen::VectorXd new_res = CalculateResiduals(new_state);
        double new_error = new_res.squaredNorm();

        // 核心：Nielsen 策略的增益比 (Gain Ratio) 计算
        // predicted_reduction 是高斯牛顿模型预测的误差下降量
        double predicted_reduction = dx.dot(lambda * dx + g);
        double rho = (current_error - new_error) / (predicted_reduction + 1e-10); // 防止除 0

        if (rho > 0) 
        {
            // Step Accepted (误差确实下降了)
            state = new_state;
            current_res = new_res;
            current_error = new_error;
            
            // 如果 rho 很大 (接近1)，说明模型非常准确，阻尼急剧减小 (最少降至 1/3)
            // 如果 rho 很小，说明地形坑洼，稍微减小阻尼
            double temp = 1.0 - std::pow(2.0 * rho - 1.0, 3);
            lambda *= std::max(1.0 / 3.0, temp);
            nu = 2.0; // 重置惩罚因子
        } 
        else 
        {
            // Step Rejected (误差反而上升了，遇到了极其恶劣的非线性区)
            // 采用倍增惩罚策略，迅速退化为保守的梯度下降法
            lambda *= nu;
            nu *= 2.0; 
        }
    }

    // 3. 将优化结果写回成员变量
    t_flu_(0) = state(0);
    t_flu_(1) = state(1);
    t_flu_(2) = state(2);
    
    // 重建被强约束纠正后的旋转矩阵
    double fixed_pitch = tools::deg2rad(15.0); 
    Eigen::AngleAxisd yawAngle(state(3), Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(fixed_pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());
    R_flu_ = (yawAngle * pitchAngle * rollAngle).toRotationMatrix(); 

    // 计算最终 RMSE
    reprojection_error_ = std::sqrt(current_error / 4.0);
}









// 计算传入的 t_flu 和 R_flu 的重投影误差
double YoloArmor::CalculateReprojectionError(const Eigen::Vector3d& t_flu, const Eigen::Matrix3d& R_flu)
{
    // 重投影需要模型点、相机内参、相机外参 t_flu 和 R_flu
    if (vertice_cv_.empty() || corners_.empty()) return -1.0;

    // 1. 将外部传入的 FLU 坐标系状态 ，在内部转换为 OpenCV 坐标系
    // t_cv = P * t_flu
    Eigen::Vector3d t_cv_eigen = P_ * t_flu;

    // R_cv = P * R_flu * P^(-1)
    Eigen::Matrix3d R_cv_eigen = P_ * R_flu * P_.transpose();

    // 2. 转为 OpenCV 需要的 cv::Mat 格式
    cv::Mat R_cv_mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R_cv_mat.at<double>(i, j) = R_cv_eigen(i, j);
        }
    }

    cv::Mat r_cv, t_cv;
    cv::Rodrigues(R_cv_mat, r_cv); // 把 R_cv_eigen 转为 OpenCV 旋转矩阵 r_cv
    t_cv = (cv::Mat_<double>(3, 1) << t_cv_eigen(0), t_cv_eigen(1), t_cv_eigen(2)); // 把 t_cv_eigen 转为 OpenCV 平移向量 t_cv


    // 3. 调用 OpenCV 进行投影计算
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(vertice_cv_, r_cv, t_cv, K_, D_, projected_points);


    // 4. 计算与实际检测角点之间的 L2 像素误差均方根 (RMSE)
    double sum_squared_error = 0.00;
    for (int i = 0; i < 4; i++) 
    {
        double dx = projected_points[i].x - corners_[i].x;
        double dy = projected_points[i].y - corners_[i].y;
        sum_squared_error += (dx * dx + dy * dy); 
    }

    reprojection_error_ = std::sqrt(sum_squared_error / 4.0); // 均方根
    LOG_DEBUG("[PNP] 重投影误差: {:.6f}", reprojection_error_);
    return reprojection_error_; 
}



// 在图像上画框并打印信息，模式 "simple" / "complex" 仅输出框 / 打印所有信息
cv::Mat YoloArmor::DrawAndPrintInfo(const cv::Mat& original_img, std::string mode_select)
{
    cv::Mat img_show = original_img.clone();

    // 1. 画框与四个角点
    cv::Scalar edge_color = cv::Scalar(235, 206, 135); 
    for (int i = 0; i < 4; i++) 
    {
        // cv::line(img_show, corners_[i], corners_[(i + 1) % 4], edge_color, 2);

        // 原始角点所在像素变成蓝色 
        cv::circle(img_show, corners_[i], 0.01, cv::Scalar(255, 0, 0), cv::FILLED);

        // 精化角点所在像素变成绿色
        cv::circle(img_show, fixed_corners_[i], 0.01, cv::Scalar(0, 255, 0), cv::FILLED);

        // 画角点序号，方便验证顺序 (0左上, 1左下, 2右下, 3右上)
        // cv::putText(img_show, std::to_string(i), corners_[i], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 128, 255), 2);
    }

    // 2. 找中心点
    cv::Point2f center((corners_[0].x + corners_[2].x)/2, (corners_[0].y + corners_[2].y)/2);
    cv::circle(img_show, center, 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // 3. 打印 RMSE
    cv::putText(img_show, "RMSE: " + std::to_string(reprojection_error_).substr(0, 6), cv::Point2f(30.00, 30.00), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    // complex 模式下打印完整信息
    if (mode_select == "complex")
    {
        // 3. 打印 PnP 的信息 (距离、Pitch、Yaw) 在角点旁边
        std::string info_conf = "Conf: " + std::to_string(confidence_).substr(0, 5);
        std::string info_dist = "Dist: " + std::to_string(t_distance_).substr(0, 4) + "m";

        cv::putText(img_show, info_conf, cv::Point2f(box_.x, box_.y - 75), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(img_show, info_dist, cv::Point2f(box_.x, box_.y - 55), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // 4. 打印装甲板 ID 和 类型 和 优化后的欧拉角
        std::string type_str = is_big_ ? "BIG" : "SMALL";
        int color_str = color_;
        std::string info_yaw = "e_Yaw: " + std::to_string(yaw_angle_).substr(0, 5);

        cv::putText(img_show, "ID:" + std::to_string(armor_id_) + " " + type_str, cv::Point2f(box_.x, box_.y + box_.height + 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(img_show, "COLOR:" + std::to_string(color_str), cv::Point2f(box_.x, box_.y + box_.height + 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(img_show, info_yaw, cv::Point2f(box_.x, box_.y + box_.height + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    return img_show;
}


cv::Mat YoloArmor::DrawMagnifiedROI(const cv::Mat& original_img, 
                                    const cv::Vec4f& left_middle_line, 
                                    const cv::Vec4f& right_middle_line,
                                    const std::vector<std::vector<cv::Point2f>>& debug_pts,
                                    int scale)
{
    // 1. 根据 YOLO 框扩大 ROI 范围 (扩大 15 像素以容纳周围背景)
    int expand = 10;
    cv::Rect roi = box_;
    roi.x -= expand; 
    roi.y -= expand;
    roi.width += 2 * expand; 
    roi.height += 2 * expand;
    
    // 增加边界判断的裁剪 roi，防止越界
    roi.x = std::max(0, roi.x);
    roi.y = std::max(0, roi.y);
    roi.width = std::min(original_img.cols - roi.x, roi.width);
    roi.height = std::min(original_img.rows - roi.y, roi.height);
    if (roi.width <= 0 || roi.height <= 0) 
    {
        return cv::Mat();
    }


    // 2. 将 roi 图像 cropped，长宽各放大到原来的 scale 倍，结果输出到 magnified 中，并且缩放时采用“最近邻插值”算法
    cv::Mat cropped = original_img(roi);
    cv::Mat magnified;
    cv::resize(cropped, magnified, cv::Size(), scale, scale, cv::INTER_NEAREST);

    // 绘制灰色细线网格，清晰看到原图的每一个像素块
    // for (int i = 0; i <= magnified.cols; i += scale) 
    // {
    //     cv::line(magnified, cv::Point(i, 0), cv::Point(i, magnified.rows), cv::Scalar(80, 80, 80), 1);
    // }
    // for (int i = 0; i <= magnified.rows; i += scale) 
    // {
    //     cv::line(magnified, cv::Point(0, i), cv::Point(magnified.cols, i), cv::Scalar(80, 80, 80), 1);
    // }


    // B. 绘制红色整数坐标轴，并标上数字 (代表 x.0 和 y.0 的像素中心位置)
    for (int i = 0; i < roi.width; i++) 
    {
        // 像素块的正中心在缩放图中的确切位置
        int x_center = i * scale + scale / 2;
        int real_x = roi.x + i;
        
        // 垂直红色指示线（表示整数 x 轴）
        cv::line(magnified, cv::Point(x_center, 0), cv::Point(x_center, magnified.rows), cv::Scalar(0, 0, 180), 1);
        
        // 在顶部写上真实的 X 坐标
        cv::putText(magnified, std::to_string(real_x), cv::Point(x_center - 10, 15), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 200), 1);
    }
    
    for (int i = 0; i < roi.height; i++) 
    {
        int y_center = i * scale + scale / 2;
        int real_y = roi.y + i;
        
        // 水平红色指示线（表示整数 y 轴）
        cv::line(magnified, cv::Point(0, y_center), cv::Point(magnified.cols, y_center), cv::Scalar(0, 0, 180), 1);
        
        // 在左侧写上真实的 Y 坐标
        cv::putText(magnified, std::to_string(real_y), cv::Point(5, y_center + 4), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 200), 1);
    }


    // 内部 Lambda 工具：将原图亚像素坐标映射到放大图的整型坐标上
    // opencv定义是像素块中心是整数坐标。但为了符合肉眼直觉，向右偏移 0.5 像素，这样边界线是整数坐标，像素块中心是.5坐标
    auto map_pt = [&](const cv::Point2f& p) {
        return cv::Point2f(std::round((p.x - roi.x + 0.5f) * scale), 
                           std::round((p.y - roi.y + 0.5f) * scale));
    };

    // 内部 Lambda 工具：在放大图上画射线
    auto draw_axis_line = [&](const cv::Vec4f& line_eq, cv::Scalar color) {
        if (line_eq[0] == 0 && line_eq[1] == 0) 
        {
            return; // 无效直线
        }
        cv::Point2f v(line_eq[0], line_eq[1]);
        cv::Point2f p0(line_eq[2], line_eq[3]);

        // 沿向量向上下各延伸 30 个像素画线
        cv::Point2f p1 = p0 - v * 30.0f; 
        cv::Point2f p2 = p0 + v * 30.0f;
        cv::line(magnified, map_pt(p1), map_pt(p2), color, 2);
    };


    // 3. 画出 RANSAC 拟合出的灯条中轴线 (黄色)
    draw_axis_line(left_middle_line, cv::Scalar(0, 255, 255));
    draw_axis_line(right_middle_line, cv::Scalar(0, 255, 255));

    // ==============================================================
    // 渲染每一层的提取点，作为强力的对齐查错手段！
    auto draw_cross = [&](const cv::Point2f& pt, cv::Scalar color, int size) {
        cv::Point2f mapped = map_pt(pt);
        // 画十字
        // cv::line(magnified, cv::Point(mapped.x - size, mapped.y), cv::Point(mapped.x + size, mapped.y), color, 1);
        // cv::line(magnified, cv::Point(mapped.x, mapped.y - size), cv::Point(mapped.x, mapped.y + size), color, 1);
    
        // 直接画点
        cv::circle(magnified, mapped, size, color, cv::FILLED);
    };

    // 4. 画出 Debug 角点，顺序：左边缘、右边缘、中点
    if(debug_pts.size() == 6) 
    {
        // 左灯条边缘与中心：左粉，右橙，中白
        for(auto& p : debug_pts[0]) draw_cross(p, cv::Scalar(255, 0, 255), scale/5); 
        for(auto& p : debug_pts[1]) draw_cross(p, cv::Scalar(0, 165, 255), scale/5); 
        for(auto& p : debug_pts[2]) draw_cross(p, cv::Scalar(255, 255, 255), scale/5); 
        
        // 右灯条边缘与中心：左粉，右橙，中白
        for(auto& p : debug_pts[3]) draw_cross(p, cv::Scalar(255, 0, 255), scale/5); 
        for(auto& p : debug_pts[4]) draw_cross(p, cv::Scalar(0, 165, 255), scale/5); 
        for(auto& p : debug_pts[5]) draw_cross(p, cv::Scalar(255, 255, 255), scale/5); 
    }
    // ==============================================================


    // 5. 画出原始的 YOLO 粗糙角点 (蓝色实心圆)
    for (const auto& pt : corners_) 
    {
        cv::circle(magnified, map_pt(pt), scale / 3.5, cv::Scalar(255, 0, 0), cv::FILLED);
    }


    // 6. 画出我们的高精度亚像素修正角点 (绿色实心圆)
    // 你会清楚地看到绿点偏离蓝点，稳稳地落在黄色直线上，并且精准卡在灯条尽头的网格内！
    for (const auto& pt : fixed_corners_) 
    {
        cv::circle(magnified, map_pt(pt), scale / 3.5, cv::Scalar(0, 255, 0), cv::FILLED);
    }
        

    // 7. 回传 magnified 图
    return magnified;
}


void YoloArmor::PrintDebugLog(bool is_debug)
{ 
    show_logger_pnp_ = is_debug;
}