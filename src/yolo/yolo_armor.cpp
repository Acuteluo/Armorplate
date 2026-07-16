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


YoloArmor::YoloArmor(int armor_id, int color, bool is_big, float confidence,
                     const cv::Rect& box, const std::vector<cv::Point2f>& corners,
                     const cv::Mat& K, const cv::Mat& D)
    : armor_id_(armor_id),
      color_(color),
      is_big_(is_big),
      confidence_(confidence),
      box_(box),
      corners_(corners),
      pnp_success_(false),  
      t_distance_(0.0),
      obs_yaw_angle_(0.0),
      obs_pitch_angle_(0.0),
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


void YoloArmor::PNP() 
{
    // 如果传入的角点数量不对，直接标记失败并返回
    if (corners_.size() != 4 || vertice_cv_.size() != 4) 
    {
        pnp_success_ = false;
        return;
    }

    // 1. 执行 OpenCV 的 solvePnP (极高精度的 2D 物理角点，直接 IPPE)
    pnp_success_ = cv::solvePnP(vertice_cv_, corners_, K_, D_, r_cv_, t_cv_, false, cv::SOLVEPNP_IPPE);

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
    t_vec_ = P_.transpose() * t_cv_eigen; // t_flu = P^(-1) * t_cv

    // (B) 处理旋转矩阵 R
    cv::Mat R_cv_mat;
    cv::Rodrigues(r_cv_, R_cv_mat); // 先把 OpenCV系 下的 旋转向量 r_cv 转成 旋转矩阵 R_cv_mat
    
    Eigen::Matrix3d R_cv_eigen;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R_cv_eigen(i, j) = R_cv_mat.at<double>(i, j);
        }
    }

    R_ = P_.transpose() * R_cv_eigen * P_; // R_flu = P^(-1) * R_cv * P

    // ==========================================================
    // 3. 供人类调试和下位机使用的极坐标 (Yaw, Pitch, Distance)
    // ==========================================================
    
    // 先马上优化欧拉角
    OptimizeEulerYaw();

    double X = t_vec_(0); // 向前
    double Y = t_vec_(1); // 向左
    double Z = t_vec_(2); // 向上

    t_distance_ = std::sqrt(X * X + Y * Y + Z * Z);
    obs_pitch_angle_ = tools::rad2deg(-std::atan2(Z, std::sqrt(X * X + Y * Y)));
    obs_yaw_angle_ = tools::rad2deg(std::atan2(Y, X));
    
}


// 进行 区间搜索 以找到最优的欧拉角 yaw
double YoloArmor::SearchOptimalEulerYaw(double initial_euler_yaw, double search_range, double search_step)
{
    double min_error = 1e9; // 初始为无穷大
    double best_euler_yaw = initial_euler_yaw;

    for (double i = -search_range; i <= search_range; i += search_step)
    {
        double test_yaw_rad = initial_euler_yaw + tools::deg2rad(i);

        double current_error = CalculateReprojectionError(test_yaw_rad);

        if (current_error < min_error)
        {
            min_error = current_error;
            best_euler_yaw = test_yaw_rad;
        }
    }

    return best_euler_yaw;
}



// 运用 重投影误差 + 基于全局网络搜索 得到最优的欧拉角yaw，来优化 PnP 解算的欧拉角
void YoloArmor::OptimizeEulerYaw()
{
    if (!pnp_success_) return;

    // 1. 获取 PnP 粗略算出的 欧拉角 Yaw，作为搜索的基准中心
    // R 已经是 FLU 坐标系下的旋转矩阵了，提取欧拉 Yaw (绕 Z 轴)
    double pnp_euler_yaw = std::atan2(R_(1, 0), R_(0, 0));

    LOG_DEBUG("[PNP] PNP 粗略算出的原欧拉角 Yaw: {:.2f}°", tools::rad2deg(pnp_euler_yaw));

    // 2. 确定搜索区间：既然 PnP 大体是对的，我们就先在 PnP_Yaw 左右各扩展 70 度进行粗搜，再在 最优解 左右各扩展 1 度进行精搜

    double coarse_euler_yaw = SearchOptimalEulerYaw(pnp_euler_yaw, 70.0, 1.0);
    double precise_euler_yaw = SearchOptimalEulerYaw(coarse_euler_yaw, 1.0, 0.1);

    euler_yaw_angle_ = tools::rad2deg(precise_euler_yaw);
    double euler_yaw_diff = std::abs((precise_euler_yaw - pnp_euler_yaw));
    LOG_DEBUG("[PNP] 区间搜索算出的最优欧拉角 Yaw: {:.2f}°", euler_yaw_angle_ );
    LOG_DEBUG("[PNP] 校准差角 Yaw: {:.2f}°", tools::rad2deg(euler_yaw_diff));
    if (tools::rad2deg(euler_yaw_diff) > 10.00) 
    {
        LOG_WARN("[PNP] 校准差角太大，请检查！");
    }

    double fixed_pitch = tools::deg2rad(15.0); 
    Eigen::AngleAxisd yawAngle(precise_euler_yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(fixed_pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());
    
    // 覆盖 Eigen 的旋转矩阵 (FLU 坐标系)
    R_ = (yawAngle * pitchAngle * rollAngle).toRotationMatrix(); 

    // 同步覆盖 OpenCV 的旋转向量 (为了画重投影框用)
    Eigen::Matrix3d R_cv_eigen = P_ * R_ * P_.transpose();
    
    cv::Mat R_cv_mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_cv_mat.at<double>(i, j) = R_cv_eigen(i, j);
            
    cv::Rodrigues(R_cv_mat, r_cv_); 
}



// 计算给定的 欧拉角yaw（弧度） 和 pitch=+15°、roll=0°时，在图上的重投影误差
double YoloArmor::CalculateReprojectionError(double test_euler_yaw)
{
    // 2. 构造临时的测试姿态旋转矩阵 R_cv_test

    double fixed_pitch = tools::deg2rad(15.0); 
    
    Eigen::AngleAxisd yawAngle(test_euler_yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(fixed_pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());
    
    // FLU 系下的测试姿态
    Eigen::Matrix3d R_flu_test = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
    Eigen::Matrix3d R_cv_eigen = P_ * R_flu_test * P_.transpose();

    cv::Mat R_cv_test(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_cv_test.at<double>(i, j) = R_cv_eigen(i, j);

    cv::Mat rvec_test;
    cv::Rodrigues(R_cv_test, rvec_test);

    // 3. 将 3D 模型点重投影到 2D
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(vertice_cv_, rvec_test, t_cv_, K_, D_, projected_points);

    // 【同济核心】：使用 4 个角点的全要素 L2 像素距离 (X 和 Y 共同约束！)
    double total_error = 0.0;
    for (int i = 0; i < 4; i++) 
    {
        double dx = projected_points[i].x - corners_[i].x;
        double dy = projected_points[i].y - corners_[i].y;
        total_error += std::sqrt(dx * dx + dy * dy); 
    }

    return total_error;
}





// 在图像上画框并打印信息，模式 "simple" / "complex" 仅输出框 / 打印所有信息
void YoloArmor::DrawAndPrintInfo(cv::Mat& img_show, std::string mode_select)
{
    // 1. 画框与四个角点
    cv::Scalar edge_color = cv::Scalar(235, 206, 135); 
    for (int i = 0; i < 4; i++) 
    {
        cv::line(img_show, corners_[i], corners_[(i + 1) % 4], edge_color, 2);
        // 画角点序号，方便验证顺序 (0左上, 1左下, 2右下, 3右上)
        cv::putText(img_show, std::to_string(i), corners_[i], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 128, 255), 2);
    }

    // 2. 找中心点
    cv::Point2f center((corners_[0].x + corners_[2].x)/2, (corners_[0].y + corners_[2].y)/2);
    cv::circle(img_show, center, 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // complex 模式下打印完整信息
    if (mode_select == "complex")
    {
        // 3. 打印 PnP 的信息 (距离、Pitch、Yaw) 在角点旁边
        std::string info_conf = "Conf: " + std::to_string(confidence_).substr(0, 5);
        std::string info_dist = "Dist: " + std::to_string(t_distance_).substr(0, 4) + "m";
        std::string info_obs_pitch = "o_Pitch: " + std::to_string(obs_pitch_angle_).substr(0, 5);
        std::string info_obs_yaw = "o_Yaw: " + std::to_string(obs_yaw_angle_).substr(0, 5);

        cv::putText(img_show, info_conf, cv::Point2f(box_.x, box_.y - 75), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(img_show, info_dist, cv::Point2f(box_.x, box_.y - 55), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(img_show, info_obs_pitch, cv::Point2f(box_.x, box_.y - 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(img_show, info_obs_yaw, cv::Point2f(box_.x, box_.y - 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        // 4. 打印装甲板 ID 和 类型 和 优化后的欧拉角
        std::string type_str = is_big_ ? "BIG" : "SMALL";
        int color_str = color_;
        std::string info_euler_yaw = "e_Yaw: " + std::to_string(euler_yaw_angle_).substr(0, 5);

        cv::putText(img_show, "ID:" + std::to_string(armor_id_) + " " + type_str, cv::Point2f(box_.x, box_.y + box_.height + 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(img_show, "COLOR:" + std::to_string(color_str), cv::Point2f(box_.x, box_.y + box_.height + 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(img_show, info_euler_yaw, cv::Point2f(box_.x, box_.y + box_.height + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
}


void YoloArmor::PrintDebugLog(bool is_debug)
{ 
    this->show_logger_pnp_ = is_debug;
}