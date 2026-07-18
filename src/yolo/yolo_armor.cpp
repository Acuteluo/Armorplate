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

    // 1. 执行 OpenCV 的 solvePnP，用 IPPE 方法
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

    // 从粗糙的 6-DoF 旋转矩阵中提取相对 Yaw 角 (仅供参考和打印)
    yaw_angle_ = tools::rad2deg(std::atan2(R_flu_(1, 0), R_flu_(0, 0)));

    if (show_logger_pnp_) 
    {
        LOG_DEBUG("[PNP] 原始PnP计算完毕, X: {:.2f}, Y: {:.2f}, Z: {:.2f}, Yaw: {:.2f}°", t_flu_(0), t_flu_(1), t_flu_(2), yaw_angle_);
    }
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
void YoloArmor::DrawAndPrintInfo(cv::Mat& img_show, std::string mode_select, std::vector<cv::Point2f> pts)
{
    // 1. 画框与四个角点
    cv::Scalar edge_color = cv::Scalar(235, 206, 135); 
    for (int i = 0; i < 4; i++) 
    {
        // cv::line(img_show, corners_[i], corners_[(i + 1) % 4], edge_color, 2);

        // 原始角点所在像素变成蓝色 
        if (pts.size() > 0)
        {
            cv::circle(img_show, pts[i], 0.01, cv::Scalar(255, 0, 0), cv::FILLED);
        }

        // 把角点所在像素变成绿色
        cv::circle(img_show, corners_[i], 0.01, cv::Scalar(0, 255, 0), cv::FILLED);

        // 画角点序号，方便验证顺序 (0左上, 1左下, 2右下, 3右上)
        // cv::putText(img_show, std::to_string(i), corners_[i], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 128, 255), 2);
    }

    // 2. 找中心点
    cv::Point2f center((corners_[0].x + corners_[2].x)/2, (corners_[0].y + corners_[2].y)/2);
    cv::circle(img_show, center, 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // 3. 打印 RMSE
    cv::putText(img_show, "RMSE: " + std::to_string(reprojection_error_).substr(0, 6), cv::Point2f(20.00, 20.00), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

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
}


void YoloArmor::PrintDebugLog(bool is_debug)
{ 
    show_logger_pnp_ = is_debug;
}