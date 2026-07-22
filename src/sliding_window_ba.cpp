#include "sliding_window_ba.hpp"
#include "tools/logger.hpp"
#include "tools/math_utils.hpp"
#include <iostream>

SlidingWindowBA::SlidingWindowBA(int window_size, const cv::Mat& K, const cv::Mat& D)
    : window_size_(window_size), K_(K), D_(D), is_initialized_(false)
{
    state_ = Eigen::VectorXd::Zero(8);
    
    // FLU -> OpenCV 转换矩阵
    P_ << 0, -1,  0,
          0,  0, -1,
          1,  0,  0;

    // 【性能提纯】：提取焦距和光心
    if (!K.empty()) 
    {
        fx_ = K.at<double>(0, 0); fy_ = K.at<double>(1, 1);
        cx_ = K.at<double>(0, 2); cy_ = K.at<double>(1, 2);
    } 
    else 
    {
        fx_ = fy_ = cx_ = cy_ = 0.0;
    }

    // 安全提取畸变系数，采用 ptr 行指针避免 at() 的单索引越界！
    if (!D.empty()) 
    {
        const double* dptr = D.ptr<double>(0);
        k1_ = D.cols > 0 ? dptr[0] : 0.0;
        k2_ = D.cols > 1 ? dptr[1] : 0.0;
        p1_ = D.cols > 2 ? dptr[2] : 0.0;
        p2_ = D.cols > 3 ? dptr[3] : 0.0;
        k3_ = D.cols > 4 ? dptr[4] : 0.0;
    } 
    else 
    {
        k1_ = k2_ = p1_ = p2_ = k3_ = 0.0;
    }
}

void SlidingWindowBA::AddObservation(double timestamp, const std::vector<cv::Point2f>& corners,
                                     const std::vector<cv::Point3f>& armor_vertices_cv,
                                     const Eigen::Vector3d& t_flu_pnp, double yaw_pnp)
{
    // 存储该帧观测数据
    FrameObservation obs;
    obs.timestamp_ = timestamp; // 单位 s
    obs.corners_ = corners;
    obs.armor_vertices_cv_ = armor_vertices_cv; // 从外部传入，保存此帧真实的 3D 尺寸
    obs.initial_t_flu_ = t_flu_pnp;
    obs.initial_yaw_ = yaw_pnp;

    window_.push_back(obs);
    
    // 维持滑动窗口大小
    if (window_.size() > window_size_) 
    {
        window_.pop_front(); // 丢弃最旧帧
    }
}

// =========================================================================================
// 【核心残差】：时空运动学预测 + 2D 重投影
// =========================================================================================
Eigen::VectorXd SlidingWindowBA::CalculateResiduals(const Eigen::VectorXd& state)
{
    double tx_curr = state(0), ty_curr = state(1), tz_curr = state(2), yaw_curr = state(3);
    double vx = state(4), vy = state(5), vz = state(6), vyaw = state(7);
    
    // 约束 pitch 和 roll
    double fixed_pitch = tools::deg2rad(15.0); 
    double fixed_roll = 0.0;

    int N = window_.size();
    Eigen::VectorXd residuals(N * 8); 

    double t_curr = window_.back().timestamp_;

    for (int i = 0; i < N; i++)
    {
        // 历史帧的 dt 是负数，计算和当前帧的时间差
        double dt = window_[i].timestamp_ - t_curr;

        // 【安全补丁】：时间戳异常保护 (防爆机制)
        // 防止系统卡顿导致 dt 过大，运动学回推崩溃
        dt = std::max(-0.5, std::min(dt, 0.0));

        // 运动学回推，根据当前状态和速度，倒推当时的理论 3D 位姿
        double frame_tx = tx_curr + vx * dt;
        double frame_ty = ty_curr + vy * dt;
        double frame_tz = tz_curr + vz * dt;
        double frame_yaw = yaw_curr + vyaw * dt;

        Eigen::AngleAxisd yawAngle(frame_yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitchAngle(fixed_pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rollAngle(fixed_roll, Eigen::Vector3d::UnitX());
        Eigen::Matrix3d R_flu = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
        Eigen::Vector3d t_flu(frame_tx, frame_ty, frame_tz);

        Eigen::Matrix3d R_cv = P_ * R_flu * P_.transpose();
        Eigen::Vector3d t_cv = P_ * t_flu;

        // 【极速性能优化】：手工编写相机投影模型，彻底避开 cv::projectPoints 的开销
        for (int k = 0; k < 4; k++) 
        {
            cv::Point3f pt = window_[i].armor_vertices_cv_[k];
            
            // 1. 坐标系转换 (世界 -> 相机)
            Eigen::Vector3d p_cam = R_cv * Eigen::Vector3d(pt.x, pt.y, pt.z) + t_cv;
            
            // 2. 归一化平面投影
            double x = p_cam(0) / p_cam(2);
            double y = p_cam(1) / p_cam(2);
            
            // 3. 径向与切向畸变模型
            double r2 = x*x + y*y;
            double radial = 1.0 + k1_*r2 + k2_*r2*r2 + k3_*r2*r2*r2;
            double x_d = x * radial + 2.0*p1_*x*y + p2_*(r2 + 2.0*x*x);
            double y_d = y * radial + p1_*(r2 + 2.0*y*y) + 2.0*p2_*x*y;
            
            // 4. 映射到像素坐标系
            double u = fx_ * x_d + cx_;
            double v = fy_ * y_d + cy_;
            
            // 计算残差
            residuals(i * 8 + k * 2)     = window_[i].corners_[k].x - u; 
            residuals(i * 8 + k * 2 + 1) = window_[i].corners_[k].y - v; 
        }
    }
    return residuals;
}

// =========================================================================================
// 【雅可比计算】：注意由于包含速度，步长必须分级！
// =========================================================================================
Eigen::MatrixXd SlidingWindowBA::CalculateJacobian(const Eigen::VectorXd& state)
{
    int N = window_.size();
    Eigen::MatrixXd J(N * 8, 8); 
    
    // 【核心修正】：位置 1cm，角度 0.001rad，速度 10cm/s，角速度 0.01rad/s
    // 只有速度步长足够大，乘以微小的 dt 后，才能在像素端产生有效差分！
    Eigen::VectorXd deltas(8);
    deltas << 0.01, 0.01, 0.01, 0.001, 0.1, 0.1, 0.1, 0.01;

    Eigen::VectorXd res_base = CalculateResiduals(state);

    for (int j = 0; j < 8; j++)
    {
        Eigen::VectorXd state_plus = state;
        state_plus(j) += deltas(j);
        Eigen::VectorXd res_plus = CalculateResiduals(state_plus);
        J.col(j) = (res_plus - res_base) / deltas(j);
    }
    return J;
}

// =========================================================================================
// 【鲁棒核函数】：Huber Loss，生成代价值与权重矩阵
// =========================================================================================
double SlidingWindowBA::ComputeHuberCost(const Eigen::VectorXd& residuals, double huber_delta, Eigen::MatrixXd& W)
{
    double total_cost = 0.0;
    int n_points = residuals.size() / 2;
    W = Eigen::MatrixXd::Identity(residuals.size(), residuals.size());

    for (int i = 0; i < n_points; i++)
    {
        double err_x = residuals(i * 2);
        double err_y = residuals(i * 2 + 1);
        double e = std::hypot(err_x, err_y); // 计算像素误差的欧氏距离

        if (e <= huber_delta) 
        {
            // 正常区域：二次型代价，权重为 1
            total_cost += 0.5 * e * e;
        } 
        else 
        {
            // 离群噪点区域：线性代价，权重衰减！拉低这个点对抗整体优化的影响力！
            total_cost += huber_delta * (e - 0.5 * huber_delta);
            double weight = huber_delta / e;
            W(i * 2, i * 2) = weight;
            W(i * 2 + 1, i * 2 + 1) = weight;
        }
    }
    return total_cost;
}

// =========================================================================================
// 【解算】：IRLS (迭代重加权) + LM (阻尼) + Nielsen (自适应)
// =========================================================================================
bool SlidingWindowBA::Optimize()
{
    if (window_.size() < 2) 
    {
        return false; // 没有时序，无法计算速度
    }
    
    // 1. 初始化 / 暖启动
    if (!is_initialized_) 
    {
        FrameObservation& curr = window_.back();
        FrameObservation& first = window_.front();

        state_.head<4>() << curr.initial_t_flu_(0), curr.initial_t_flu_(1), curr.initial_t_flu_(2), curr.initial_yaw_;
        
        double dt = curr.timestamp_ - first.timestamp_;
        if (dt > 0.02 && dt < 1.0) 
        {
            state_(4) = (curr.initial_t_flu_(0) - first.initial_t_flu_(0)) / dt;
            state_(5) = (curr.initial_t_flu_(1) - first.initial_t_flu_(1)) / dt;
            state_(6) = (curr.initial_t_flu_(2) - first.initial_t_flu_(2)) / dt;
            double dyaw = curr.initial_yaw_ - first.initial_yaw_;
            tools::limit_rad_inplace(dyaw); // limit rad
            state_(7) = dyaw / dt;
        } 
        else 
        {
            state_.tail<4>().setZero(); 
        }
        is_initialized_ = true;
    }
    else 
    {
        // 暖启动：利用上一帧算出的速度，预测当前帧的位置
        double dt = window_.back().timestamp_ - window_[window_.size() - 2].timestamp_;

        // 【隐患防爆修复】：绝对不允许利用惯性预测超过 100 毫秒以外的未来！
        dt = std::max(0.0, std::min(dt, 0.1));

        state_(0) += state_(4) * dt;
        state_(1) += state_(5) * dt;
        state_(2) += state_(6) * dt;
        state_(3) += state_(7) * dt;
        tools::limit_rad_inplace(state_(3)); // limit rad
    }

    // 2. IRLS - LM - Nielsen 优化
    int max_iterations = 15;
    double lambda = 1.0;     
    double nu = 2.0;          
    double epsilon = 1e-6;
    double huber_delta = 1.5; // Huber 阈值：误差超过 1.5 个像素的点将被强制降权！

    Eigen::VectorXd current_res = CalculateResiduals(state_);
    Eigen::MatrixXd W;
    double current_cost = ComputeHuberCost(current_res, huber_delta, W);

    // LM 迭代主循环
    for (int iter = 0; iter < max_iterations; iter++)
    {
        Eigen::MatrixXd J = CalculateJacobian(state_);
        
        // 【核心】：H = J^T * W * J，g = -J^T * W * r，利用权重矩阵抵抗离群噪点
        Eigen::MatrixXd H = J.transpose() * W * J;        
        Eigen::VectorXd g = -J.transpose() * W * current_res; 

        // 梯度无穷小判定：已经陷入极其平坦的谷底，直接收敛
        if (g.lpNorm<Eigen::Infinity>() < 1e-8) 
        {
            break;
        }

        // Levenberg-Marquardt 阻尼方程： (H + lambda * I) * dx = g
        Eigen::MatrixXd H_lm = H + lambda * Eigen::MatrixXd::Identity(8, 8);
        
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

        // 防爆步长裁剪 (保护 8 维空间)
        double trans_step = dx.head<3>().norm();
        if (trans_step > 0.5) 
        {
            dx.head<3>() *= 0.5 / trans_step; // 单步平移不得超过 50 厘米
        }
        
        double vel_step = dx.segment<3>(4).norm();
        if (vel_step > 5.0) 
        {
            dx.segment<3>(4) *= 5.0 / vel_step; // 速度单步最大 5m/s变化
        }

        if (std::abs(dx(3)) > 0.2) 
        {
            dx(3) = (dx(3) > 0) ? 0.2 : -0.2; // 单步 Yaw角 旋转最大 11.4 度 (0.2 rad)
        }

        // 步长尝试
        Eigen::VectorXd new_state = state_ + dx;
        tools::limit_rad_inplace(new_state(3)); // 约束 Yaw 在 -PI 到 PI

        Eigen::VectorXd new_res = CalculateResiduals(new_state);
        Eigen::MatrixXd new_W;
        double new_cost = ComputeHuberCost(new_res, huber_delta, new_W);

        // 依据代价函数的下降量计算 Nielsen rho
        // predicted_reduction 是高斯牛顿模型预测的误差下降量
        // 【数学辟谣】：这里的 `+ g` 是 100% 正确的！
        // 因为我们在上方定义 g = -J^T * W * res，它实质上是下降方向（即负梯度）。
        double predicted_reduction = 0.5 * dx.dot(lambda * dx + g);
        double rho = (current_cost - new_cost) / (predicted_reduction + 1e-10);

        if (rho > 0) 
        {
            // Step Accepted (误差确实下降了)
            state_ = new_state;
            current_res = new_res;
            current_cost = new_cost;
            W = new_W; // 更新权重矩阵

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
    
    return true;
}

Eigen::Vector3d SlidingWindowBA::GetCurrentPosition() const 
{ 
    return state_.head<3>(); 
}
double SlidingWindowBA::GetCurrentYaw() const 
{ 
    return state_(3); 
}
Eigen::Vector3d SlidingWindowBA::GetVelocity() const 
{ 
    return state_.segment<3>(4); 
}
double SlidingWindowBA::GetYawVelocity() const 
{ 
    return state_(7); 
}

// 【核心新增】：兜底读取，专防数据关联时的"婴儿夭折"
Eigen::Vector3d SlidingWindowBA::GetLastPosition() const 
{
    if (is_initialized_) return state_.head<3>();
    else if (!window_.empty()) return window_.back().initial_t_flu_;
    else return Eigen::Vector3d::Zero();
}

double SlidingWindowBA::GetLastTimestamp() const 
{ 
    return window_.empty() ? 0.0 : window_.back().timestamp_; 
}
