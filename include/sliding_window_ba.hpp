#ifndef SLIDING_WINDOW_BA_HPP
#define SLIDING_WINDOW_BA_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

// 存储单帧观测数据的结构体
struct FrameObservation 
{
    double timestamp_;                        // 该帧的时间戳 (秒)
    std::vector<cv::Point2f> corners_;        // 该帧提取到的 4 个高精度亚像素角点
    
    std::vector<cv::Point3f> armor_vertices_cv_; // 存入该帧对应的真实物理尺寸 3D 顶点 (OpenCV系)

    Eigen::Vector3d initial_t_flu_;           // 单帧 BA 算出的 t (提供初值)
    double initial_yaw_;                      // 单帧 BA 算出的 yaw (提供初值)
};

class SlidingWindowBA 
{
public:
    // 构造函数，传入滑动窗口最大帧数 N，以及相机内参
    SlidingWindowBA(int window_size = 5, const cv::Mat& K = cv::Mat(), const cv::Mat& D = cv::Mat());
    ~SlidingWindowBA() = default;

    /**
     * @brief 塞入最新的观测数据
     */
    void AddObservation(double timestamp, const std::vector<cv::Point2f>& corners,
                        const std::vector<cv::Point3f>& armor_vertices_cv,
                        const Eigen::Vector3d& t_flu_pnp, double yaw_pnp);

    /**
     * @brief 执行时序滑动窗口联合优化 (LM + Huber + Nielsen)
     * @return 优化是否成功
     */
    bool Optimize();

    // --- 获取优化后的结果 ---
    Eigen::Vector3d GetCurrentPosition() const;
    double GetCurrentYaw() const;
    Eigen::Vector3d GetVelocity() const;     // 附带计算出的 3D 线速度 (m/s)
    double GetYawVelocity() const;           // 附带计算出的自旋角速度 (rad/s)

    // 【核心新增】：获取追踪器最后一次的绝对位置 (专防冷启动时的数据关联爆炸)
    Eigen::Vector3d GetLastPosition() const;

    // 【新增】：获取追踪器最后一次观测的时间戳，供清理器使用
    double GetLastTimestamp() const;

private:
    // 计算整个时间窗口内所有帧的重投影误差向量 (8N 维)
    Eigen::VectorXd CalculateResiduals(const Eigen::VectorXd& state);
    
    // 计算 8 维状态量的雅可比矩阵 (8N x 8)
    Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& state);

    // 计算带有 Huber 鲁棒核的总体代价值，抵抗极端噪点
    double ComputeHuberCost(const Eigen::VectorXd& residuals, double huber_delta, Eigen::MatrixXd& W);

private:
    int window_size_;
    cv::Mat K_, D_;
    Eigen::Matrix3d P_;

    // =======================================================
    // 【性能优化】：缓存展开相机内参与畸变参数，告别 OpenCV 调用开销！
    // =======================================================
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, p1_, p2_, k3_;

    std::deque<FrameObservation> window_; // 滑动窗口双端队列

    bool is_initialized_;

    // 核心优化状态量：8 维 
    // [0 - 3]: tx, ty, tz, yaw (代表最新帧时刻的物理状态)
    // [4 - 7]: vx, vy, vz, vyaw (代表窗口内的平均运动速度)
    Eigen::VectorXd state_; 
};

#endif // SLIDING_WINDOW_BA_HPP