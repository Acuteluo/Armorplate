#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <vector>
#include <string>

// --- rosbag2_cpp 解析依赖 ---
#include <rosbag2_cpp/reader.hpp>
#include <rclcpp/serialization.hpp>
#include <sensor_msgs/msg/image.hpp>

// --- 纯净算法依赖 ---
#include "tools/logger.hpp"
#include "tools/math_utils.hpp"
#include "yolo/yolo_detector.hpp"
#include "yolo/yolo_armor.hpp"
#include "armor_corner_refiner.hpp"

// ================= 全局共享数据与锁 =================
struct SharedData 
{
    cv::Mat current_frame;         // 拉流线程写入：刚从 rosbag 取出的原图
    cv::Mat display_frame;         // 视觉线程写入：画好框和打印好数据的渲染图
    cv::Mat refined_frame;         // 视觉线程写入：经过亚像素角点修正的放大图
    bool frame_updated = false;    // 标志位：是否有新图
};

SharedData g_data;
std::mutex g_mtx;
std::condition_variable g_cv;
std::atomic<bool> g_running{true}; // 线程运行标志位

// 敌方颜色 (假设考核视频里打蓝色 1)
const int ENEMY_COLOR = 1; 


// ================= 辅助函数 1：解析 全局路径配置文件 (args_config.yaml) =================
bool LoadArgsConfig(const std::string& yaml_path, 
                    std::string& model_path, 
                    std::string& camera_config_path, 
                    std::string& bag_path, 
                    std::string& topic_name)
{
    // 打开 YAML 文件进行读取
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) 
    {
        LOG_ERROR("无法打开全局参数配置文件: {}", yaml_path);
        return false;
    }

    try 
    {
        // 1. 读取 YOLO 模型路径
        cv::FileNode yolo_node = fs["YoloModel"];
        model_path = (std::string)yolo_node["Path"];

        // 2. 读取 相机内参配置文件 路径
        cv::FileNode cam_node = fs["CameraConfig"];
        camera_config_path = (std::string)cam_node["Path"];

        // 3. 读取 Rosbag 路径和解析话题名
        cv::FileNode bag_node = fs["Rosbag"];
        bag_path = (std::string)bag_node["Path"];
        topic_name = (std::string)bag_node["Topic"];

        LOG_INFO("成功加载 args_config.yaml 配置！");
        LOG_INFO("  > 模型路径: {}", model_path);
        LOG_INFO("  > 相机配置: {}", camera_config_path);
        LOG_INFO("  > Bag路径: {}", bag_path);
        LOG_INFO("  > 图像话题: {}", topic_name);
    }
    catch (const cv::Exception& e) 
    {
        LOG_ERROR("解析 args_config.yaml 文件出错，请检查字段名称是否拼写正确: {}", e.what());
        return false;
    }
    
    fs.release();
    return true;
}


// ================= 辅助函数 2：解析 相机内参文件 (camera_config.yaml) =================
bool LoadCameraConfig(const std::string& yaml_path, cv::Mat& K, cv::Mat& D)
{
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) 
    {
        LOG_ERROR("无法打开相机配置文件: {}", yaml_path);
        return false;
    }

    try 
    {
        // 提取 3x3 相机内参矩阵
        cv::FileNode k_node = fs["camera_matrix"]["data"];
        std::vector<double> k_vec;
        k_node >> k_vec;
        K = cv::Mat(k_vec).clone().reshape(1, 3); // 将 1D 数组重塑为 3x3 矩阵

        // 提取 1x5 畸变系数矩阵
        cv::FileNode d_node = fs["distortion_coefficients"]["data"];
        std::vector<double> d_vec;
        d_node >> d_vec;
        D = cv::Mat(d_vec).clone().reshape(1, 1); // 将 1D 数组重塑为 1x5 矩阵

        LOG_INFO("成功加载相机内参: 宽 {} x 高 {}", (int)fs["image_width"], (int)fs["image_height"]);
    }
    catch (const cv::Exception& e) 
    {
        LOG_ERROR("解析 相机内参 YAML 文件出错: {}", e.what());
        return false;
    }
    
    fs.release();
    return true;
}


// ================= 子线程 A：拉取 Rosbag 数据 (生产者) =================
void RosbagReadThread(const std::string& bag_path, const std::string& topic_name) 
{
    rosbag2_cpp::Reader reader;
    
    try 
    {
        reader.open(bag_path);
        LOG_INFO("[读取线程] 成功打开 rosbag: {}", bag_path);
    } 
    catch (const std::exception& e) 
    {
        LOG_ERROR("[读取线程] 无法打开 rosbag: {}", e.what());
        g_running = false;
        g_cv.notify_all(); // 通知主线程和其他线程退出
        return;
    }

    // 准备反序列化工具
    rclcpp::Serialization<sensor_msgs::msg::Image> serialization;

    while (g_running && reader.has_next()) 
    {
        auto bag_message = reader.read_next();

        // 仅处理我们关心的原图话题 (如 /image_raw)
        if (bag_message->topic_name == topic_name) 
        {
            // 1. 将底层 SQLite 读出的二进制乱码，反序列化为 ROS2 的 Image 消息格式
            sensor_msgs::msg::Image::SharedPtr msg = std::make_shared<sensor_msgs::msg::Image>();
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
            serialization.deserialize_message(&serialized_msg, msg.get());

            // 2. 将 ROS2 Image 内存映射为 OpenCV Mat
            cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data(), msg->step);
            cv::Mat frame_bgr;

            // 绝大多数相机的 ROS 图像格式是 rgb8，必须反转通道为 OpenCV 的 bgr8，否则红蓝板颜色反转
            if (msg->encoding == "rgb8") 
            {
                cv::cvtColor(frame, frame_bgr, cv::COLOR_RGB2BGR);
            } 
            else 
            {
                frame_bgr = frame.clone(); 
            }

            // 3. 将新帧送入共享内存 (锁步模型机制)
            {
                std::unique_lock<std::mutex> lock(g_mtx);
                
                // 【核心锁步机制】：一直等，直到视觉线程把上一张图用完 (frame_updated 变回 false)
                // 这样能绝对保证不会漏掉 rosbag 里的任何一帧，从而绘制出连续无断点的评估曲线！
                g_cv.wait(lock, [] { return !g_data.frame_updated || !g_running; });
                if (!g_running) break;

                // 塞入新图，拉响标志位
                g_data.current_frame = frame_bgr.clone();
                g_data.frame_updated = true;
            }
            // 唤醒处于休眠等待状态的视觉线程去干活
            g_cv.notify_one(); 
        }
    }

    LOG_INFO("[读取线程] Rosbag 播放结束。");
    g_running = false;
    g_cv.notify_all();
}


// ================= 子线程 B：视觉处理 (消费者) =================
void VisionThread(YoloDetector* detector, const cv::Mat& K, const cv::Mat& D) 
{
    cv::Mat process_frame;
    int frame_count = 0;
    auto fps_start_time = std::chrono::steady_clock::now();

    // 【新增】：实例化角点精化器
    ArmorCornerRefiner corner_refiner;

    while (g_running) 
    {
        {
            // 1. 等待读取线程送入新图片
            std::unique_lock<std::mutex> lock(g_mtx);
            g_cv.wait(lock, [] { return g_data.frame_updated || !g_running; });
            
            if (!g_running) break;
            
            // 深拷贝原图，防止在画框时污染底层数据
            process_frame = g_data.current_frame.clone();
            g_data.frame_updated = false; // 重置标志位
        }
        
        // 唤醒读取线程，告诉它：“我把图拿走了，你去读下一张吧！”
        g_cv.notify_one(); 

        if (!process_frame.empty()) 
        {
            cv::Mat img_show = {};
            cv::Mat img_refined = {};

            // 2. YOLO 推理 (计时开始)
            auto t0 = std::chrono::steady_clock::now();
            std::vector<YoloObject> detections = detector->Detect(process_frame, ENEMY_COLOR);
            auto t1 = std::chrono::steady_clock::now();
            
            // 3. 姿态解算：PnP 与 网格搜索姿态优化
            for (auto& obj : detections) 
            {
                std::vector<cv::Point2f> original_corners = obj.pts;
                cv::Vec4f left_middle_line(0,0,0,0), right_middle_line(0,0,0,0); // 存储左、右线
                bool refine_success = corner_refiner.Refine(process_frame, obj.pts, obj.color, left_middle_line, right_middle_line);
                
                if (!refine_success) 
                {
                    LOG_WARN("[Refiner] 角点精化失败，降级使用 YOLO 原始角点");
                    // 即使精化失败，obj.pts 仍保留原样，不会崩溃，系统具有鲁棒性
                }

                // 将从 YAML 动态读取的内参传入 Armor 对象，先传入原始点后传入精化后的点
                YoloArmor armor(obj.number, obj.color, obj.is_big, obj.prob, obj.box, original_corners, obj.pts, K, D);
                armor.SetArmorplateSize();
                armor.PrintDebugLog(false); // 关闭刷屏打印
                
                // 对原始点，执行解算 PNP IPPE
                armor.PNP(); 

                // 画出解算结果
                if (armor.pnp_success_) 
                {
                    armor.CalculateReprojectionError(armor.t_flu_, armor.R_flu_); // 计算原始 pnp 的重投影误差
                    
                    // 亚像素图，放大 40 倍！
                    img_refined = armor.DrawMagnifiedROI(process_frame, left_middle_line, right_middle_line, corner_refiner.GetDebugPoints(), 40);
                
                    img_show = armor.DrawAndPrintInfo(process_frame, "complex");
                }

                    
            }
            auto t2 = std::chrono::steady_clock::now();

            // 4. 将画好各种框和文字的图片，传回给主线程去渲染显示
            {
                std::lock_guard<std::mutex> lock(g_mtx);
                g_data.display_frame = img_show.clone();
                g_data.refined_frame = img_refined.clone();
            }

            // 5. 耗时与 FPS 统计 (每隔 1 秒打印一次，防止刷屏)
            double nn_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double pnp_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            
            ++frame_count;
            auto fps_current_time = std::chrono::steady_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(fps_current_time - fps_start_time).count();
            
            if (elapsed_seconds >= 0.50) 
            {
                double fps = frame_count / elapsed_seconds;
                // 这行日志是我们最关注的算法耗时评估指标！
                LOG_INFO("[视觉线程] 处理帧率: {:.1f} FPS | YOLO 耗时: {:.2f} ms | PnP+优化 耗时: {:.2f} ms", fps, nn_ms, pnp_ms);
                frame_count = 0;
                fps_start_time = fps_current_time;
            }

            // 【新增 可逆 控制速度】：真正的全局控速阀门！
            // 让视觉线程处理完后休息 30ms。由于 A 和 B 是锁步的，这会强制让整个 Rosbag 的读取速度降到约 30fps，且绝不漏帧！
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
        }
    }
}


// ================= 主线程：UI 渲染循环 =================
int main(int argc, char** argv) 
{
    // 配置全局 logger 输出等级
    tools::logger()->set_level(spdlog::level::debug);
    LOG_INFO("================= ACE 视觉考核测试系统 ==================");

    // 1. 定义存储路径变量的容器
    std::string args_config_path = "../args_config.yaml"; // 这是写死的入口文件路径
    std::string model_path, camera_config_path, bag_path, topic_name;

    // 2. 加载全局路径参数
    if (!LoadArgsConfig(args_config_path, model_path, camera_config_path, bag_path, topic_name)) 
    {
        LOG_ERROR("[main.cpp] args_config.yaml 解析失败！");
        return -1; // 路径解析失败，直接退出
    }

    // 3. 动态加载相机内参 (根据刚才读出的 camera_config_path)
    cv::Mat K, D;
    if (!LoadCameraConfig(camera_config_path, K, D)) 
    {
        LOG_ERROR("[main.cpp] camera_config.yaml 解析失败！");
        return -1; // 内参加载失败，直接退出
    }

    // 4. 初始化 YOLO 模型
    LOG_INFO("正在加载 YOLO 模型...");
    YoloDetector detector(model_path, true); // 是否使用 GPU 加速

    // 5. 启动双线程并行处理
    // 把内参 K 和 D 传给视觉线程，使其独立完成解算
    std::thread bag_thread(RosbagReadThread, bag_path, topic_name);
    std::thread vision_thread(VisionThread, &detector, K, D);

    // 6. UI 显示主循环
    while (g_running) 
    {
        cv::Mat show_frame_display;
        cv::Mat show_frame_refined;

        {
            // 安全地从共享内存获取渲染图
            std::lock_guard<std::mutex> lock(g_mtx);
            if (!g_data.display_frame.empty()) 
            {
                show_frame_display = g_data.display_frame.clone();
                show_frame_refined = g_data.refined_frame.clone();
            }
        }

        if (!show_frame_display.empty()) 
        {
            cv::imshow("ACE Vision - Optimization Test", show_frame_display);
            cv::imshow("Refined", show_frame_refined);

            // std::vector<cv::Mat> bgr;
            // cv::split(show_frame, bgr);

            // cv::Mat blue  = bgr[0];   // 蓝通道
            // cv::Mat green = bgr[1];
            // cv::Mat red   = bgr[2];   // 红通道

            // // 显示
            // cv::imshow("Blue Channel", blue);
            // cv::imshow("Red Channel", red);
            
            // waitKey(1) 仅用于刷新 OpenCV 的 GUI 缓冲区和监听键盘事件。
            // 因为真正的画面更新频率已经被 VisionThread 里的 sleep 限制死了，所以这里的 1 只是保证画面极度流畅，不会产生抽帧。
            char key = (char)cv::waitKey(1); 
            if (key == 27 || key == 'q' || key == 'Q') 
            {
                LOG_WARN("收到退出指令...");
                g_running = false;
                g_cv.notify_all(); // 叫醒所有还在 wait 的线程
                break;
            }
            else if (key == ' ')
            {
                g_running = false;
                char wait_key = (char)cv::waitKey(10000000);
            }
        }
        else
        {
            // 如果还没图，稍微休息下，防止吃满单核 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // 优雅退出清理区
    g_running = false;
    g_cv.notify_all();

    if (bag_thread.joinable()) bag_thread.join();
    if (vision_thread.joinable()) vision_thread.join();

    cv::destroyAllWindows();
    LOG_INFO("====================== 系统已安全退出 ======================");
    return 0;
}