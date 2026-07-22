cmake_minimum_required(VERSION 3.10)
project(Armorplate)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 开启 Release 优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 寻找 OpenCV 库
# find_package(OpenCV REQUIRED)
# ===== OpenVINO GPU 测试开关 =====
set(OpenCV_DIR "/home/cly/opencv_openvino/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
# 寻找 Eigen3 库
find_package(Eigen3 REQUIRED)
# 寻找 spdlog
find_package(spdlog REQUIRED)

# 寻找线程库 (pthread)
set(THREADS_PREFER_PTHREAD_FLAG ON) 
find_package(Threads REQUIRED)

# 寻找 ROS2 相关的核心包
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)

# 包含目录
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${rosbag2_cpp_INCLUDE_DIRS}
)

# 生成可执行文件
add_executable(optimizer
    src/main.cpp 
    src/yolo/yolo_armor.cpp
    src/yolo/yolo_detector.cpp
    src/tools/logger.cpp
    src/tools/math_utils.cpp
    src/armor_corner_refiner.cpp
    src/sliding_window_ba.cpp
)

# 链接库
target_link_libraries(optimizer 
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES}
    Threads::Threads
    spdlog::spdlog
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${rosbag2_cpp_LIBRARIES}
)

install(DIRECTORY model DESTINATION share/${PROJECT_NAME})
