#include "tools/logger.hpp"

#include <fmt/chrono.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>
#include <chrono>

namespace tools
{
std::shared_ptr<spdlog::logger> logger_ = nullptr;

void set_logger()
{
    // 1. 确保日志文件夹存在 (C++17)，防止 spdlog 找不到目录抛出异常崩溃
    std::string log_dir = "logs";
    if (!std::filesystem::exists(log_dir)) 
    {
        std::filesystem::create_directory(log_dir);
    }

    // 2. 配置文件输出 Sink
    auto file_name = fmt::format("{}/ACE_Vision_{:%Y-%m-%d_%H-%M-%S}.log", log_dir, std::chrono::system_clock::now());
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name, true);
    file_sink->set_level(spdlog::level::debug);
    
    // 文件格式: [时间] [级别] 内容
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); 

    // 3. 配置控制台颜色输出 Sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); 

    // 🌟 定制颜色 (白, 绿, 黄, 红)
    console_sink->set_color(spdlog::level::debug, console_sink->white);
    console_sink->set_color(spdlog::level::info,  console_sink->green);
    console_sink->set_color(spdlog::level::warn,  console_sink->yellow_bold);
    console_sink->set_color(spdlog::level::err,   console_sink->red_bold);

    // 4. 组合 Sinks 并注册
    logger_ = std::make_shared<spdlog::logger>("ACE_Logger", spdlog::sinks_init_list{file_sink, console_sink});
    
    // 设置全局最低触发级别和自动 flush 级别
    logger_->set_level(spdlog::level::debug);
    logger_->flush_on(spdlog::level::info); 

    // 设置为 spdlog 的默认 logger (可选，但推荐)
    spdlog::set_default_logger(logger_);
}

std::shared_ptr<spdlog::logger> logger()
{
    // 懒汉式单例初始化
    if (!logger_) 
    {
        set_logger();
    }
    return logger_;
}

} // namespace tools