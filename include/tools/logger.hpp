#ifndef TOOLS_LOGGER_HPP
#define TOOLS_LOGGER_HPP

#include <spdlog/spdlog.h>
#include <memory>

namespace tools
{
    // 获取全局 logger 实例的单例接口
    std::shared_ptr<spdlog::logger> logger();
}

// ============================================================================
// 封装类似 ROS 的轻量级宏，支持 fmt 格式化，例如: LOG_INFO("耗时: {:.2f} ms", time)
// ============================================================================

#define LOG_DEBUG(...) tools::logger()->debug(__VA_ARGS__)
#define LOG_INFO(...)  tools::logger()->info(__VA_ARGS__)
#define LOG_WARN(...)  tools::logger()->warn(__VA_ARGS__)
#define LOG_ERROR(...) tools::logger()->error(__VA_ARGS__)

#endif // TOOLS_LOGGER_HPP