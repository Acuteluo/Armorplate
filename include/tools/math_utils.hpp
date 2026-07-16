#ifndef TOOLS_MATH_UTILS_HPP
#define TOOLS_MATH_UTILS_HPP

#include <cmath>

namespace tools
{

/**
 * @brief 弧度转角度
 */
inline double rad2deg(double rad) 
{ 
    return rad * 180.0 / M_PI; 
}

/**
 * @brief 角度转弧度
 */
inline double deg2rad(double deg) 
{ 
    return deg * M_PI / 180.0; 
}

/**
 * @brief 将弧度限制在 [-PI, PI] 之间 (就地修改)
 */
inline void limit_rad_inplace(double& rad) 
{
    while (rad > M_PI) rad -= 2.0 * M_PI;
    while (rad < -M_PI) rad += 2.0 * M_PI;
}

/**
 * @brief 将弧度限制在 [-PI, PI] 之间 (返回副本)
 */
inline double limit_rad(double rad) 
{
    limit_rad_inplace(rad);
    return rad;
}

} // namespace tools

#endif // TOOLS_MATH_UTILS_HPP