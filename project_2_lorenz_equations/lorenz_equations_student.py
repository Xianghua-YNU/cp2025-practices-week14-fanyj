#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程学生模板
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

def lorenz_system(state: np.ndarray, sigma: float, r: float, b: float) -> np.ndarray:
    """
    定义洛伦兹系统方程
    
    参数:
        state: 当前状态向量 [x, y, z]
        sigma, r, b: 系统参数
        
    返回:
        导数向量 [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    return np.array([
        sigma * (y - x),
        r * x - y - x * z,
        x * y - b * z
    ])

def solve_lorenz_equations(sigma: float = 10.0, r: float = 28.0, b: float = 8/3,
                          x0: float = 0.1, y0: float = 0.1, z0: float = 0.1,
                          t_span: tuple[float, float] = (0, 50), dt: float = 0.01):
    """
    求解洛伦兹方程
    
    返回:
        t: 时间点数组
        y: 解数组，形状为(3, n_points)
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lambda t, state: lorenz_system(state, sigma, r, b),
        t_span, 
        [x0, y0, z0], 
        t_eval=t_eval, 
        method='RK45',
        rtol=1e-6,  # 相对误差控制
        atol=1e-9   # 绝对误差控制
    )
    return sol.t, sol.y

def plot_lorenz_attractor(t: np.ndarray, y: np.ndarray):
    """
    绘制洛伦兹吸引子
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[0], y[1], y[2], lw=0.5, color='steelblue')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('洛伦兹吸引子', fontsize=15)
    plt.tight_layout()
    plt.show()

def compare_initial_conditions(ic1: tuple[float, float, float], 
                              ic2: tuple[float, float, float], 
                              sigma: float = 10.0, r: float = 28.0, b: float = 8/3,
                              t_span: tuple[float, float] = (0, 50), dt: float = 0.01):
    """
    比较不同初始条件的解
    """
    # 求解两个初始条件下的轨迹
    t1, y1 = solve_lorenz_equations(sigma, r, b, ic1[0], ic1[1], ic1[2], t_span, dt)
    t2, y2 = solve_lorenz_equations(sigma, r, b, ic2[0], ic2[1], ic2[2], t_span, dt)
    
    # 计算轨迹间的欧氏距离
    distance = np.sqrt((y1[0]-y2[0])**2 + (y1[1]-y2[1])**2 + (y1[2]-y2[2])**2)
    
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制X分量随时间的变化
    ax1.plot(t1, y1[0], 'b-', label=f'初始条件1: {ic1}')
    ax1.plot(t2, y2[0], 'r-', label=f'初始条件2: {ic2}')
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('X值', fontsize=12)
    ax1.set_title('不同初始条件下X分量随时间的变化', fontsize=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 绘制轨迹间距离随时间的变化
    ax2.plot(t1, distance, 'g-', label='轨迹间距离')
    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('距离', fontsize=12)
    ax2.set_title('轨迹间距离随时间的变化', fontsize=15)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_yscale('log')  # 使用对数刻度更清晰地显示指数增长
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数，执行所有任务
    """
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 任务A: 求解洛伦兹方程
    t, y = solve_lorenz_equations()
    
    # 任务B: 绘制洛伦兹吸引子
    plot_lorenz_attractor(t, y)
    
    # 任务C: 比较不同初始条件
    ic1 = (0.1, 0.1, 0.1)
    ic2 = (0.10001, 0.1, 0.1)  # 微小变化
    compare_initial_conditions(ic1, ic2)

if __name__ == '__main__':
    main()
