#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程学生模板
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


def lorenz_system(t, state, sigma, r, b):
    """
    定义洛伦兹系统方程
    
    参数:
        t: 当前时间
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
        fun=lambda t, state: lorenz_system(t, state, sigma, r, b),
        t_span=t_span,
        y0=[x0, y0, z0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
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
    ax.set_title('Lorenz Attractor', fontsize=15)
    plt.tight_layout()
    plt.show()


def compare_initial_conditions(ic1: tuple[float, float, float], 
                              ic2: tuple[float, float, float], 
                              t_span: tuple[float, float] = (0, 50), dt: float = 0.01):
    """
    比较不同初始条件的解
    """
    t1, y1 = solve_lorenz_equations(x0=ic1[0], y0=ic1[1], z0=ic1[2], t_span=t_span, dt=dt)
    t2, y2 = solve_lorenz_equations(x0=ic2[0], y0=ic2[1], z0=ic2[2], t_span=t_span, dt=dt)
    
    # 计算轨迹距离
    distance = np.sqrt((y1[0]-y2[0])**2 + (y1[1]-y2[1])**2 + (y1[2]-y2[2])**2)
    
    # 绘制比较图
    plt.figure(figsize=(12, 6))
    plt.plot(t1, y1[0], 'b-', label=f'IC1: {ic1}')
    plt.plot(t2, y2[0], 'r-', label=f'IC2: {ic2}')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('X', fontsize=12)
    plt.title('Comparison of X(t) with Different Initial Conditions', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(t1, distance, 'g-', label='Distance between trajectories')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Distance (log scale)', fontsize=12)
    plt.title('Distance between Trajectories over Time', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
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
