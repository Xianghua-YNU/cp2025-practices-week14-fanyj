#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程学生模板
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


def lorenz_system(state, sigma, r, b):
    """
    定义洛伦兹系统方程
    
    参数:
        state: 当前状态向量 [x, y, z]
        sigma, r, b: 系统参数
        
    返回:
        导数向量 [dx/dt, dy/dt, dz/dt]
    """
    dxdt = sigma * (state[1] - state[0])
    dydt = state[0] * (r - state[2]) - state[1]
    dzdt = state[0] * state[1] - b * state[2]
    return [dxdt, dydt, dzdt]


def solve_lorenz_equations(sigma=10.0, r=28.0, b=8/3,
                          x0=0.1, y0=0.1, z0=0.1,
                          t_span=(0, 50), dt=0.01):
    """
    求解洛伦兹方程
    
    返回:
        t: 时间点数组
        y: 解数组，形状为(3, n_points)
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lorenz_system, t_span, [x0, y0, z0], 
                   args=(sigma, r, b), t_eval=t_eval)
    return sol.t, sol.y.T


def plot_lorenz_attractor(t: np.ndarray, y: np.ndarray):
    """
    绘制洛伦兹吸引子3D图
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[0], y[1], y[2], lw=0.5)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Lorenz Attractor')
    ax.view_init(elev=30, azim=45)
    plt.show()


def compare_initial_conditions(ic1, ic2, t_span=(0,50), dt=0.01):
    """
    比较不同初始条件的解
    """
    # 求解两个初始条件
    t1, y1 = solve_lorenz_equations(ic1[0], ic1[1], ic1[2], t_span, dt)
    t2, y2 = solve_lorenz_equations(ic2[0], ic2[1], ic2[2], t_span, dt)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制两条轨迹
    ax.plot(y1[0], y1[1], y1[2], color='blue', label='IC1: {}'.format(ic1))
    ax.plot(y2[0], y2[1], y2[2], color='red', linestyle='--', 
            label='IC2: {}'.format(ic2))
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Comparison of Different Initial Conditions')
    ax.legend()
    plt.show()


def main():
    """
    主函数，执行所有任务
    """
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
