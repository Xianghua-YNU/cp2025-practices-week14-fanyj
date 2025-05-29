#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程学生模板
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

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
    dx_dt = sigma * (y - x)
    dy_dt = r * x - y - x * z
    dz_dt = x * y - b * z
    return [dx_dt, dy_dt, dz_dt]

def solve_lorenz_equations(sigma=10.0, r=28.0, b=8/3,
                          x0=0.1, y0=0.1, z0=0.1,
                          t_span=(0, 50), dt=0.01):
    """
    求解洛伦兹方程
    
    返回:
        t: 时间点数组
        y: 解数组，形状为(3, n_points)
    """
    # 定义时间点
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # 求解ODE，明确传递所有参数
    sol = solve_ivp(
        fun=lambda t, state, s=sigma, r_val=r, b_val=b: lorenz_system(t, state, s, r_val, b_val),
        t_span=t_span,
        y0=[x0, y0, z0],
        t_eval=t_eval,
        method='RK45',
        dense_output=True
    )
    
    return sol.t, sol.y

def plot_lorenz_attractor(t: np.ndarray, y: np.ndarray):
    """
    绘制洛伦兹吸引子3D图
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    x, y, z = y
    ax.plot(x, y, z, color='blue', alpha=0.7, linewidth=0.8)
    
    # 设置标题和标签
    ax.set_title('洛伦兹吸引子', fontsize=15)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

def compare_initial_conditions(ic1, ic2, sigma=10.0, r=28.0, b=8/3, t_span=(0, 50), dt=0.01):
    """
    比较不同初始条件的解
    """
    # 求解两种初始条件下的方程
    t1, y1 = solve_lorenz_equations(sigma, r, b, *ic1, t_span, dt)
    t2, y2 = solve_lorenz_equations(sigma, r, b, *ic2, t_span, dt)
    
    # 计算欧氏距离
    distance = np.sqrt(np.sum((y1 - y2)**2, axis=0))
    
    # 创建图形
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 3D轨迹对比图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(y1[0], y1[1], y1[2], color='blue', alpha=0.7, linewidth=0.8, label=f'初始条件: {ic1}')
    ax1.plot(y2[0], y2[1], y2[2], color='red', alpha=0.7, linewidth=0.8, label=f'初始条件: {ic2}')
    ax1.set_title('不同初始条件下的轨迹对比', fontsize=13)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 2. 距离随时间变化图
    ax2 = fig.add_subplot(222)
    ax2.semilogy(t1, distance, color='green', linewidth=1.5)
    ax2.set_title('轨迹间的欧氏距离 (对数刻度)', fontsize=13)
    ax2.set_xlabel('时间')
    ax2.set_ylabel('距离')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 3. X分量随时间变化
    ax3 = fig.add_subplot(223)
    ax3.plot(t1, y1[0], color='blue', linewidth=0.8, label=f'X1 (初始条件: {ic1})')
    ax3.plot(t2, y2[0], color='red', linewidth=0.8, label=f'X2 (初始条件: {ic2})')
    ax3.set_title('X分量随时间变化', fontsize=13)
    ax3.set_xlabel('时间')
    ax3.set_ylabel('X值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 局部放大图
    ax4 = fig.add_subplot(224)
    mid_idx = len(t1) // 2
    ax4.plot(t1[mid_idx:], y1[0][mid_idx:], color='blue', linewidth=0.8)
    ax4.plot(t2[mid_idx:], y2[0][mid_idx:], color='red', linewidth=0.8)
    ax4.set_title('X分量局部放大图 (时间后半段)', fontsize=13)
    ax4.set_xlabel('时间')
    ax4.set_ylabel('X值')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
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
