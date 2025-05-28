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
    dx_dt = sigma * (y - x)
    dy_dt = r * x - y - x * z
    dz_dt = x * y - b * z
    return [dx_dt, dy_dt, dz_dt]


def solve_lorenz_equations(sigma=10.0, r=28.0, b=8 / 3,
                           x0=0.1, y0=0.1, z0=0.1,
                           t_span=(0, 50), dt=0.01):
    """
    求解洛伦兹方程

    返回:
        t: 时间点数组
        y: 解数组，形状为(3, n_points)
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    initial_state = [x0, y0, z0]
    # 调整了传递参数的方式，使用functools.partial或者lambda表达式来固定参数
    # 这里使用lambda表达式
    sol = solve_ivp(
        fun=lambda t, state: lorenz_system(t, state, sigma, r, b),
        t_span=t_span,
        y0=initial_state,
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
    x, y, z = y[0], y[1], y[2]

    # 绘制轨迹
    ax.plot(x, y, z, lw=0.5, color='blue')

    # 设置标题和标签
    ax.set_title('洛伦兹吸引子', fontsize=15)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    # 调整视角
    ax.view_init(elev=30, azim=130)

    plt.tight_layout()
    plt.show()


def compare_initial_conditions(ic1, ic2, t_span=(0, 50), dt=0.01):
    """
    比较不同初始条件的解
    """
    # 求解两组初始条件下的方程
    t1, y1 = solve_lorenz_equations(
        x0=ic1[0], y0=ic1[1], z0=ic1[2],
        t_span=t_span, dt=dt
    )

    t2, y2 = solve_lorenz_equations(
        x0=ic2[0], y0=ic2[1], z0=ic2[2],
        t_span=t_span, dt=dt
    )

    # 计算欧氏距离
    distance = np.sqrt(
        (y1[0] - y2[0]) ** 2 +
        (y1[1] - y2[1]) ** 2 +
        (y1[2] - y2[2]) ** 2
    )

    # 创建图形
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))

    # 绘制x分量随时间的变化
    ax1.plot(t1, y1[0], 'b-', label='IC1')
    ax1.plot(t2, y2[0], 'r-', label='IC2')
    ax1.set_ylabel('x')
    ax1.legend()
    ax1.set_title('x分量随时间的演化')

    # 绘制y分量随时间的变化
    ax2.plot(t1, y1[1], 'b-', label='IC1')
    ax2.plot(t2, y2[1], 'r-', label='IC2')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.set_title('y分量随时间的演化')

    # 绘制z分量随时间的变化
    ax3.plot(t1, y1[2], 'b-', label='IC1')
    ax3.plot(t2, y2[2], 'r-', label='IC2')
    ax3.set_ylabel('z')
    ax3.set_xlabel('时间')
    ax3.legend()
    ax3.set_title('z分量随时间的演化')

    # 绘制距离随时间的变化（对数尺度）
    ax4.semilogy(t1, distance, 'g-')
    ax4.set_ylabel('距离 (对数尺度)')
    ax4.set_xlabel('时间')
    ax4.set_title('两个轨迹之间的欧氏距离')

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
