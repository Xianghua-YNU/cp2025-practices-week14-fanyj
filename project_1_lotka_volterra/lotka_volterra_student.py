#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：[请填写您的姓名]
学号：[请填写您的学号]
完成日期：[请填写完成日期]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float, 
                          gamma: float, delta: float) -> np.ndarray:
    """
    Lotka-Volterra方程组的右端函数
    
    方程组：
    dx/dt = α*x - β*x*y  (猎物增长率 - 被捕食率)
    dy/dt = γ*x*y - δ*y  (捕食者增长率 - 死亡率)
    
    参数:
        state: np.ndarray, 形状为(2,), 当前状态向量 [x, y]
        t: float, 时间（本系统中未显式使用，但保持接口一致性）
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率
    
    返回:
        np.ndarray, 形状为(2,), 导数向量 [dx/dt, dy/dt]
    """
    x, y = state
    
    # 实现Lotka-Volterra方程组
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    
    return np.array([dxdt, dydt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    欧拉法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数，签名为 f(y, t, *args)
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # 实现欧拉法迭代
    for i in range(n_steps - 1):
        y[i+1] = y[i] + dt * f(y[i], t[i], *args)
    
    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    改进欧拉法（2阶Runge-Kutta法）求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # 实现改进欧拉法
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + k2) / 2
    
    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], 
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶龙格-库塔法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # 实现4阶龙格-库塔法
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1/2, t[i] + dt/2, *args)
        k3 = dt * f(y[i] + k2/2, t[i] + dt/2, *args)
        k4 = dt * f(y[i] + k3, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y


def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                        x0: float, y0: float, t_span: Tuple[float, float], 
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用4阶龙格-库塔法求解Lotka-Volterra方程组
    
    参数:
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率
        x0: float, 初始猎物数量
        y0: float, 初始捕食者数量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
    
    返回:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量数组
        y: np.ndarray, 捕食者种群数量数组
    """
    # 调用runge_kutta_4函数求解方程组
    y0_vec = np.array([x0, y0])
    t, y = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x = y[:, 0]
    y = y[:, 1]
    
    return t, x, y


def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                   x0: float, y0: float, t_span: Tuple[float, float], 
                   dt: float) -> dict:
    """
    比较三种数值方法求解Lotka-Volterra方程组
    
    参数:
        alpha, beta, gamma, delta: 模型参数
        x0, y0: 初始条件
        t_span: 时间范围
        dt: 时间步长
    
    返回:
        dict: 包含三种方法结果的字典，格式为：
        {
            'euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'improved_euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'rk4': {'t': t_array, 'x': x_array, 'y': y_array}
        }
    """
    # 使用三种方法求解并返回结果字典
    y0_vec = np.array([x0, y0])
    
    # 欧拉法
    t_euler, y_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_euler = y_euler[:, 0]
    y_predator_euler = y_euler[:, 1]
    
    # 改进欧拉法
    t_improved, y_improved = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_improved = y_improved[:, 0]
    y_predator_improved = y_improved[:, 1]
    
    # 4阶龙格-库塔法
    t_rk4, y_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_rk4 = y_rk4[:, 0]
    y_predator_rk4 = y_rk4[:, 1]
    
    results = {
        'euler': {'t': t_euler, 'x': x_euler, 'y': y_predator_euler},
        'improved_euler': {'t': t_improved, 'x': x_improved, 'y': y_predator_improved},
        'rk4': {'t': t_rk4, 'x': x_rk4, 'y': y_predator_rk4}
    }
    
    return results


def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray, 
                           title: str = "Lotka-Volterra种群动力学") -> None:
    """
    绘制种群动力学图
    
    参数:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量
        y: np.ndarray, 捕食者种群数量
        title: str, 图标题
    """
    # 绘制两个子图
    plt.figure(figsize=(12, 5))
    
    # 子图1：时间序列图
    plt.subplot(1, 2, 1)
    plt.plot(t, x, 'b-', label='猎物')
    plt.plot(t, y, 'r-', label='捕食者')
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title('种群数量随时间变化')
    plt.legend()
    plt.grid(True)
    
    # 子图2：相空间轨迹图
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'g-')
    plt.xlabel('猎物数量')
    plt.ylabel('捕食者数量')
    plt.title('相空间轨迹')
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=15)
    plt.show()


def plot_method_comparison(results: dict) -> None:
    """
    绘制不同数值方法的比较图
    
    参数:
        results: dict, compare_methods函数的返回结果
    """
    # 绘制方法比较图
    plt.figure(figsize=(15, 10))
    
    methods = ['euler', 'improved_euler', 'rk4']
    method_names = ['欧拉法', '改进欧拉法', '4阶龙格-库塔法']
    
    for i, method in enumerate(methods, 1):
        t = results[method]['t']
        x = results[method]['x']
        y = results[method]['y']
        
        # 上排：时间序列图
        plt.subplot(2, 3, i)
        plt.plot(t, x, 'b-', label='猎物')
        plt.plot(t, y, 'r-', label='捕食者')
        plt.xlabel('时间')
        plt.ylabel('种群数量')
        plt.title(f'{method_names[i-1]} - 种群数量随时间变化')
        plt.legend()
        plt.grid(True)
        
        # 下排：相空间图
        plt.subplot(2, 3, i+3)
        plt.plot(x, y, 'g-')
        plt.xlabel('猎物数量')
        plt.ylabel('捕食者数量')
        plt.title(f'{method_names[i-1]} - 相空间轨迹')
        plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle('不同数值方法求解Lotka-Volterra方程组的比较', y=1.02, fontsize=15)
    plt.show()


def analyze_parameters() -> None:
    """
    分析不同参数对系统行为的影响
    
    分析内容：
    1. 不同初始条件的影响
    2. 守恒量验证
    """
    # 设置基本参数
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    # 测试不同初始条件
    initial_conditions = [(2.0, 2.0), (3.0, 1.0), (1.0, 3.0)]
    
    plt.figure(figsize=(15, 10))
    
    # 绘制不同初始条件下的相空间轨迹
    for i, (x0, y0) in enumerate(initial_conditions, 1):
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        
        plt.subplot(2, 2, i)
        plt.plot(x, y, 'g-')
        plt.xlabel('猎物数量')
        plt.ylabel('捕食者数量')
        plt.title(f'初始条件: x0={x0}, y0={y0}')
        plt.grid(True)
        
        # 计算并验证守恒量
        C = delta * np.log(x) - gamma * x + beta * y - alpha * np.log(y)
        mean_C = np.mean(C)
        std_C = np.std(C)
        print(f"初始条件: x0={x0}, y0={y0}")
        print(f"守恒量平均值: {mean_C:.6f}")
        print(f"守恒量标准差: {std_C:.6f}")
        print(f"相对波动: {std_C/mean_C*100:.4f}%\n")
    
    # 绘制守恒量随时间的变化
    plt.subplot(2, 2, 4)
    for x0, y0 in initial_conditions:
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        C = delta * np.log(x) - gamma * x + beta * y - alpha * np.log(y)
        plt.plot(t, C, label=f'x0={x0}, y0={y0}')
    
    plt.xlabel('时间')
    plt.ylabel('守恒量 C')
    plt.title('不同初始条件下守恒量随时间的变化')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle('不同初始条件对系统行为的影响及守恒量验证', y=1.02, fontsize=15)
    plt.show()


def main():
    """
    主函数：演示Lotka-Volterra模型的完整分析
    
    执行步骤：
    1. 设置参数并求解基本问题
    2. 比较不同数值方法
    3. 分析参数影响
    4. 输出数值统计结果
    """
    # 参数设置（根据题目要求）
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")
    
    try:
        # 1. 基本求解
        print("\n1. 使用4阶龙格-库塔法求解...")
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_population_dynamics(t, x, y, "Lotka-Volterra模型 - 4阶龙格-库塔法求解")
        
        # 2. 方法比较
        print("\n2. 比较不同数值方法...")
        results = compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_method_comparison(results)
        
        # 3. 参数分析
        print("\n3. 分析参数影响...")
        analyze_parameters()
        
        # 4. 数值结果统计
        print("\n4. 数值结果统计:")
        # 计算统计数据
        methods = ['euler', 'improved_euler', 'rk4']
        method_names = ['欧拉法', '改进欧拉法', '4阶龙格-库塔法']
        
        for method in methods:
            x = results[method]['x']
            y = results[method]['y']
            t = results[method]['t']
            
            # 计算周期（通过峰值检测）
            x_peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
            if len(x_peaks) > 1:
                periods = np.diff(t[x_peaks])
                avg_period = np.mean(periods)
            else:
                avg_period = float('nan')
            
            # 计算振幅
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            x_amplitude = (x_max - x_min) / 2
            y_amplitude = (y_max - y_min) / 2
            
            # 计算平均值
            x_avg = np.mean(x)
            y_avg = np.mean(y)
            
            # 计算相对误差（以RK4为参考）
            if method != 'rk4':
                x_error = np.mean(np.abs(x - results['rk4']['x'])) / np.mean(results['rk4']['x']) * 100
                y_error = np.mean(np.abs(y - results['rk4']['y'])) / np.mean(results['rk4']['y']) * 100
            else:
                x_error = 0
                y_error = 0
            
            print(f"\n{method_names[methods.index(method)]}统计结果:")
            print(f"  猎物数量平均值: {x_avg:.4f}, 振幅: {x_amplitude:.4f}")
            print(f"  捕食者数量平均值: {y_avg:.4f}, 振幅: {y_amplitude:.4f}")
            print(f"  估计周期: {avg_period:.4f} (如果可用)")
            print(f"  与RK4方法的相对误差: 猎物={x_error:.2f}%, 捕食者={y_error:.2f}%")
            
            # 计算守恒量
            C = delta * np.log(x) - gamma * x + beta * y - alpha * np.log(y)
            C_mean = np.mean(C)
            C_std = np.std(C)
            print(f"  守恒量平均值: {C_mean:.6f}, 标准差: {C_std:.6f}, 相对波动: {C_std/C_mean*100:.4f}%")
            
    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请完成相应函数的实现后再运行主程序。")


if __name__ == "__main__":
    main()
