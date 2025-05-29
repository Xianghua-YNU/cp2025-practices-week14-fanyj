import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        mu: float, 非线性阻尼参数
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dx_dt = v
    dv_dt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dx_dt, dv_dt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    k1 = dt * ode_func(state, t, **kwargs)
    k2 = dt * ode_func(state + 0.5 * k1, t + 0.5 * dt, **kwargs)
    k3 = dt * ode_func(state + 0.5 * k2, t + 0.5 * dt, **kwargs)
    k4 = dt * ode_func(state + k3, t + dt, **kwargs)
    
    next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return next_state

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, num_steps)
    n_states = len(initial_state)
    states = np.zeros((num_steps, n_states))
    states[0] = initial_state
    
    for i in range(num_steps - 1):
        states[i+1] = rk4_step(ode_func, states[i], t[i], dt, **kwargs)
    
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], 'b-', label='位置 x')
    plt.plot(t, states[:, 1], 'r-', label='速度 v')
    plt.xlabel('时间')
    plt.ylabel('状态')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], 'g-')
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算van der Pol振子的能量。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        omega: float, 角频率
    
    返回:
        float: 系统的能量
    """
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。
    
    参数:
        states: np.ndarray, 状态数组
    
    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    # 找到所有局部极大值点
    x = states[:, 0]
    maxima_indices = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            maxima_indices.append(i)
    
    if len(maxima_indices) < 2:
        print("警告: 未能找到足够的局部极大值来计算周期")
        return np.max(np.abs(x)), 0
    
    # 计算振幅和周期
    amplitudes = [x[i] for i in maxima_indices]
    avg_amplitude = np.mean(amplitudes)
    
    # 假设dt已知，从第一个到最后一个极大值的平均时间间隔
    dt = 0.01  # 硬编码，因为solve_ode中没有返回dt
    periods = [(maxima_indices[i] - maxima_indices[i-1]) * dt for i in range(1, len(maxima_indices))]
    avg_period = np.mean(periods)
    
    return avg_amplitude, avg_period

def main():
    # 设置基本参数
    mu = 1.0
    omega = 1.0
    t_span = (0, 20)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # 任务1 - 基本实现
    # 1. 求解van der Pol方程
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    # 2. 绘制时间演化图
    plot_time_evolution(t, states, f'van der Pol振子时间演化 (μ={mu})')
    
    # 任务2 - 参数影响分析
    mu_values = [0.1, 1.0, 2.0, 5.0]
    plt.figure(figsize=(12, 8))
    for i, mu in enumerate(mu_values):
        _, states_mu = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plt.subplot(2, 2, i+1)
        plt.plot(t, states_mu[:, 0], 'b-', label='位置 x')
        plt.plot(t, states_mu[:, 1], 'r-', label='速度 v')
        plt.title(f'μ={mu}')
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 任务3 - 相空间分析
    # 1. 绘制相空间轨迹
    plot_phase_space(states, f'van der Pol振子相空间轨迹 (μ={mu})')
    # 2. 分析极限环特征
    amplitude, period = analyze_limit_cycle(states)
    print(f"极限环分析: 振幅 = {amplitude:.4f}, 周期 = {period:.4f}")
    
    # 任务4 - 能量分析
    # 1. 计算和绘制能量随时间的变化
    energies = np.array([calculate_energy(state, omega) for state in states])
    plt.figure(figsize=(10, 6))
    plt.plot(t, energies, 'm-')
    plt.xlabel('时间')
    plt.ylabel('能量')
    plt.title(f'van der Pol振子能量演化 (μ={mu})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 2. 分析能量的耗散和补充
    # 观察能量图可以看到，系统在极限环上能量会达到稳定的周期性变化

if __name__ == "__main__":
    main()
