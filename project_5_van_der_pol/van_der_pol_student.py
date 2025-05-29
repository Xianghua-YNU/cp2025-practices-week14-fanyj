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
    if not isinstance(state, np.ndarray) or len(state) != 2:
        raise TypeError("state should be a 2 - element numpy array")
    x, v = state
    dx_dt = v
    dv_dt = mu * (1 - x ** 2) * v - omega ** 2 * x
    return np.array([dx_dt, dv_dt])


def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格 - 库塔方法进行一步数值积分。

    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数

    返回:
        np.ndarray: 下一步的状态
    """
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = ode_func(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)

    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
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
    t_points = np.linspace(t_start, t_end, num_steps)
    state_dim = len(initial_state)
    states = np.zeros((num_steps, state_dim))
    states[0] = initial_state

    for i in range(num_steps - 1):
        states[i + 1] = rk4_step(ode_func, states[i], t_points[i], dt, **kwargs)

    return t_points, states


def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。

    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 5))
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
    return 0.5 * (v ** 2 + omega ** 2 * x ** 2)


def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。

    参数:
        states: np.ndarray, 状态数组

    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    x = states[:, 0]
    # 更稳健地寻找局部极大值，使用scipy.signal的argrelextrema函数
    from scipy.signal import argrelextrema
    local_maxima_indices = argrelextrema(x, np.greater)[0]
    if len(local_maxima_indices) < 2:
        print("警告: 未能找到足够的局部极大值来估计周期")
        return np.max(np.abs(x)), 0.0
    local_maxima = x[local_maxima_indices]
    # 振幅估计为平均局部极大值
    amplitude = np.mean(local_maxima)
    # 周期估计为相邻极大值之间的平均时间间隔
    dt = 0.01  # 假设时间步长为0.01
    periods = np.diff(local_maxima_indices) * dt
    period = np.mean(periods)
    return amplitude, period


def main():
    # 设置基本参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])

    # 任务1 - 基本实现
    mu = 1.0
    t_points, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t_points, states, f'van der Pol振子时间演化 (μ={mu})')

    # 任务2 - 参数影响分析
    mu_values = [0.1, 1.0, 3.0]
    plt.figure(figsize=(12, 8))

    for i, mu in enumerate(mu_values):
        _, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plt.subplot(len(mu_values), 1, i + 1)
        plt.plot(t_points, states[:, 0], label=f'位置 x (μ={mu})')
        plt.plot(t_points, states[:, 1], label=f'速度 v (μ={mu})')
        plt.xlabel('时间')
        plt.ylabel('状态')
        plt.legend()
        plt.grid(True)

    plt.suptitle('不同μ值下van der Pol振子的时间演化')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # 任务3 - 相空间分析
    for mu in mu_values:
        _, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'van der Pol振子相空间轨迹 (μ={mu})')

        # 分析极限环
        amplitude, period = analyze_limit_cycle(states)
        print(f'μ={mu}时的极限环特性: 振幅={amplitude:.4f}, 周期={period:.4f}')

    # 任务4 - 能量分析
    plt.figure(figsize=(10, 5))

    for mu in mu_values:
        _, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        energies = np.array([calculate_energy(state, omega) for state in states])
        plt.plot(t_points, energies, label=f'能量 (μ={mu})')

    plt.xlabel('时间')
    plt.ylabel('能量')
    plt.title('不同μ值下van der Pol振子的能量演化')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
