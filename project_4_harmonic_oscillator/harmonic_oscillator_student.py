import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List
from scipy.signal import find_peaks  # 添加这行导入find_peaks函数

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。

    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率

    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    return np.array([v, -omega**2 * x])

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。

    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率

    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    return np.array([v, -omega**2 * x**3])

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

    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

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
    states = np.zeros((num_steps, len(initial_state)))
    states[0] = initial_state

    for i in range(num_steps - 1):
        states[i + 1] = rk4_step(ode_func, states[i], t[i], dt, **kwargs)

    return t, states

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
    plt.xlabel('时间 t')
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
    plt.plot(states[:, 0], states[:, 1], 'k-')
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。

    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组

    返回:
        float: 估计的振动周期
    """
    x = states[:, 0]
    peaks, _ = find_peaks(x)

    if len(peaks) < 2:
        print("警告: 未找到足够的峰值来估计周期")
        return np.nan

    periods = np.diff(t[peaks])
    return np.mean(periods)

def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01

    # 任务1 - 简谐振子的数值求解
    initial_state = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_time_evolution(t, states, '简谐振子的时间演化')

    # 任务2 - 振幅对周期的影响分析
    amplitudes = [0.5, 1.0, 2.0, 5.0]
    periods = []

    for A in amplitudes:
        initial_state = np.array([A, 0.0])
        t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        periods.append(period)
        print(f"简谐振子 - 振幅: {A:.1f}, 估计周期: {period:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(amplitudes, periods, 'o-')
    plt.xlabel('振幅')
    plt.ylabel('周期')
    plt.title('简谐振子: 振幅对周期的影响')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 任务3 - 非谐振子的数值分析
    for A in amplitudes:
        initial_state = np.array([A, 0.0])
        t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        print(f"非谐振子 - 振幅: {A:.1f}, 估计周期: {period:.4f}")
        plot_time_evolution(t, states, f'非谐振子的时间演化 (振幅={A})')

    # 任务4 - 相空间分析
    initial_state = np.array([2.0, 0.0])

    # 简谐振子
    t, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_harmonic, '简谐振子的相空间轨迹')

    # 非谐振子
    t, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_anharmonic, '非谐振子的相空间轨迹')


if __name__ == "__main__":
    main()
