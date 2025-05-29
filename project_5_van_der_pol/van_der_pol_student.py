import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, List

def van_der_pol_ode(t: float, state: np.ndarray, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        t: float, 当前时间
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        mu: float, 非线性阻尼参数
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dx_dt = v
    dv_dt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dx_dt, dv_dt])

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用scipy.integrate.solve_ivp求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数，形式为 func(t, state, *args)
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(
        fun=ode_func,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        args=tuple(kwargs.values()),
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    return sol.t, sol.y.T

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], 'b-', label='位置 x(t)')
    plt.plot(t, states[:, 1], 'r-', label='速度 v(t)')
    plt.xlabel('时间 t')
    plt.ylabel('状态变量')
    plt.title(title)
    plt.grid(True)
    plt.legend()
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
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def analyze_limit_cycle(states: np.ndarray, dt: float) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。
    
    参数:
        states: np.ndarray, 状态数组
        dt: float, 时间步长
    
    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    # 跳过初始瞬态（取后50%的数据）
    skip = int(len(states) * 0.5)
    x = states[skip:, 0]
    
    # 寻找局部极大值
    peaks = []
    peak_indices = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(x[i])
            peak_indices.append(i)
    
    # 计算振幅（峰值的平均值）
    amplitude = np.mean(peaks) if peaks else 0.0
    
    # 计算周期（相邻峰值之间的平均时间间隔）
    if len(peak_indices) >= 2:
        periods = np.diff(peak_indices) * dt
        period = np.mean(periods)
    else:
        period = 0.0
    
    return amplitude, period

def main():
    # 设置基本参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # 任务1 - 基本实现
    mu = 1.0
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'van der Pol振子时间演化 (μ={mu})')
    
    # 任务2 - 参数影响分析
    mu_values = [1.0, 2.0, 4.0]
    plt.figure(figsize=(12, 8))
    
    for i, mu in enumerate(mu_values):
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plt.subplot(len(mu_values), 1, i+1)
        plt.plot(t, states[:, 0], label=f'位置 x (μ={mu})')
        plt.plot(t, states[:, 1], label=f'速度 v (μ={mu})')
        plt.xlabel('时间')
        plt.ylabel('状态')
        plt.legend()
        plt.grid(True)
        
        # 分析极限环
        amplitude, period = analyze_limit_cycle(states, dt)
        print(f'μ={mu}时的极限环特性: 振幅={amplitude:.4f}, 周期={period:.4f}')
    
    plt.suptitle('不同μ值下van der Pol振子的时间演化')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # 任务3 - 相空间分析
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'van der Pol振子相空间轨迹 (μ={mu})')

if __name__ == "__main__":
    main()
