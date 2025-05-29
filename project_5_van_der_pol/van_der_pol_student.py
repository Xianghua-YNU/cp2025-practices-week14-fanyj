import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, List

def van_der_pol_ode(t, state, mu=1.0, omega=1.0):
    """van der Pol振子的一阶微分方程组。"""
    x, v = state
    return np.array([v, mu*(1-x**2)*v - omega**2*x])

def solve_ode(ode_func, initial_state, t_span, dt, **kwargs):
    """使用solve_ivp求解常微分方程组"""
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(ode_func, t_span, initial_state, 
                   t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """Plot the time evolution of states."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """Plot the phase space trajectory."""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """计算van der Pol振子的能量。"""
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray, dt: float) -> Tuple[float, float]:
    """分析极限环的特征（振幅和周期）。"""
    # 跳过初始瞬态，只分析稳定部分
    skip = int(len(states) * 0.3)
    x = states[skip:, 0]
    
    # 寻找峰值点
    peak_indices = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peak_indices.append(i)
    
    # 计算振幅（峰值的平均值）
    if not peak_indices:
        print("警告: 未能找到峰值点")
        return np.nan, np.nan
    
    amplitudes = [x[i] for i in peak_indices]
    amplitude = np.mean(amplitudes)
    
    # 计算周期（相邻峰值的平均时间间隔）
    if len(peak_indices) < 2:
        print("警告: 峰值点不足，无法计算周期")
        return amplitude, np.nan
    
    periods = [(peak_indices[i] - peak_indices[i-1]) * dt for i in range(1, len(peak_indices))]
    period = np.mean(periods)
    
    return amplitude, period

def plot_energy_evolution(t: np.ndarray, states: np.ndarray, omega: float = 1.0, title: str = None) -> None:
    """绘制能量随时间的演化。"""
    energies = np.array([calculate_energy(state, omega) for state in states])
    plt.figure(figsize=(10, 6))
    plt.plot(t, energies)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(title or 'Energy Evolution')
    plt.grid(True)
    plt.show()

def compare_mu_effect(mu_values: List[float], initial_state: np.ndarray, t_span: Tuple[float, float], dt: float, omega: float = 1.0) -> None:
    """比较不同mu值对系统行为的影响。"""
    plt.figure(figsize=(15, 10))
    
    for i, mu in enumerate(mu_values):
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        
        # 时间演化图
        plt.subplot(len(mu_values), 2, 2*i + 1)
        plt.plot(t, states[:, 0], label='Position')
        plt.plot(t, states[:, 1], label='Velocity')
        plt.title(f'μ = {mu}')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.grid(True)
        if i == 0:
            plt.legend()
        
        # 相空间图
        plt.subplot(len(mu_values), 2, 2*i + 2)
        plt.plot(states[:, 0], states[:, 1])
        plt.title(f'Phase Space (μ = {mu})')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.grid(True)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    # 设置基本参数
    mu = 2.0  # 增加mu值以显示更明显的非线性效应
    omega = 1.0
    t_span = (0, 50)  # 延长时间范围以观察稳定状态
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # 任务1 - 基本实现
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'van der Pol Oscillator Time Evolution (μ={mu})')
    
    # 任务2 - 参数影响分析
    mu_values = [0.5, 1.0, 2.0, 5.0]  # 测试不同的mu值
    compare_mu_effect(mu_values, initial_state, t_span, dt, omega)
    
    # 任务3 - 相空间分析和极限环特性
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'Phase Space Trajectory (μ={mu})')
        
        amplitude, period = analyze_limit_cycle(states, dt)
        print(f'μ = {mu}: 振幅 ≈ {amplitude:.3f}, 周期 ≈ {period:.3f}')
    
    # 任务4 - 能量分析
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_energy_evolution(t, states, omega, f'Energy Evolution (μ={mu})')

if __name__ == "__main__":
    main()
