import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """
    受驱单摆的常微分方程
    state: [theta, omega]
    返回: [dtheta/dt, domega/dt]
    """
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g/l) * np.sin(theta) - C * omega + np.sin(Omega * t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    sol = solve_ivp(
        lambda t, y: forced_pendulum_ode(t, y, l, g, C, Omega),
        t_span, y0, method='RK45', 
        dense_output=True, t_eval=np.linspace(t_span[0], t_span[1], 1000)
    )
    return sol.t, sol.y[0]

def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0,200), y0=[0,0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    if Omega_range is None:
        Omega_range = np.linspace(0.1, 15, 50)
    
    amplitudes = []
    for Omega in Omega_range:
        t, theta = solve_pendulum(l, g, C, Omega, t_span, y0)
        # 取最后四分之一时间段内的振幅作为稳态振幅
        steady_state_theta = theta[int(len(theta)*3/4):]
        amplitude = (np.max(steady_state_theta) - np.min(steady_state_theta)) / 2
        amplitudes.append(amplitude)
    
    return Omega_range, amplitudes

def plot_results(t, theta, title):
    """绘制结果"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def plot_resonance_curve(Omega_range, amplitudes):
    """绘制共振曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range, amplitudes)
    plt.title('共振曲线: 振幅 vs 驱动频率')
    plt.xlabel('驱动频率 (rad/s)')
    plt.ylabel('振幅 (rad)')
    plt.grid(True)
    plt.show()

def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    t, theta = solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0])
    plot_results(t, theta, '受迫单摆运动 (Ω=5 rad/s)')
    
    # 任务2: 探究共振现象
    Omega_range, amplitudes = find_resonance(l=0.1, g=9.81, C=2, Omega_range=np.linspace(0.1, 15, 50), t_span=(0,200))
    plot_resonance_curve(Omega_range, amplitudes)
    
    # 找到共振频率并绘制共振情况
    resonance_idx = np.argmax(amplitudes)
    resonance_Omega = Omega_range[resonance_idx]
    print(f"共振频率: {resonance_Omega:.3f} rad/s")
    
    t_resonance, theta_resonance = solve_pendulum(l=0.1, g=9.81, C=2, Omega=resonance_Omega, t_span=(0,100))
    plot_results(t_resonance, theta_resonance, f'共振状态下的受迫单摆运动 (Ω={resonance_Omega:.3f} rad/s)')

if __name__ == '__main__':
    main()
