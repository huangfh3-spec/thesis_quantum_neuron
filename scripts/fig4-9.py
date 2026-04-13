import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
import pandas as pd  # 新增：用于导出 CSV

# ====================== 全局参数 ========================
dim = 7  # 七能级系统
total_t = 160
delta = 0.040
Delta = -5.23 * 2 * np.pi  # 角频率
dt =4
N=total_t//dt
measure_times =int(total_t // dt + 1)
print(measure_times)
Gamma = 19.6 * 2 * np.pi  # 线频率
options = Options(nsteps=1500000)  # 增加积分步数
basis_states = [basis(dim, i) for i in range(dim)]
Dt = np.linspace(0, dt, 80)
tlist2 = np.linspace(0, total_t, measure_times)
gamma_real1 = [0]

beta = 0

V1 = []
I1 = []

V2 = []
I2 = []

V3 = []
I3 = []
t = []

Sx1 = [1]
Sy1 = [0]


Sx11 = []
Sy11 = []

b = Bloch()

P0 = [1/2]
P1 = [1/2]
P2 = [0]
P3 = [0]


# 量子比特算子定义
Sigx = basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
Sigy = -1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
Sigz = basis_states[1] * basis_states[1].dag() - basis_states[0] * basis_states[0].dag()
Iden= basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
Iden_excited=basis_states[2] * basis_states[2].dag() + basis_states[3] * basis_states[3].dag()+basis_states[4] * basis_states[4].dag()+basis_states[5] * basis_states[5].dag()+basis_states[6] * basis_states[6].dag()

#微波旋转算符定义  (theta 旋转角度  phi 旋转轴)
def U(theta , phi):
    return (np.cos(theta/2)*Iden-1j*np.sin(theta/2)*(np.cos(phi)*Sigx+np.sin(phi)*Sigy)+Iden_excited)



# ====================== 七能级系统定义 ====================
H_offset = (
    delta * basis_states[1] * basis_states[1].dag()
    + (-Delta) * basis_states[6] * basis_states[6].dag()
    + (Delta) * basis_states[4] * basis_states[4].dag()
    + delta * basis_states[5] * basis_states[5].dag()
)

H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()

H_structure2 = (
    basis_states[2] * basis_states[4].dag()
    + basis_states[4] * basis_states[2].dag()
    + basis_states[3] * basis_states[6].dag()
    + basis_states[6] * basis_states[3].dag()
)

J1 = [0]

# 耗散项 (示例，根据实际情况调整)
collapse_ops = [
    *[
        np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag()
        for k in [2, 1, 3]
    ],
    *[
        np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag()
        for k in [1, 2, 0]
    ],
    *[
        np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag()
        for k in [3, 0, 1]
    ],
]


def build_Hamiltonian(J1_val):
    return   J1_val * H_structure1 
tomo_x=[1]
tomo_y=[0.5]
J2=[]
states = (basis_states[1]+basis_states[0] ).unit()

for i in range(measure_times - 1): 
      #i 从0-159
    H2 = build_Hamiltonian(J1[i])
    result = mesolve(
        H2,
        states,
        Dt,
        collapse_ops,
        [
            Sigx,
            Sigy,
            Sigz,
            basis_states[0] * basis_states[0].dag(),
            basis_states[1] * basis_states[1].dag(),
            basis_states[2] * basis_states[2].dag(),
            basis_states[3] * basis_states[3].dag()
           
        ],
        options=options,
    )
    result1 = mesolve(H2, states, Dt, collapse_ops, [], options=options)
    
    b.add_points([Sx1[i], Sy1[i], result.expect[2][-1]])

    # #微波脉冲测量Sy，Sx
    rho_y= U(np.pi/2,-delta*4*(i+1))*result1.states[-1]*U(np.pi/2,-delta*4*(i+1)).dag()

    rho_x= U(np.pi/2,-delta*4*(i+1)-np.pi/2)*result1.states[-1]*U(np.pi/2,-delta*4*(i+1)-np.pi/2).dag()



    Sx1.append(rho_x[1,1]-rho_x[0,0])
    Sy1.append(rho_y[1,1]-rho_y[0,0])
       #/ (result.expect[3][-1] + result.expect[4][-1])

    # tomo_y.append((result.expect[1][-1]+1)/2)
    # tomo_x.append((result.expect[0][-1]+1)/2)


    
    P0.append(result.expect[3][-1])
    P1.append(result.expect[4][-1])
    P2.append(result.expect[5][-1])
    P3.append(result.expect[6][-1])

    gamma_real1.append(gamma_real1[i] +0.5 * Sy1[i] * dt * 0.0004)

    # J1.append(
    #     (
    #         3* gamma_real1[i + 1]* P1[i + 1]/ (4* Gamma* (2 * P1[i + 1] / (Gamma**2 )
    #                - beta**2 * (P2[i + 1] + P3[i + 1]) / (Gamma**2 + 4 * Delta**2)
    #            ))
    #     )
    #     ** 0.5
    
    # J1.append(0)
    J1.append(

         (3*(Gamma)*gamma_real1[i+1]/8)**0.5)
    
    
    



    states = result1.states[-1]
v_m=[]
# 截断到 measure_times 长度
t=[]   #160长度
t2=tlist2[:measure_times-1]
# J1 = J1[:measure_times-1]
# gamma_real1 = gamma_real1[:measure_times]
dy_dt=[]
I_m=[]
# 计算 I1, V1, aomrf, t
# print(len(Sx1))

# plt.plot(tlist2,Sy1, label='Sigy_J')

# plt.plot(tlist2, Sx1, label='Sigx_J')

# # =================== 新增：将波形导出为 CSV 文件 ===================
# # 构造 DataFrame，包含 time 两列
# Sigmay = pd.DataFrame({
#     'time': tlist2,   # 时间轴，从 0 到 total_t，长度为 measure_times
#     'sigmay': Sy1          # 对应的 J1 值
# })
# # 保存为 CSV（保存在当前工作目录下，文件名可自行修改）
# Sigmay.to_csv('Sigmay_waveform_20260213.csv', index=False)

# # 构造 DataFrame，包含 time 两列
# Sigmax = pd.DataFrame({
#     'time': tlist2,   # 时间轴，从 0 到 total_t，长度为 measure_times
#     'sigmay': Sx1          # 对应的 J1 值
# })
# # 保存为 CSV（保存在当前工作目录下，文件名可自行修改）
# Sigmax.to_csv('Sigmax_waveform_20260213.csv', index=False)


# for i in range(len(Sx1) - 1):
#     I_m.append(-0.75 * Sy1[i])
#     # V1.append(((Sy1[i + 1] - Sy1[i]) / dt - delta* Sx1[i]))
#     v_m.append(I_m[i]*gamma_real1[i])
#     dy_dt.append((Sy1[i + 1] - Sy1[i]) / dt)
#     J2.append(beta*J1[i])
#     t.append(tlist2[i])
# Sx1 = Sx1[:N]
# Sy1 = Sy1[:N]
# t=tlist2[:N]
# dy_dt_sim = (np.array(Sy1[2:]) - np.array(Sy1[:-2])) / (2*dt)
# Sy1 = Sy1[1:N-1]
# Sx1 = Sx1[1:N-1]
# t=tlist2[1:N-1]
# V1 = dy_dt_sim - 0.04 * np.array(Sx1)
# I1 = -0.75 * np.array(Sy1)
# # t_sim = tlist2[:40-1]
# # v_m=[I1[i]*gamma_real1[i] for i in range(len(I1))]

# # sigmax_sim = sigmax_sim[:n_points]
# # sigmay_sim = sigmay_sim[:n_points]
# # t_sim = t_sim[:n_points]

# # dy_dt_sim = (sigmay_sim[2:] - sigmay_sim[:-2]) / (2*step)
# # sigmay_sim = sigmay_sim[1:n_points-1]
# # sigmax_sim = sigmax_sim[1:n_points-1]
# # t_sim = t_sim[1:n_points-1]
# # V_sim = dy_dt_sim - 0.04 * sigmax_sim
# # I_sim = -0.75 * sigmay_sim

# # =================== 新增：将 J1 波形导出为 CSV 文件 ===================
# # 构造 DataFrame，包含 time 和 J1 两列
# df_J1 = pd.DataFrame({
#     'time': tlist2,   # 时间轴，从 0 到 total_t，长度为 measure_times
#     'J1': J1          # 对应的 J1 值
# })
# # 保存为 CSV（保存在当前工作目录下，文件名可自行修改）
# df_J1.to_csv('J1_waveform_20260113.csv', index=False)

# # print("已将 J1 波形保存到 J1_waveform.csv")

# # ====================== 绘图 ======================
# # 绘制布洛赫球
# b.point_size = [6]
# b.make_sphere()
# b.show()


# fig = plt.figure(figsize=(24, 12))

# plt.subplot(3, 2, 1)
# plt.plot(I1, V1, label='definition')
# plt.plot(I_m, v_m,  label=r'I$\gamma$')
# plt.ylabel('Voltage (V)')
# plt.xlabel('Current (I)')
# plt.legend()

# # plt.subplot(3, 2, 2)
# # plt.plot(tlist2, gamma_real1, 'r--', label=r'gamma1')
# # plt.legend()

# plt.subplot(3, 2, 3)
# # 创建阶梯状的J1数据 - 每个4us段保持常值
# J1_stair = []
# t_stair = []
# for i in range(len(J1)):
#     J1_stair.extend([J1[i], J1[i]])  # 每个值重复两次
#     if i < len(J1)-1:
#         t_stair.extend([tlist2[i], tlist2[i+1]])
#     else:
#         t_stair.extend([tlist2[i], tlist2[i] + dt])
# plt.plot(t_stair, J1_stair, linewidth=2, drawstyle='steps-post', label='J1 (staircase)')
# plt.xlabel('Time (us)')
# plt.ylabel('J1')
# plt.title('J1 Coupling Strength (4us steps)')
# plt.grid(True, alpha=0.3)
# plt.legend()


# # plt.subplot(3, 2, 4)
# # # plt.plot(t,Sy11, label='Sigy')

# # # plt.plot(t, Sx11, label='Sigx')


# # plt.xlabel('Time/us')
# # plt.legend()

# # plt.subplot(3, 2, 5)
# # plt.plot(t,dy_dt, label='sigmay_deriv')

# # # plt.plot(tlist2, P1, label='1 state')
# # plt.xlabel('Time/us')
# # plt.legend()


# # plt.show()

# # fig = plt.figure(figsize=(16, 12))
# # plt.subplot(3, 2, 6)
# # plt.plot(t,V1, label='V1')
# # plt.plot(t,v_m, label='v_m')
# # # plt.plot(tlist2, P1, label='1 state')
# # plt.xlabel('Time/us')
# plt.legend()


# plt.show()


import matplotlib as mpl
J1_stair = []

t_stair = []
plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 30,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
    'font.family': 'Arial',  # 改为Arial字体
    'mathtext.fontset': 'custom',  # 自定义数学字体
    'mathtext.it': 'Arial:italic',  # 数学斜体使用Arial
    'mathtext.bf': 'Arial:bold',    # 数学粗体使用Arial
    'mathtext.rm': 'Arial',         # 数学常规使用Arial
    'axes.linewidth': 2,
    'grid.alpha': 0.3
})
for i in range(len(J1)):
    J1_stair.extend([J1[i], J1[i]])  # 每个值重复两次
    if i < len(J1)-1:
        t_stair.extend([tlist2[i], tlist2[i+1]])
    else:
        t_stair.extend([tlist2[i], tlist2[i] + dt])
fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    # 8. 网格设置
ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)

# 9. 设置背景色
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 10. 调整刻度参数
ax.tick_params(axis='both', which='major', length=6, width=2, color='black')
ax.tick_params(axis='both', which='minor', length=4, width=1, color='black')

# 11. 设置刻度标签字体为Arial
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
    label.set_color('black')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')
    label.set_color('black')
for text in ax.yaxis.get_majorticklabels():
    text.set_fontname('Arial')
    text.set_color('black')
ax.plot(t_stair, J1_stair,color="#4A07C7", linewidth=3, drawstyle='steps-post')
ax.set_xlabel('Time (us)', fontweight='bold', fontname='Arial', labelpad=5, color='black')
ax.set_ylabel('J1(MHz)', fontweight='bold', fontname='Arial', labelpad=5, color='black')

ax.grid(True, alpha=0.3)
plt.show()
# save_path='J1_waveform.pdf'
# plt.savefig(save_path,
#             format='PDF',
#             bbox_inches='tight',
#             dpi=300,
#             facecolor='white',
#             edgecolor='none',
#             transparent=False)
# ax.legend()
