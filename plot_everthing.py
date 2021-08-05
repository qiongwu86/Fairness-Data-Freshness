import numpy as np
import os
import vehicle_numbers as vn
from avg_bo_tr_time import *
from Ki_compute import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from pylab import *         #支持中文
plt.rcParams['font.sans-serif']=['SimSun'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 载入数据
def loadDatadet(infile,k):
	f = open(infile,'r')
	sourceInLine = f.readlines()
	dataset = []
	for line in sourceInLine:
		temp1 = line.strip('\n')
		temp2 = temp1.split('\t')
		dataset.append(temp2)
	for i in range(0,len(dataset)):
		for j in range(k):
			dataset[i].append(float(dataset[i][j]))
		del(dataset[i][0:k])
	return np.array(dataset)

N = 2 # lane number
l = 5 # vehicle length
s0 = 2 # minimum intra-platoon spacing
v0 = 30 # v_max
Th = 1.6 # time headwayk
mv = 2 # platoon average vehicle numbers
D = 200 # rsu range
v_avg,W0_avg = 24.5,128
v_date = 2 # lane speed difference

SlotTime = 50*(1e-06) # Tslot
Ts = 8972*(1e-06)
V_range = np.arange(22.5,26.5,0.5)

len_V = len(V_range)
Kindex_avg = [compute_Kindex(v_avg,W0_avg)]*len_V

path_W0 = 'img_txt/data/content_window_1.txt'
path_W0_ = 'img_txt/data/content_window_2.txt'
k_W0 = 2
W0 = (loadDatadet(path_W0,k_W0)).reshape(-1,2)
W0_ = (loadDatadet(path_W0_,k_W0)).reshape(-1,2)

# --------------------------------------------------------------------------------------------------
# Compute lambda rate and vehicle number
Lambdas_1, Lambdas_075 = [],[]
Nv_1, Nv_075 = [],[] 
lanmuda_weight_1, lanmuda_weight_075 = 1,0.75

for v in np.arange(22.5,26.5,0.5):
	V = [v+v_date,v-v_date] # Vehicle speed in lane1 and lane2(lane1>lane2).
	lanmudas, ni = [], []
	for i in range(0,N): # Calculate the platoon arrival rate and the number of lane vehicles.
		lanmuda_max = vn.compute_lanmuda_max(s0,v0,V[i],Th,mv,l)
		lanmuda = lanmuda_weight_1*lanmuda_max
		lanmudas.append(lanmuda) # Calculate the maximum arrival rate of each lane.
		n_i = vn.compute_vehicle_numbers(s0,v0,V[i],Th,lanmudas[i],mv,l,D) # Calculate the number of vehicles in each lane at the corresponding speed.
		ni.append(n_i)
	Lambdas_1.append(lanmudas)
	Nv_1.append(ni)

for v in np.arange(22.5,26.5,0.5):
	V = [v+v_date,v-v_date] # Vehicle speed in lane1 and lane2(lane1>lane2).
	lanmudas, ni = [], []
	for i in range(0,N): # Calculate the platoon arrival rate and the number of lane vehicles.
		lanmuda_max = vn.compute_lanmuda_max(s0,v0,V[i],Th,mv,l)
		lanmuda = lanmuda_weight_075*lanmuda_max
		lanmudas.append(lanmuda) # Calculate the maximum arrival rate of each lane.
		n_i = vn.compute_vehicle_numbers(s0,v0,V[i],Th,lanmudas[i],mv,l,D) # Calculate the number of vehicles in each lane at the corresponding speed.
		ni.append(n_i)
	Lambdas_075.append(lanmudas)
	Nv_075.append(ni)

# age = []
# for i in range(len_V):
# 	aoi = compute_age(W0_[i],SlotTime,N,Nv_075[i],Ts)*1000
# 	age.append(aoi)
# print(age)

Lambdas_1 = np.array(Lambdas_1)
Lambdas_075 = np.array(Lambdas_075)
Nv_1 = np.array(Nv_1)
Nv_075 = np.array(Nv_075)

print(Lambdas_1)
print(Nv_1)
print(Lambdas_075)
print(Nv_075)

# Platoon arrival rate
plt.plot(V_range, Lambdas_1[:,0], color='#1f77b4', linestyle='--', marker='*', label='车道1，最大车队到达率')
plt.plot(V_range, Lambdas_1[:,1], color='#FF0000', linestyle='--', marker='+', label='车道2,最大车队到达率')
plt.plot(V_range, Lambdas_075[:,0], color='blueviolet', linestyle=':', marker='d', label='lane1,0.75*Lambda_max')
plt.plot(V_range, Lambdas_075[:,1], color='gold', linestyle=':', marker='p', label='lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Kindex1_difference')
plt.ylabel('Kindex1_difference')
plt.show()
# The number of vehicles
plt.plot(V_range, Nv_1[:,0], color='#1f77b4', linestyle='--', marker='*', label='lane1,Lambda_max')
plt.plot(V_range, Nv_1[:,1], color='#FF0000', linestyle='--', marker='+', label='lane2,Lambda_max')
plt.plot(V_range, Nv_075[:,0], color='blueviolet', linestyle=':', marker='d', label='lane1,0.75*Lambda_max')
plt.plot(V_range, Nv_075[:,1], color='gold', linestyle=':', marker='p', label='lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('optimal solution')
plt.ylabel('The number of vehicles')
plt.show()
# Total number of vehicles in the network
N_t1 = np.sum(Nv_1,axis=1)
N_t075 = np.sum(Nv_075,axis=1)
plt.plot(V_range, N_t1, color='#1f77b4', linestyle='--', marker='*', label='Lambda_max')
plt.plot(V_range, N_t075, color='#FF0000', linestyle='--', marker='+', label='0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Average velocity of two lanes')
plt.ylabel('The number of vehicles')
plt.show()
# --------------------------------------------------------------------------------------------------
# Compute Sv
sv1_1, sv2_1 = [], []
sv1_075, sv2_075 = [], []
# The Sv of lambda = max
for v in V_range:
	v_1 = v + v_date
	v_2 = v - v_date
	sv1 = vn.compute_sv(s0,v0,v_1,Th)
	sv2 = vn.compute_sv(s0,v0,v_2,Th)
	sv1_1.append(sv1)
	sv2_1.append(sv2)
# The Sv of lambda = 0.75*max
for v in V_range:
	v_1 = v + v_date
	v_2 = v - v_date
	sv1 = vn.compute_sv(s0,v0,v_1,Th)
	sv2 = vn.compute_sv(s0,v0,v_2,Th)
	sv1_075.append(sv1)
	sv2_075.append(sv2)

# The intra-platoon spacing
plt.plot(V_range, sv1_1, color='#1f77b4', linestyle='--', marker='*', label='lane1,Lambda_max')
plt.plot(V_range, sv2_1, color='#FF0000', linestyle='--', marker='+', label='lane2,Lambda_max')
plt.plot(V_range, sv1_075, color='blueviolet', linestyle=':', marker='d', label='lane1,0.75*Lambda_max')
plt.plot(V_range, sv2_075, color='gold', linestyle=':', marker='p', label='lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Average velocity of two lanes')
plt.ylabel('The intra-platoon spacing/m')
plt.show()
# --------------------------------------------------------------------------------------------------
# Compute Sp
sp1_1, sp2_1 = [], []
sp1_075, sp2_075 = [], []
# The Sp of lambda = max
for i in range(len_V):
	v_1 = V_range[i] + v_date
	v_2 = V_range[i] - v_date
	lanmuda1 = Lambdas_1[i][0]
	lanmuda2 = Lambdas_1[i][1]
	sp1 = vn.compute_sp(s0,v0,v_1,Th,lanmuda1,mv,l)
	sp2 = vn.compute_sp(s0,v0,v_2,Th,lanmuda2,mv,l)
	sp1_1.append(sp1)
	sp2_1.append(sp2)
# The Sp of lambda = 0.75*max
for i in range(len_V):
	v_1 = V_range[i] + v_date
	v_2 = V_range[i] - v_date
	lanmuda1 = Lambdas_075[i][0]
	lanmuda2 = Lambdas_075[i][1]
	sp1 = vn.compute_sp(s0,v0,v_1,Th,lanmuda1,mv,l)
	sp2 = vn.compute_sp(s0,v0,v_2,Th,lanmuda2,mv,l)
	sp1_075.append(sp1)
	sp2_075.append(sp2)

# The inter-platoon spacing
plt.plot(V_range, sp1_1, color='#1f77b4', linestyle='--', marker='*', label='lane1,Lambda_max')
plt.plot(V_range, sp2_1, color='#FF0000', linestyle='--', marker='+', label='lane2,Lambda_max')
plt.plot(V_range, sp1_075, color='blueviolet', linestyle=':', marker='d', label='lane1,0.75*Lambda_max')
plt.plot(V_range, sp2_075, color='gold', linestyle=':', marker='p', label='lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Average velocity of two lanes')
plt.ylabel('The inter-platoon spacing/m')
plt.show()
# --------------------------------------------------------------------------------------------------
# The contention window 0.005
plt.plot(V_range, W0[:,0], color='#1f77b4', linestyle='--', marker='*', label='lane1,Lambda_max')
plt.plot(V_range, W0[:,1], color='#FF0000', linestyle='--', marker='+', label='lane2,Lambda_max')
plt.plot(V_range, W0[:,0], color='blueviolet', linestyle=':', marker='d', label='lane1,0.75*Lambda_max')
plt.plot(V_range, W0[:,1], color='gold', linestyle=':', marker='p', label='lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Average velocity of two lanes')
plt.ylabel('Optimal minimum contention window')
plt.show()
# --------------------------------------------------------------------------------------------------
# Kindex - 0.005
path_Kindex_1 = 'img_txt/data/standard_Kindex_1.txt'
path_Kindex_075 = 'img_txt/data/standard_Kindex_075.txt'
k_Kindex = 2
standard_Kindex_1 = (loadDatadet(path_Kindex_1,k_Kindex)).reshape(-1,2)
standard_Kindex_075 = (loadDatadet(path_Kindex_075,k_Kindex)).reshape(-1,2)

Kindex1 = []
for i in range(len_V):
	V = np.array([V_range[i]+v_date,V_range[i]-v_date])
	K_temp = compute_Kindex(V,W0[i])
	Kindex1.append(K_temp)
Kindex1 = np.array(Kindex1)

print(Kindex1)

# plt.plot(V_range, Kindex_avg, color='aqua', linestyle='-', label='Avg_Kindex')
plt.plot(V_range, Kindex1[:,0], color='#1f77b4', linestyle='--', marker='*', label='Optimal,lane1,Lambda_max')
plt.plot(V_range, Kindex1[:,1], color='#FF0000', linestyle='--', marker='+', label='Optimal,lane2,Lambda_max')

plt.plot(V_range, standard_Kindex_1[:,0], color='orange', linestyle='--', marker='o', label='Standard,lane1,Lambda_max')
plt.plot(V_range, standard_Kindex_1[:,1], color='hotpink', linestyle='--', marker='s', label='Standard,lane2,Lambda_max')

plt.plot(V_range, Kindex1[:,0], color='blueviolet', linestyle=':', marker='d', label='Optimal,lane1,0.75*Lambda_max')
plt.plot(V_range, Kindex1[:,1], color='gold', linestyle=':', marker='p', label='Optimal,lane2,0.75*Lambda_max')

plt.plot(V_range, standard_Kindex_075[:,0], color='lightgreen', linestyle=':', marker='x', label='Standard,lane1,0.75*Lambda_max')
plt.plot(V_range, standard_Kindex_075[:,1], color='deepskyblue', linestyle=':', marker='h', label='Standard,lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Average velocity of two lanes')
plt.ylabel('Fairness index of each lane')
plt.show()
# --------------------------------------------------------------------------------------------------
# Network average age - 0.005
# 输入统计数据
velocity = ['22.5', '23.0', '23.5', '24.0', '24.5', '25.0', '25.5', '26.0']
barWidth = 0.25

path_o_age_1 = 'img_txt/data/optimal_AoI_1_0.005.txt'
path_s_age_1 = 'img_txt/data/standard_AoI_1.txt'
path_o_age_075 = 'img_txt/data/optimal_AoI_075_0.005.txt'
path_s_age_075 = 'img_txt/data/standard_AoI_075.txt'
k_AoI = 1
optimal_AoI_1 = (loadDatadet(path_o_age_1,k_AoI).reshape(1,-1)).tolist()[0]
standard_AoI_1 = (loadDatadet(path_s_age_1,k_AoI).reshape(1,-1)).tolist()[0]
optimal_AoI_075 = (loadDatadet(path_o_age_075,k_AoI).reshape(1,-1)).tolist()[0]
standard_AoI_075 = (loadDatadet(path_s_age_075,k_AoI).reshape(1,-1)).tolist()[0]

r1 = np.arange(len(standard_AoI_1))
r2 = [x + barWidth for x in r1]

# age compare under max lambda
plt.bar(r1,standard_AoI_1,color='b',width=barWidth,label='Standard contention window,Lambda_max',edgecolor='white')
plt.bar(r2,optimal_AoI_1,color='r',width=barWidth,label='Optimal contention window,Lambda_max',edgecolor='white')

# plt.grid(linestyle=':')
plt.legend()
plt.xticks([r + barWidth/2 for r in range(len(standard_AoI_1))], velocity)
plt.xlabel('Average velocity of two lanes(m/s)')
plt.ylabel('Network average age(ms)')
plt.show()

# age compare under 0.75*lambda
plt.bar(r1,standard_AoI_075,color='b',width=barWidth,label='Standard contention window,0.75*Lambda_max',edgecolor='white')
plt.bar(r2,optimal_AoI_075,color='r',width=barWidth,label='Optimal contention window,0.75*Lambda_max',edgecolor='white')

# plt.grid(linestyle=':')
plt.legend()
plt.xticks([r + barWidth/2 for r in range(len(standard_AoI_075))], velocity)
plt.xlabel('Average velocity of two lanes(m/s)')
plt.ylabel('Network average age(ms)')
plt.show()
# --------------------------------------------------------------------------------------------------
# The contention window 0.005
plt.plot(V_range, W0_[:,0], color='#1f77b4', linestyle='--', marker='*', label='lane1,Lambda_max')
plt.plot(V_range, W0_[:,1], color='#FF0000', linestyle='--', marker='+', label='lane2,Lambda_max')
plt.plot(V_range, W0_[:,0], color='blueviolet', linestyle=':', marker='d', label='lane1,0.75*Lambda_max')
plt.plot(V_range, W0_[:,1], color='gold', linestyle=':', marker='p', label='lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Average velocity of two lanes')
plt.ylabel('Minimum contention window')
plt.show()
# --------------------------------------------------------------------------------------------------
# # Kindex - 0.02
Kindex1 = []
for i in range(len_V):
	V = np.array([V_range[i]+v_date,V_range[i]-v_date])
	K_temp = compute_Kindex(V,W0_[i])
	Kindex1.append(K_temp)
Kindex1 = np.array(Kindex1)

plt.plot(V_range, Kindex1[:,0], color='#1f77b4', linestyle='--', marker='*', label='Optimal,lane1,Lambda_max')
plt.plot(V_range, Kindex1[:,1], color='#FF0000', linestyle='--', marker='+', label='Optimal,lane2,Lambda_max')

plt.plot(V_range, standard_Kindex_1[:,0], color='orange', linestyle='--', marker='o', label='Standard,lane1,Lambda_max')
plt.plot(V_range, standard_Kindex_1[:,1], color='hotpink', linestyle='--', marker='s', label='Standard,lane2,Lambda_max')

plt.plot(V_range, Kindex1[:,0], color='blueviolet', linestyle=':', marker='d', label='Optimal,lane1,0.75*Lambda_max')
plt.plot(V_range, Kindex1[:,1], color='gold', linestyle=':', marker='p', label='Optimal,lane2,0.75*Lambda_max')

plt.plot(V_range, standard_Kindex_075[:,0], color='lightgreen', linestyle=':', marker='x', label='Standard,lane1,0.75*Lambda_max')
plt.plot(V_range, standard_Kindex_075[:,1], color='deepskyblue', linestyle=':', marker='h', label='Standard,lane2,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Average velocity of two lanes')
plt.ylabel('Kindex')
plt.show()
# --------------------------------------------------------------------------------------------------
# # Network average age - 0.02
# # 输入统计数据
path_o_age_1 = 'img_txt/data/optimal_AoI_1_0.02.txt'
path_o_age_075 = 'img_txt/data/optimal_AoI_075_0.02.txt'

optimal_AoI_1 = (loadDatadet(path_o_age_1,k_AoI).reshape(1,-1)).tolist()[0]
optimal_AoI_075 = (loadDatadet(path_o_age_075,k_AoI).reshape(1,-1)).tolist()[0]

# age compare under max lambda
plt.bar(r1,standard_AoI_1,color='b',width=barWidth,label='Standard contention window,Lambda_max',edgecolor='white')
plt.bar(r2,optimal_AoI_1,color='r',width=barWidth,label='Optimal contention window,Lambda_max',edgecolor='white')

# plt.grid(linestyle=':')
plt.legend()
plt.xticks([r + barWidth/2 for r in range(len(standard_AoI_1))], velocity)
plt.xlabel('Average velocity of two lanes(m/s)')
plt.ylabel('Network average age(ms)')
plt.show()

# age compare under 0.75*lambda
plt.bar(r1,standard_AoI_075,color='b',width=barWidth,label='Standard contention window,0.75*Lambda_max',edgecolor='white')
plt.bar(r2,optimal_AoI_075,color='r',width=barWidth,label='Optimal contention window,0.75*Lambda_max',edgecolor='white')

# plt.grid(linestyle=':')
plt.legend()
plt.xticks([r + barWidth/2 for r in range(len(standard_AoI_075))], velocity)
plt.xlabel('Average velocity of two lanes(m/s)')
plt.ylabel('Network average age(ms)')
plt.show()
# --------------------------------------------------------------------------------------------------
# Different bound
path_db_age_23_1 = 'img_txt/data/db_age_23.0_1.txt'
path_db_age_24_1 = 'img_txt/data/db_age_24.0_1.txt'
path_db_age_25_1 = 'img_txt/data/db_age_25.0_1.txt'
path_db_age_26_1 = 'img_txt/data/db_age_26.0_1.txt'

path_db_age_23_075 = 'img_txt/data/db_age_23.0_075.txt'
path_db_age_24_075 = 'img_txt/data/db_age_24.0_075.txt'
path_db_age_25_075 = 'img_txt/data/db_age_25.0_075.txt'
path_db_age_26_075 = 'img_txt/data/db_age_26.0_075.txt'

db_age_23_1 = (loadDatadet(path_db_age_23_1,k_AoI).reshape(1,-1)).tolist()[0]
db_age_24_1 = (loadDatadet(path_db_age_24_1,k_AoI).reshape(1,-1)).tolist()[0]
db_age_25_1 = (loadDatadet(path_db_age_25_1,k_AoI).reshape(1,-1)).tolist()[0]
db_age_26_1 = (loadDatadet(path_db_age_26_1,k_AoI).reshape(1,-1)).tolist()[0]

db_age_23_075 = (loadDatadet(path_db_age_23_075,k_AoI).reshape(1,-1)).tolist()[0]
db_age_24_075 = (loadDatadet(path_db_age_24_075,k_AoI).reshape(1,-1)).tolist()[0]
db_age_25_075 = (loadDatadet(path_db_age_25_075,k_AoI).reshape(1,-1)).tolist()[0]
db_age_26_075 = (loadDatadet(path_db_age_26_075,k_AoI).reshape(1,-1)).tolist()[0]

x1 = np.array([0.0005,0.0007,0.0009,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.0175,0.025])
x2 = np.array([0.0005,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.0175,0.025])
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
x2_smooth = np.linspace(x2.min(), x2.max(), 300)

y1_smooth = make_interp_spline(x2, db_age_23_1)(x2_smooth)
y2_smooth = make_interp_spline(x2, db_age_24_1)(x2_smooth)
y3_smooth = make_interp_spline(x2, db_age_25_1)(x2_smooth)
y4_smooth = make_interp_spline(x2, db_age_26_1)(x2_smooth)

y11_smooth = make_interp_spline(x2, db_age_23_075)(x2_smooth)
y12_smooth = make_interp_spline(x2, db_age_24_075)(x2_smooth)
y13_smooth = make_interp_spline(x1, db_age_25_075)(x1_smooth)
y14_smooth = make_interp_spline(x1, db_age_26_075)(x1_smooth)

a1 = [9.255556861646985567e+01]*len(x2)
a2 = [8.354293702552332945e+01]*len(x2)
a3 = [7.451925505084950885e+01]*len(x2)
a4 = [6.547930522567402534e+01]*len(x2)

a5 = [7.451925505084950885e+01]*len(x2)
a6 = [7.451925505084950885e+01]*len(x2)
a7 = [6.547930522567402534e+01]*len(x2)
a8 = [5.641397199958362307e+01]*len(x2)

# plt.plot(x2_smooth,y1_smooth,color='red', linestyle='--', label='v=23.0,Lambda_max')
# plt.plot(x2_smooth,y2_smooth,color='hotpink', linestyle='--', label='v=24.0,Lambda_max')
# plt.plot(x2_smooth,y3_smooth,color='orange', linestyle='--', label='v=25.0,Lambda_max')
# plt.plot(x2_smooth,y4_smooth,color='goldenrod', linestyle='--', label='v=26.0,Lambda_max')

# plt.plot(x2,a1,color='darkred',linestyle='-',label='v=23.0,Standard,Lambda_max')
# plt.plot(x2,a2,color='magenta',linestyle='-',label='v=24.0,Standard,Lambda_max')
# plt.plot(x2,a3,color='darkorange',linestyle='-',label='v=25.0,Standard,Lambda_max')
# plt.plot(x2,a4,color='yellow',linestyle='-',label='v=26.0,Standard,Lambda_max')

# plt.plot(x2_smooth,y11_smooth,color='lime', linestyle=':', label='v=23.0,0.75*Lambda_max')
# plt.plot(x2_smooth,y12_smooth,color='deepskyblue', linestyle=':', label='v=24.0,0.75*Lambda_max')
plt.plot(x1_smooth,y13_smooth,color='blue', linestyle=':', label='v=25.0,0.75*Lambda_max')
plt.plot(x1_smooth,y14_smooth,color='blueviolet', linestyle=':', label='v=26.0,0.75*Lambda_max')

# plt.plot(x2,a5,color='g',linestyle='-',label='v=23.0,Standard,0.75*Lambda_max')
# plt.plot(x2,a6,color='dodgerblue',linestyle='-',label='v=24.0,Standard,0.75*Lambda_max')
plt.plot(x2,a7,color='midnightblue',linestyle='-',label='v=25.0,Standard,0.75*Lambda_max')
plt.plot(x2,a8,color='purple',linestyle='-',label='v=26.0,Standard,0.75*Lambda_max')

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Kindex error')
plt.ylabel('Network average age(ms)')
plt.show()
# --------------------------------------------------------------------------------------------------