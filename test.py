import numpy as np
import vehicle_numbers as vn
from avg_bo_tr_time import *
from Ki_compute import *
from sympy import symbols, solve


N = 2 # lane number
l = 5 # vehicle length
s0 = 2 # minimum intra-platoon spacing
v0 = 30 # v_max
Th = 1.6 # time headwayk
mv = 2 # platoon average vehicle numbers
D = 200 # rsu range
v_date = 2 # lane speed difference
Ts = 8972*(1e-03) # average duration of successful transmission
SlotTime = 13*(1e-03) # Tslot

lanmuda_weight = 1
lanmuda_weight = 0.8
lanmuda_weight = 0.61
p_upperbound = 0.05
W0_upperbound = 1024
Kindex_avg = 0.12403100775193798
K_bound = 0.01

W_min = 64
W_max = 256

for v in np.arange(22.5,26,0.5):
# v = 22.5
	print("----------v:" + str(v) + "----------")
	V = [v+2,v-2] # Vehicle speed in lane1 and lane2(lane1>lane2).
	lanmudas, ni = [], []

	for i in range(0,N): # Calculate the fleet arrival rate and the number of lane vehicles.
		lanmuda_max = vn.compute_lanmuda_max(s0,v0,V[i],Th,mv,l)
		lanmuda = lanmuda_weight*lanmuda_max
		lanmudas.append(lanmuda) # Calculate the maximum arrival rate of each lane.
		n_i = vn.compute_vehicle_numbers(s0,v0,V[i],Th,lanmudas[i],mv,l,D) # Calculate the number of vehicles in each lane at the corresponding speed.
		ni.append(n_i)

	# Lambdas.append(lanmudas)
	# Nis.append(ni)
	print('车辆到达率:' + str(lanmudas))
	print('车辆数目:' + str(ni))

	W0_new_range = []


	W1 = symbols('W1')
	W2 = symbols('W2')
	f1 = abs(Kindex_avg - (2*D)/(V[0]*W1+1))
	f2 = abs(Kindex_avg - (2*D)/(V[1]*W2+1))
	print(solve([f1 < K_bound, f2 < K_bound]))

	# message = [V,[64,64],lanmuda_weight]
	# tao,p = compute_p_tao(message)
	# print(tao,p)

	for i in range(W_min,W_max+1):

		for j in range(W_min,W_max+1):

			Kindex1 = compute_Kindex(V[0],i)
			Kindex2 = compute_Kindex(V[1],j)

			if (abs(Kindex1 - Kindex_avg) < K_bound) and (abs(Kindex2 - Kindex_avg) < K_bound):
				W0_new_range.append([i,j])

	# print('满足Kindex约束后范围:' + str(W0_new_range))

	Ages = []
	for W0s in W0_new_range:
		age = compute_age(W0s,SlotTime,N,ni,Ts)
		Ages.append(age)

	index_ = Ages.index(min(Ages))
	print('最优窗口：')
	print(W0_new_range[index_])
	print('最优指标：' + str(compute_Kindex(np.array(V),np.array(W0_new_range[index_]))))
	print('最优年龄：' + str(Ages[index_]))