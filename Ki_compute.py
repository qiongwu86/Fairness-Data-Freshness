import numpy as np
import math
import vehicle_numbers as vn


N = 2 # lane number
s0 = 2 # m
v0 = 30 # m/s
Th = 1.6 # s
mv = 2
l = 5 # m
D = 200 # m

lanmudas = []
Nv = []

def compute_p_tao(message):
	vs = message[0]
	W0is = message[1]
	lanmuda_weight = message[2]

	for i in range(0,N):
		lanmuda_max = vn.compute_lanmuda_max(s0,v0,vs[i],Th,mv,l) # 车队的最大到达率
		lanmuda = lanmuda_weight*lanmuda_max
		lanmudas.append(lanmuda)
		ni = vn.compute_vehicle_numbers(s0,v0,vs[i],Th,lanmudas[i],mv,l,D) # 车队的车辆数目
		Nv.append(ni)

	tao1 = 2 / (W0is[0] + 1)
	tao2 = 2 / (W0is[1] + 1)
	p1 = 1 - np.power((1 - tao1),Nv[0]-1) * np.power((1 - tao2),Nv[1])
	p2 = 1 - np.power((1 - tao1),Nv[0]) * np.power((1 - tao2),Nv[1]-1)
	
	return [tao1,tao2],[p1,p2]

def compute_W_lowerbound(p_upper,Nv):
	Nv = sum(Nv)
	W0_upper_bound = 2/(1-math.exp(math.log(1-p_upper)/(Nv-1)))-1

	return round(W0_upper_bound)

def compute_Kindex(V,W0):
	Kc = (V * (W0 + 1))/2
	return D/Kc

