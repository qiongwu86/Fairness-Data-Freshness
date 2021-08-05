import numpy as np
import random
import math

# 计算车队内的车间距
def compute_sv(s0,v0,vi,Th):# 最小车间距、最大速度、车辆速度、时间进展
	sv = (s0+vi*Th)/(np.power(1-(np.power((vi/v0),4)),1.0/2))
	return sv

# 计算lane的最大车队到达率
def compute_lanmuda_max(s0,v0,vi,Th,mv,l):# 最小车间距、最大速度、车辆速度、时间进展、完整车队的平均车辆数、车长
	sv = compute_sv(s0,v0,vi,Th)
	lanmuda_max = vi/(mv*(l+sv))
	return lanmuda_max

# 计算车队间距
def compute_sp(s0,v0,vi,Th,lanmuda,mv,l):# 最小车间距、最大速度、车辆速度、时间进展、lane的车队到达率、完整车队的平均车辆数、车长
	sv = compute_sv(s0,v0,vi,Th)
	sp = (vi/lanmuda) - ((mv-1)*(l+sv)+l)
	return sp

# 计算lane中完整的车队的数目
def compute_c_platoon_numbers(s0,v0,vi,Th,lanmuda,mv,l,D):# 最小车间距、最大速度、车辆速度、时间进展、车队到达率、完整车队的平均车辆数、车长、RSU范围
	sv = compute_sv(s0,v0,vi,Th)
	sp = compute_sp(s0,v0,vi,Th,lanmuda,mv,l)
	k_t = D/((mv-1)*(l+sv)+l+sp)
	k = math.floor(k_t)
	return k

# 计算lane中所有完整的车队的车数量
def compute_c_vehicle_numbers(s0,v0,vi,Th,lanmuda,mv,l,D):
	k = compute_c_platoon_numbers(s0,v0,vi,Th,lanmuda,mv,l,D)
	nc = k * mv
	return nc

# 计算lane中部分车队中的车数量
def compute_p_vehicle_numbers(s0,v0,vi,Th,lanmuda,mv,l,D):
	k = compute_c_platoon_numbers(s0,v0,vi,Th,lanmuda,mv,l,D)
	sv = compute_sv(s0,v0,vi,Th)
	sp = compute_sp(s0,v0,vi,Th,lanmuda,mv,l)
	np_t = (D-k*((mv-1)*(l+sv)+l+sp))/(l+sv)
	if np_t > mv:
		np = mv
	else:
		np = math.ceil(np_t)
	return np

# 计算lane中的总的车辆数目
def compute_vehicle_numbers(s0,v0,vi,Th,lanmuda,mv,l,D):
	nc = compute_c_vehicle_numbers(s0,v0,vi,Th,lanmuda,mv,l,D)
	np = compute_p_vehicle_numbers(s0,v0,vi,Th,lanmuda,mv,l,D)
	n = nc + np
	return n
