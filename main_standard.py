import numpy as np
import time
from avg_bo_tr_time import *
from Ki_compute import *


N = 2 # lane number
l = 5 # vehicle length
s0 = 2 # minimum intra-platoon spacing
v0 = 30 # v_max
Th = 1.6 # time headwayk
mv = 2 # platoon average vehicle numbers
D = 200 # rsu range
v_date = 2 # lane speed difference
Ts = 8972*(1e-03) # average duration of successful transmission
SlotTime = 50*(1e-03) # Tslot
v_avg ,W0_avg = 25,128
W0_stand = 128

# lanmuda_weight = 1
lanmuda_weight = 0.75
# lanmuda_weight = 0.5
Kindex_avg = compute_Kindex(v_avg,W0_avg)
K_bound = 0.005
K_index,AoI = [],[]

for v in np.arange(22.5,26.5,0.5):
	print("----------v:" + str(v) + "----------")
	V = [v+2,v-2] # Vehicle speed in lane1 and lane2(lane1>lane2).
	lanmudas, ni = [], []

	for i in range(0,N): # Calculate the fleet arrival rate and the number of lane vehicles.
		lanmuda_max = vn.compute_lanmuda_max(s0,v0,V[i],Th,mv,l)
		lanmuda = lanmuda_weight*lanmuda_max
		lanmudas.append(lanmuda) # Calculate the maximum arrival rate of each lane.
		n_i = vn.compute_vehicle_numbers(s0,v0,V[i],Th,lanmudas[i],mv,l,D) # Calculate the number of vehicles in each lane at the corresponding speed.
		ni.append(n_i)

	print('车辆到达率:' + str(lanmudas))
	print('车辆数目:' + str(ni))

	V = np.array(V)
	K_index.append(compute_Kindex(V,W0_stand))
	AoI.append(compute_age([W0_stand,W0_stand],SlotTime,N,ni,Ts))

K_index = np.array(K_index)
AoI = np.array(AoI)

np.savetxt("./img_txt/data/standard_Kindex.txt",K_index)
np.savetxt("./img_txt/data/standard_AoI.txt",AoI)

print(K_index,Kindex_avg)
print(AoI)


