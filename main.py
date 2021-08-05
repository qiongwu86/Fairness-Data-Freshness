#encoding: utf-8
import numpy as np
import time
from Mopso import *
import vehicle_numbers as vn
from avg_bo_tr_time import *
from Ki_compute import *
from sympy import symbols, solve
start = time.time() # 开始计时
 
def main():
    w = 0.8 # 惯性因子
    c1 = 1 # 局部速度因子1
    c2 = 2 # 全局速度因子2
    particals = 100 # 粒子群的数量
    cycle_ = 100 # 迭代次数
    mesh_div = 10 # 网格等分数量
    thresh = 250 # 外部存档阀值
    min_ = np.array([64,64]) # 粒子坐标的最小值
    max_ = np.array([256,256]) # 粒子坐标的最大值
    mopso_ = Mopso(particals,w,c1,c2,max_,min_,thresh,N,ni,V,v,SlotTime,Ts,Kindex_avg,K_bound,mesh_div) # 粒子群实例化
    pareto_in,pareto_fitness,filter_in,filter_fitness = mopso_.done(cycle_) # 经过cycle_轮迭代后，pareto边界粒子
    np.savetxt("./img_txt/v_"+str(v)+"_mopso/pareto_in_" + str(v) + ".txt",pareto_in) # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/v_"+str(v)+"_mopso/pareto_fitness_" + str(v) + ".txt",pareto_fitness) #打印pareto边界粒子的适应值
    np.savetxt("./img_txt/v_"+str(v)+"_mopso/filter_in_" + str(v) + ".txt",filter_in) # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/v_"+str(v)+"_mopso/filter_fitness_" + str(v) + ".txt",filter_fitness) # 打印pareto边界粒子的适应值
    print("\n","pareto边界的坐标保存于:/img_txt/v_"+str(v)+"_mopso/pareto_in_" + str(v) + ".txt")
    print("pareto边界的适应值保存于:/img_txt/v_"+str(v)+"_mopso/pareto_fitness_" + str(v) + ".txt")
    print("\n迭代结束,over.")

N = 2 # lane number
l = 5 # vehicle length
s0 = 2 # minimum intra-platoon spacing
v0 = 30 # v_max
Th = 1.6 # time headway
mv = 2 # platoon average vehicle numbers
D = 200 # rsu range
v_date = 2 # lane speed difference
Ts = 8972*(1e-03) # average duration of successful transmission
SlotTime = 50*(1e-03) # Tslot under platoon arrival rate $\lambda_{max}$
v_avg ,W0_avg = 25,128

lanmuda_weight = 1
# lanmuda_weight = 0.75
# lanmuda_weight = 0.5
Kindex_avg = compute_Kindex(v_avg,W0_avg)
K_bound = 0.005

# for v in np.arange(22.5,26.5,0.5):
v = 24.5
print("----------v:" + str(v) + "----------")
V = [v+v_date,v-v_date] # Vehicle velocity in lane1 and lane2(lane1>lane2).
lanmudas, ni = [], []

for i in range(0,N): # Calculate the fleet arrival rate and the number of lane vehicles.
    lanmuda_max = vn.compute_lanmuda_max(s0,v0,V[i],Th,mv,l)
    lanmuda = lanmuda_weight*lanmuda_max
    lanmudas.append(lanmuda) # Calculate the maximum arrival rate of each lane.
    n_i = vn.compute_vehicle_numbers(s0,v0,V[i],Th,lanmudas[i],mv,l,D) # Calculate the number of vehicles in each lane at the corresponding speed.
    ni.append(n_i)

print('车辆到达率:' + str(lanmudas))
print('车辆数目:' + str(ni))

if __name__ == "__main__":
    main()

end = time.time()
print("循环时间：%2f秒"%(end-start))
