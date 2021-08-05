#encoding: utf-8
import numpy as np
from Ki_compute import *
from avg_bo_tr_time import *

#为了便于图示观察，试验测试函数为二维输入、二维输出
#适应值函数：实际使用时请根据具体应用背景自定义
def fitness_(in_,N,N_v,V,Tslot,Ts,Kindex_avg):
    fit_1 = abs(compute_Kindex(V[0],in_[0]) - Kindex_avg)
    fit_2 = abs(compute_Kindex(V[1],in_[1]) - Kindex_avg)
    fit_3 = compute_age(in_,Tslot,N,N_v,Ts)
    return [fit_1,fit_2,fit_3]
