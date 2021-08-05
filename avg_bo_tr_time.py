import numpy as np

# compute the average back-off time of stage i.
def compute_Tbi(Wi,Tslot):
	Tb = ((Wi-1)/2)*Tslot
	return Tb,1/Tb

# compute all lanes AoI.
def compute_age(W0s,Tslot,N,N_v,Ts):
	part1 = 0
	part2 = 0
	CR = 0
	Rs = []
	Hs = []
	for j in range(N):
		Tbi, Ri = compute_Tbi(W0s[j],Tslot)
		Hi = 1/Ts
		Rs.append(Ri)
		Hs.append(Hi)
		CR = CR + N_v[j] * (Rs[j]/Hs[j])
	CR = 1 + CR
	part1 = CR * (N_v[0]*(1/Rs[0]) + N_v[1]*(1/Rs[1]))

	fenzi = 0
	for i in range(N):
		fenzi += N_v[i] * (Rs[i]/np.power(Hs[i],2))
	part2 = (sum(N_v) * fenzi)/CR
	
	age = part1 + part2
	return age/sum(N_v)

age = compute_age([16,16],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([32,32],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([48,48],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([64,64],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([96,96],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([128,128],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([256,256],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([382,382],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([512,512],50*(1e-03),2,[1,1],8.972)
print(age)
age = compute_age([1024,1024],50*(1e-03),2,[1,1],8.972)
print(age)
