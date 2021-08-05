import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


archive_fitness = np.loadtxt("img_txt/lambda_1_0.005/v_24.5_mopso/v_24.5_test4/filter_fitness_24.5_.txt")
fitness_ = np.loadtxt("img_txt/lambda_1_0.005/v_24.5_mopso/v_24.5_test4/pareto_fitness_24.5_.txt")

def show_final(fitness_,archive_fitness):
    fig = plt.figure('最终迭代总图')
    ax4 = fig.add_subplot(projection='3d')
    # X, Y, Z = np.meshgrid(fitness_[:,0],fitness_[:,1],fitness_[:,2])
    ax4.set_xlabel('Kindex1 difference')#,color='g'
    ax4.set_ylabel('Kindex2 difference')#,color='b'
    ax4.set_zlabel('Average age of network')#,color='r'
    surf = ax4.plot_trisurf(fitness_[:,0],fitness_[:,1],fitness_[:,2],cmap='coolwarm',linewidth=0,antialiased=True)
    ax4.scatter(archive_fitness[0],archive_fitness[1],archive_fitness[2],s=50, c='red', marker=".",alpha = 1.0)
    plt.show()

show_final(fitness_,archive_fitness)