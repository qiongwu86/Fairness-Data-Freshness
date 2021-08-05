#encoding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import fitness_funs as fit

class Plot_pareto:
    def __init__(self,N,Nv,V,v,Tslot,Ts,Kindex_avg):
        self.v = v
        # 绘制测试函数的曲面,（x1，x2）表示两位度的输入,（y1，y2，y3）表示三位的适应值
        self.x1 = np.linspace(0,10,100)
        self.x2 = np.linspace(0,10,100)
        self.x1,self.x2 = np.meshgrid(self.x1,self.x2) # 根据横纵坐标生成网格点
        self.m,self.n = np.shape(self.x1)
        self.y1,self.y2,self.y3 = np.zeros((self.m,self.n)),np.zeros((self.m,self.n)),np.zeros((self.m,self.n))
        for i in range(self.m):
            for j in  range(self.n):
                [self.y1[i,j],self.y2[i,j],self.y3[i,j]] = fit.fitness_([self.x1[i,j],self.x2[i,j]],N,Nv,V,Tslot,Ts,Kindex_avg)
        if os.path.exists('./img_txt') == False:
            os.makedirs('./img_txt')
            print ('创建文件夹img_txt:保存粒子群每一次迭代的图片')

    def show(self,in_,fitness_,archive_in,archive_fitness,i):
        # 共4个子图，第1、2、3/子图绘制输入坐标与适应值关系，第4图展示pareto边界的形成过程
        fig = plt.figure('第'+str(i+1)+'次迭代子图',figsize = (17,5))
 
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_xlim(64,256)
        ax1.set_ylim(64,256)
        ax1.set_zlim(0,0.6)
        ax1.set_xlabel('lane1_W0')
        ax1.set_ylabel('lane2_W0')
        ax1.set_zlabel('Kindex1 difference')
        # ax1.plot_surface(self.x1,self.x2,self.y1,alpha = 0.6)
        ax1.scatter(in_[:,0],in_[:,1],fitness_[:,0],s=20, c='blue', marker=".")
        ax1.scatter(archive_in[:,0],archive_in[:,1],archive_fitness[:,0],s=50, c='red', marker=".")

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_xlim(64,256)
        ax2.set_ylim(64,256)
        ax2.set_zlim(0,0.6)
        ax2.set_xlabel('lane1_W0')
        ax2.set_ylabel('lane2_W0')
        ax2.set_zlabel('Kindex2 difference')
        # ax2.plot_surface(self.x1,self.x2,self.y2,alpha = 0.6)
        ax2.scatter(in_[:,0],in_[:,1],fitness_[:,1],s=20, c='blue', marker=".")
        ax2.scatter(archive_in[:,0],archive_in[:,1],archive_fitness[:,1],s=50, c='red', marker=".")

        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_xlim(64,256)
        ax3.set_ylim(64,256)
        ax3.set_zlim(40,150)
        ax3.set_xlabel('lane1_W0')
        ax3.set_ylabel('lane2_W0')
        ax3.set_zlabel('Average age of network')
        # ax3.plot_surface(self.x1,self.x2,self.y3,alpha = 0.6)
        ax3.scatter(in_[:,0],in_[:,1],fitness_[:,2],s=20, c='blue', marker=".")
        ax3.scatter(archive_in[:,0],archive_in[:,1],archive_fitness[:,2],s=50, c='red', marker=".")
        # plt.show()
        plt.savefig('./img_txt/v_' + str(self.v) + '_mopso/'+'sub'+str(i+1)+'.eps')
        print ('第'+str(i+1)+'次迭代的图片子图保存于 img_txt 文件夹')
        plt.close()

    # def show_final(self,in_,fitness_,archive_in,archive_fitness):
    #     fig = plt.figure('最终迭代总图')
    #     ax4 = fig.add_subplot(111, projection='3d')
    #     ax4.set_xlabel('Kindex1_difference')
    #     ax4.set_ylabel('Kindex2_difference')
    #     ax4.set_zlabel('Average_AoI')
    #     # ax4.plot_surface(self.y1,self.y2,self.y3,alpha = 0.6)
    #     ax4.scatter(fitness_[:,0],fitness_[:,1],fitness_[:,2],s=10, c='blue', marker=".")
    #     ax4.scatter(archive_fitness[0],archive_fitness[1],archive_fitness[2],s=30, c='red', marker=".",alpha = 1.0)
    #     plt.show()
    #     # plt.savefig('./img_txt/v_' + str(self.v) + '_mopso/'+'final.eps')
    #     print ('最终迭代的图片总图保存于 img_txt 文件夹')
    #     plt.close()

    def show_final(self,in_,fitness_,archive_in,archive_fitness):
        fig = plt.figure('最终迭代总图')
        ax4 = fig.add_subplot(projection='3d')
        # X, Y, Z = np.meshgrid(fitness_[:,0],fitness_[:,1],fitness_[:,2])
        ax4.set_xlabel('Kindex1 difference')#,color='g'
        ax4.set_ylabel('Kindex2 difference')#,color='b'
        ax4.set_zlabel('Average age of network')#,color='r'
        surf = ax4.plot_trisurf(fitness_[:,0],fitness_[:,1],fitness_[:,2],cmap='coolwarm',linewidth=0,antialiased=True)
        # ax4.plot_surface(X,Y,Z)
        ax4.scatter(archive_fitness[0],archive_fitness[1],archive_fitness[2],s=50, c='red', marker=".",alpha = 1.0)
        # fig.colorbar(surf, shrink=0.5, aspect=5) # 宽，高
        plt.show()
        # plt.savefig('./img_txt/v_' + str(self.v) + '_mopso/'+'final.eps')
        print ('最终迭代的图片总图保存于 img_txt 文件夹')
        plt.close()

    # def show_final(self,in_,fitness_,archive_in,archive_fitness):
    #     fig = plt.figure('最终迭代总图')
    #     ax4 = fig.add_subplot(111, projection='3d')
    #     X, Y = np.meshgrid(fitness_[:,0],fitness_[:,1])
    #     l1,l2 = X.shape
    #     Z = np.ones([l1,l2])*fitness_[:,2]
    #     ax4.set_xlabel('Kindex1_difference')
    #     ax4.set_ylabel('Kindex2_difference')
    #     ax4.set_zlabel('Average_AoI')
    #     # ax4.plot_surface(self.y1,self.y2,self.y3,alpha = 0.6)
    #     ax4.plot_surface(X,Y,Z)
    #     ax4.scatter(archive_fitness[0],archive_fitness[1],archive_fitness[2],s=30, c='red', marker=".",alpha = 1.0)
    #     plt.show()
    #     # plt.savefig('./img_txt/v_' + str(self.v) + '_mopso/'+'final.eps')
    #     print ('最终迭代的图片总图保存于 img_txt 文件夹')
    #     plt.close()
