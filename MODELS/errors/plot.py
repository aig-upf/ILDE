import numpy as np
from matplotlib import pyplot as plt

alpha = np.asarray([[0.3, 1.2], [0.2, 0.1]])
beta = np.asarray([[0.8, 2.6], [0.4, 0.2]])
mu = np.asarray([0.2, 0.006])

parameters = [alpha, beta, mu]
original_parameters_flat = np.append(np.append(alpha, beta), mu).flatten()
original_parameters_flat = np.append(original_parameters_flat,original_parameters_flat[0:4] / original_parameters_flat[4:8] * (1. - np.exp(-original_parameters_flat[4:8] * 50.)))

params=[]

for ll in ["a","b"]:
    for i in range(5):
        for j in range(10):
            try:
                params_ = np.loadtxt(str(i)+ll+"trained_params"+str(j)+".csv")
                params.append(params_)
            except: print ll,i,j
params=np.asarray(params)
print np.shape(params)
print params[:,5,9]


titles=[r'$\alpha^{11}$',r'$\alpha^{12}$',  r'$\alpha^{21}$',r'$\alpha^{22}$', r'$\beta^{11}$',r'$\beta^{12}$',  r'$\beta^{21}$',r'$\beta^{22}$', r'$\mu^1$',r'$\mu^2$',r'$E(N^{11}_T)$',r'$E(N^{12}_T)$', r'$E(N^{21}_T)$',r'$E(N^{22}_T)$']
for p in range(8,15):
    mean=np.mean(params, 0)
    median=np.median(params,0)
    std=np.std(params, 0)
    plt.xscale('log', nonposy='clip')
    plt.axhline(y=original_parameters_flat[p],c="red")
    plt.xlim(1, 1000)
    plt.ylabel(titles[p],fontsize=35)
    plt.xlabel("Number of sessions",fontsize=25)
    plt.scatter([10,20,50,100,200,500], median[:,p],c=np.random.rand(3,1))
    plt.errorbar([10,20,50,100,200,500], median[:,p], std[:,p], fmt='none')
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
    #plt.gcf().set_size_inches(18.5, 10.5)

    plt.show()
