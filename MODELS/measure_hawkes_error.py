import random
import numpy as np
import math
import datetime
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import time
from data import dataset
from mpl_toolkits.mplot3d import Axes3D
from plots import plot_timeline_with_intensity,plot_timelines_multivariate
import csv
import inside_session_multi as ism
import multiprocessing as mtp

def simulate_sessions(parameters,n_sessions,n_events_per_session):
    #simulate a session given a set of parameters
    sessions=[]
    for i in range(n_sessions):
        sess=ism.simulate_multi_hawkes(parameters[0], parameters[1], parameters[2], N_or_T_max=n_events_per_session)
        totalsess=np.append(sess[0],sess[1])
        sess.append([np.min(totalsess), np.max(totalsess)])
        sessions.append(sess)
    return sessions



def find_error(filename="errors.csv",plot=False):
    #for different number of training sessions, train the parameters several times and return the results.
    alpha = np.asarray([[0.3, 0.8], [0.2, 0.1]])
    beta = np.asarray([[1., 1.], [1., 1.]])
    mu = np.asarray([0.1, 0.01])

    parameters=[alpha,beta,mu]
    original_parameters_flat=np.append(np.append(alpha, beta), mu).flatten()
    original_parameters_flat=np.append(original_parameters_flat, original_parameters_flat[0:4]/original_parameters_flat[4:8]*(1.-np.exp(-original_parameters_flat[4:8]*50.)))
    original_parameters_flat=np.append(original_parameters_flat, np.zeros(1))

    errors_list=[[],np.empty([0,15])]
    nlist= [5,10]#,20,50,100,200,500,1000,2000]#range(5,29,5)+range(30,90,10)+range(90,200,30)+range(200,500,50)
    for n_sessions in nlist:
        print "ANEM PER LA ", n_sessions
        errors=np.empty([0,15])
        loglikes=[]

        for i in range(50):
            sessions=simulate_sessions(parameters=parameters,n_sessions=n_sessions, n_events_per_session=10)

            try:
                result=ism.train_hawkes_parameters(sessions)
                loglikes.append(result[-1])
                trained_parameters_flat = np.append(result[:-1], result[0:4] / result[4:8]*(1.-np.exp(-result[4:8]*50.)))
                trained_parameters_flat = np.append(trained_parameters_flat, np.asarray([result[-1]]))

                error=np.abs(original_parameters_flat-trained_parameters_flat)
                errors=np.vstack((errors,error))
            except: pass

        print "loglikes: ",loglikes
        min_loglike=min(loglikes)

        mean=errors[loglikes.index(min_loglike)]

        errors_list[0].append(n_sessions)
        errors_list[1]=np.vstack((errors_list[1],mean))
        print mean[-6:]

    np.savetxt(filename,errors_list[1])
    print errors_list
    if plot:
        for i in range(14):
            plt.plot(errors_list[0], errors_list[1][:,i])
            plt.show()

def charge_file():
    errors=np.loadtxt('errors.csv')
    for i in range(15):
        plt.plot([5,10,20,50,100,200,500,1000,2000], errors[:,i])
        plt.show()


jobs=[]
for i in range(1):
    p=mtp.Process(target=find_error, args=("error"+str(i)+".csv",False))
    jobs.append(p)
    p.start()
#charge_file()