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


def find(event_kinds,dataset_name):
    #train the hawkes parameters for the dataset "dataset name" several times and save the execution with minimum negative log likelihood.
    M = 2

    plot_and_compare = False

    # Open and prepare database
    data = dataset(dataset_name=dataset_name, event_kinds=event_kinds)
    sessions = data.prepare_multivariate(combined=True, add_start_end_sess=1)[dataset_name]
    event_kinds = "_".join(str(x) for x in event_kinds)
    filename = 'Files/multi_combined_results_' + dataset_name + '_' + event_kinds + '.csv'

    for i in range(20):
        params = ism.train_hawkes_parameters(sessions)

        parameters = params
        print " UPDATED!"

        # update parameters file
        file = open(filename, "at")
        if any(params):
            s = ",".join(str(param) for param in parameters)
            file.write(s+"\n")
        file.close()
        print filename, "has been saved"

    file = open(filename, "rt")
    params=[]
    for line in file:
        params1=line.split(",")
        params.append(params1)
    params=np.asarray(params).astype(float)
    params=params[params[:,-1]==min(params[:,-1])][0]
    print params
    filename = 'Files/in_sess_multi_combined_' + dataset_name + '_' + event_kinds + '.csv'
    file = open(filename, "wt")
    s = ",".join(str(param) for param in params)
    file.write(s)
    file.close()
    print filename, "has been saved"

def plot():
    event_kinds = ['edits', 'comments']
    event_kinds = "_".join(str(x) for x in event_kinds)
    dataset_name = "handson3"
    filename = 'Files/multi_combined_results_' + dataset_name + '_' + event_kinds + '.csv'

    results = np.loadtxt(filename)
    gamma=(results[:,0:4] / results[:,4:8]*(1.-np.exp(-results[:,4:8]*50.)))
    results=np.append(results,gamma,axis=1)
    print results.shape
    file=open("statanal.csv","wt")
    for i in range(0,15):
        print i
        print "mean", np.mean(results[:,i])
        print "median", np.median(results[:,i])
        print "std", np.std(results[:,i]),"\n"
        file.write(str(i)+","+str(np.mean(results[:,i]).round(3))+","+str(np.median(results[:,i]).round(3))+","+str(np.std(results[:,i]).round(3))+"\n")
        #plt.scatter([1]*len(results[:,i]),results[:,i])
        #plt.show()
    file.close()


find(event_kinds = ['views', 'edits'], dataset_name = "handson3")



dsts=["handson3","demo","handson2"]
eks=[['views', 'comments'],['edits', 'comments'],['views', 'tools'],['edits','tools'], ["comments","tools"],['views', 'edits']]

def find_pairs(ek):
    for ds in dsts:
        find(event_kinds=ek,dataset_name=ds)

jobs=[]
for ek in eks:
    p=mtp.Process(target=find_pairs, args=(ek))
    jobs.append(p)
    p.start()