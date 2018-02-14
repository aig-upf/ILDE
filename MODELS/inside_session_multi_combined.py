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

if __name__ == "__main__":

    M = 2
    event_kinds = ['views', 'edits']
    dataset_name = "handson3"
    plot_and_compare=False

    #Open and prepare database
    data = dataset(dataset_name = dataset_name, event_kinds=event_kinds)
    sessions = data.prepare_multivariate(combined=True, add_start_end_sess=1)[dataset_name]
    event_kinds="_".join(str(x) for x in event_kinds)

    #read the already calculated parameters
    try:

        reader = csv.reader(open('Files/in_sess_multi_combined' + dataset_name + '_' +event_kinds+'.csv', 'r'))
        for row in reader:
            parameters= np.asarray(row[:]).astype(np.float32)
    except: parameters=[]

    params = ism.train_hawkes_parameters(sessions)

    if not any(params):
        print "The training have failed"
        exit()

    if len(parameters) >2:
        print "old: ", parameters[-1], " new:", params[-1]
        if params[-1] > parameters[-1]:
            print "Parameters not updated"
            exit()

    parameters = params
    print " UPDATED!"

    if plot_and_compare:
        alpha = np.reshape(params[0:M ** 2], [M, M])
        beta = np.reshape(params[M ** 2:2 * M ** 2], [M, M])
        mu = np.reshape(params[2 * M ** 2:2 * M ** 2 + M], [M])

        for i in range(3):
            x_reals = [random.choice(sessions) for i in range(10)]

            T = float("-inf")
            N = 0
            for x_real in x_reals:
                N_ = 0
                for sec in x_real:
                    N_ += len(sec)
                    try:
                        val = sec[-1] - sec[0] + 1
                    except:
                        val = float("-inf")
                    T = max(T, val)
                N = max(N, N_)
            print T
            x = ism.simulate_multi_hawkes(alpha, beta, mu, N)
            print len(x[0]), len(x[1])
            plot_timeline_with_intensity(x, alpha, beta, mu, T)
            plot_timelines_multivariate([x] + x_reals)


    #update parameters file
    filename='Files/in_sess_multi_combined_' + dataset_name + '_' + event_kinds + '.csv'
    file = open(filename, "wt")
    if any(params):
        s = ",".join(str(param) for param in parameters)
        file.write(s)
    file.close()
    print filename, "has been saved"
