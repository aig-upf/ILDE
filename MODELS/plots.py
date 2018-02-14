import os
import random
import numpy as np
import math
import datetime
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import time
from data import dataset
from mpl_toolkits.mplot3d import Axes3D
from inside_session import simulate_uni_hawkes
from scipy import stats

def simulate_multi_hawkes(alpha, beta, mu, N_or_T_max, reference_stop_event="none", maxN_total=1000, plot=0, time=0):
    #simulate a multivariate hawkes process
    M=len(mu)
    events=[[] for j in range(M)]
    N=np.zeros([M])
    s = 0

    def landa(m,t):
        l=mu[m]
        for n in range(M):
            for t_ in events[n]:
                l+=alpha[m,n]*np.exp(-beta[m,n]*(t-t_))
        return l

    def I(k, t):
        I = 0
        for m in range(k+1):
            I+=landa(m,t)
        return I

    def attribution(s, I_,D):
        for m in range(M):
            if D <= I(m, s) / I_: return m

    end=False
    while not end:
        I_max=I(M-1,s)

        U=random.random()
        w= -np.log(U)/I_max
        s+=w
        D=random.random()

        if D<= I(M-1,s) / I_max:
            #here we choose the criteria to stop simulating
            if time: N_or_T_ref=s
            elif reference_stop_event=="none": N_or_T_ref=np.sum(N)
            else: N_or_T_ref= N[reference_stop_event]

            if N_or_T_ref<N_or_T_max and np.sum(N)<maxN_total:
                k=attribution(s,I_max,D)
                N[k]+=1
                events[k].append(s)
            else: end=True

    for i in range(len(events)):
        events[i]=np.asarray(events[i])
    return events

def plot_timeline_with_intensity(xs, alpha,beta, mu, T):
    #plot the timeline of a multivariate hawkes process and the intensity of the process

    M=len(mu)
    def landa(m, t):
        l = mu[m]
        for n in range(M):
            for t_ in xs[n]:
                if t_<t: l += alpha[m, n] * np.exp(-beta[m, n] * (t - t_))
                else: break
        return l    #plt.figure(figsize=(5, 1))

    plt.xlabel("Time (minutes)")
    plt.ylabel("Intensity")
    c=["r","b"]
    labels=["edits", "views"]
    for i in range(M):
        plt.plot(xs[i], [0.]*len(xs[i]), c=c[i], marker='.', ls="", label= labels[i])
    plt.legend(loc='upper right', shadow=True)
    plt.axis((0,T,-0.1,2))
    linespace=np.linspace(0,T,1000)

    land_values=[[] for i in range(M)]

    for m in range(M):
        for t in linespace:
            land_values[m].append(landa(m,t))

        plt.plot(linespace, land_values[m],c=c[m])

    plt.show()

def f(x):
    return x.total_seconds()
f = np.vectorize(f)

def plot_timelines(xs):
    #plot the timeline of a time series

    #plt.figure(figsize=(5, 1))
    plt.xlabel("Time (minutes)")
    #plt.ylabel("User id")
    xs_=[]
    if True:
        for x in xs:
            if len(x)>10:
                xs_.append(x)
    j=0
    n_xs=len(xs_)
    axes = plt.gca()
    axes.set_ylim([-0.5, n_xs - 0.5])
    plt.yticks([],[])
    for x in xs_:
        n = len(x)
        yusr = [j] * n
        x = np.asarray(x)
        x = x - x[0]
        if j == 0:
            c = "b"
        else:
            c = "r"
        plt.plot(x, yusr, c=c, marker='|', ls="")
        j += 1

    plt.show()

def plot_timelines_multivariate(xs):
    #plot the timeline of a multivariate time series
    M=len(xs[0])
    #plt.figure(figsize=(5, 1))
    plt.xlabel("Time (minutes)")
    #plt.ylabel("User id")
    xs_=[]
    if True:
        for x in xs:
            if len(x[0])>0:
                xs_.append(x)
    j=0
    n_xs=len(xs_)
    axes = plt.gca()
    axes.set_ylim([-0.5, n_xs - 0.5])
    plt.yticks([],[])
    lens=[]
    for x in xs_:
        c = ["r", "b"]
        prelen=[]
        x0=float("inf")
        for sec in x:
            try:
                val = sec[0]
            except:
                val = float("inf")
            x0 = min(x0, val)

        for i in range(M):
            n = len(x[i])
            prelen.append(n)
            if n==0: continue
            x[i] = np.asarray(x[i])
            x[i] = x[i] - x0
            plt.plot(x[i], [j] * n, c=c[i], marker='.', ls="")
        lens.append(prelen)
        j += 1
    print "lens:", lens

    plt.show()

def cdf(USERS, IDlist="none", plotAll=0, just_number_events=0):
    #function that computs de cumulative distrivution function of the dataset USERS
    #USERS is a list of two dicts, one for simulated data and the other for real data
    if IDlist == "none": IDlist = USERS[0].keys()
    dts = np.empty([0])
    prehists=[np.empty([0]),np.empty([0])]

    for id in IDlist:
        i=0
        for users in USERS:
            user = users[id]
            user_ = np.asarray(user)
            user_ant_ = np.roll(user_, 1)

            dt = user_ - user_ant_
            dt = dt[1:]

            if plotAll:
                hist=[array for array in np.histogram(dt,bins=50,normed=False)]
                hist[0]=np.insert(hist[0], 0, 0)
                plt.plot(hist[1],hist[0])

            prehists[i]=np.append(prehists[i],dt)
            i=1
        if plotAll:plt.show()

    if just_number_events: return len(prehists[0]), len(prehists[1])

    kolmogorov=0
    if kolmogorov:
        prehists[0]=np.sort(prehists[0])
        prehists[1]=np.sort(prehists[1])
        print "kolmogorov: ",stats.ks_2samp(prehists[0],prehists[1])

    c=["r","b"]
    label=["simulated","real"]
    for cum in range(2):
        plt.figure(1)
        for logscale in range(2):
            if logscale: plt.subplot(211)
            else: plt.subplot(212)
            for i in range(2):
                hist=prehists[i]
                hist = [array for array in np.histogram(hist, bins=np.logspace(-1, 5, 100))]
                ###poso aixo enlloc del que hi ha a sota
                if i==0: hist[0] = np.insert(hist[0], 0, 0)
                else: hist[0] = np.insert(hist[0], 0, 0)
                ###
                #hist[0] = np.insert(hist[0], 0, 0)*(4-3*i)
                ###
                hist[0] = hist[0]/np.sum(hist[0]).astype(np.float32)

                #hist[0]=np.cumsum(hist[0])
                if cum:
                    hist[0]=np.cumsum(hist[0][::-1])[::-1]

                plt.plot(hist[1], hist[0],c=c[i],label=label[i])
                plt.legend()
            plt.grid()
            if logscale: plt.yscale('log', nonposy='clip')
            plt.gca().set_xscale("log", base=2)
            plt.xlabel("Time (minutes)")
            plt.ylabel("Fraction of events")
        plt.show()
    return len(prehists[0]), len(prehists[1])


def plot_1v1(v1, v2):
    plt.scatter(v1, v2)
    plt.show()

def mean_std(data):
    print len(data)
    max_alphabeta=10
    max=10
    #data = data[np.logical_and(np.logical_and(data[:, 0]/data[:,4] < max_alphabeta, data[:, 1]/data[:,5] < max_alphabeta),np.logical_and(data[:, 2]/data[:,6] < max_alphabeta, data[:, 3]/data[:,7] < max_alphabeta))]
    #data = data[ np.logical_and(data[:, 0] / data[:, 4] < max_alphabeta, data[:, 3] / data[:, 7] < max_alphabeta)]
    data = data[np.logical_and(np.logical_and(data[:, 0]< max, data[:, 1]< max),np.logical_and(data[:, 2]< max, data[:, 3]< max))]
    data = data[np.logical_and(np.logical_and(data[:, 4] < max, data[:, 6] < max), np.logical_and(data[:, 5] < max, data[:, 7] < max))]
    mean=np.mean(data,0)
    median=np.median(data,0)
    std=np.std(data,0)
    alphabeta=data[:,0:4]/data[:,4:8]
    alphabetaacc=data[:,0:4]/data[:,4:8]*(1-np.exp(-data[:,4:8]*10))
    meanab=np.mean(alphabeta,0)
    stdab=np.std(alphabeta,0)
    medianab=np.median(alphabeta,0)
    meanabacc = np.mean(alphabetaacc, 0)
    stdabacc = np.std(alphabetaacc, 0)

    print mean
    print std
    print median
    print "avg alpha/beta = ", meanab, "   std = ", stdab, "  median = ", medianab
    print "avg  = ", meanabacc, "   std = ", stdabacc
    exit()

    for i in range(4):
        plt.hist(alphabeta[:,i], bins=20)
        plt.show()

def plot_combined_with_intensity():
    #plot the timeline of a multivariate hawkes process and the intensity of the process

    import csv
    M=2
    dataset_name = "handson3"
    event_kinds = ['edits', 'views']
    reader = csv.reader(open(
        'Files/in_sess_multi_combined_' + dataset_name + "_" + "_".join(str(x) for x in event_kinds) + '.csv',
        'r'))
    in_sess = []
    for row in reader:
        parameters = row[:]
        alpha = np.reshape(parameters[0:M ** 2], [M, M]).astype(np.float32)
        beta = np.reshape(parameters[M ** 2:2 * M ** 2], [M, M]).astype(np.float32)
        mu = np.reshape(parameters[2 * M ** 2:2 * M ** 2 + M], [M]).astype(np.float32)
        in_sess = [alpha, beta, mu]
    T=30
    x = simulate_multi_hawkes(alpha, beta, mu, T, time=1)
    plot_timeline_with_intensity(x, alpha, beta, mu, T)

def statistical_analysis():
    names=['handson2',"handson3", 'demo']
    name2={"handson3":'MOOC2','handson2':"MOOC1", 'demo':"demo"}
    #MODES
    for name in names:
        data=dataset(dataset_name=name)
        events_combined={}
        for id in data.IDs:
            events_combined[id]=[]
            for ek in data.total_event_kinds:
                try:events_combined[id]+=data.data[ek][id]
                except: print id, " have no ",ek
            events_combined[id].sort()
        total_combined={"total":[]}
        for id in data.IDs:
            total_combined["total"]+=events_combined[id]

        h=[]
        b=[]
        h_,b_=data.modes(users=events_combined, return_dist=1)
        h.append(h_)
        b.append(b_)

        plt.plot(b_,h_,label=name2[name])
    plt.legend()
    plt.grid()
    plt.gca().set_xscale("log", base=2)
    plt.gca().set_yscale("log", base=2)
    plt.ylabel("Number of repetitions")
    plt.xlabel("Time (minutes)")
    plt.show()

    #DAY DIST
    for name in names:
        data = dataset(dataset_name=name)
        events_combined = {}
        for id in data.IDs:
            events_combined[id] = []
            for ek in data.total_event_kinds:
                try:
                    events_combined[id] += data.data[ek][id]
                except:
                    print id, " have no ", ek
            events_combined[id].sort()
        total_combined = {"total": []}
        for id in data.IDs:
            total_combined["total"] += events_combined[id]

        h, b = data.average_hour_distribution(users=total_combined,ids=["total"], return_dist=1)
        plt.plot(b, h, label=name2[name])
    plt.legend()
    plt.grid()
    plt.xlabel("Hour of the day")
    plt.ylabel("Activity")
    plt.xlim([0, 24])
    plt.show()


    #WEEK DIST
    #DAY DIST
    for name in names:
        data = dataset(dataset_name=name)
        events_combined = {}
        for id in data.IDs:
            events_combined[id] = []
            for ek in data.total_event_kinds:
                try:
                    events_combined[id] += data.data[ek][id]
                except:
                    print id, " have no ", ek
            events_combined[id].sort()
        total_combined = {"total": []}
        for id in data.IDs:
            total_combined["total"] += events_combined[id]

        h, b = data.day_distribution(users=total_combined,ids=["total"], return_dist=1)
        plt.plot(b, h, label=name2[name])
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(7) + 1.5, ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
    plt.xlabel("Day of the week")
    plt.ylabel("Activity")
    plt.show()


    data.average_hour_distribution(events_combined)
    data.day_distribution(events_combined)

def show_gammas():
    dsts = ["handson3", "demo", "handson2"]
    eks = [['views', 'comments'], ['edits', 'comments'], ['views', 'tools'], ['edits', 'tools'], ["comments", "tools"],
           ['views', 'edits']]
    names=[]
    results=np.empty([0,11])
    for ek in eks:
        for ds in dsts:
            try:
                event_kinds = "_".join(str(x) for x in ek)
                filename = 'Files/in_sess_multi_combined_' + ds + '_' + event_kinds + '.csv'
                file=open(filename,"rt")
                for line in file:
                    vals=np.asarray(line.split(",")).astype(float)
                    results=np.vstack((results,vals))
                names.append([event_kinds, ds])
                if ds=="handson3":print  vals[0:4] / vals[4:8] * (1. - np.exp(-vals[4:8] * 50.)),vals[8:10],event_kinds, ds
            except: pass
    names=np.asarray(names)
    gammas=results[:,0:4] / results[:,4:8] * (1. - np.exp(-results[:,4:8] * 50.))
    print results
    print names
    np.savetxt("results.csv",results)
    file=open("resultnames.csv", "wt")
    for n in names:
        file.write("_".join(x for x in n)+"\n")
    file.close()

if __name__ == "__main__":
    statistical_analysis()

    show_gammas()
    plot_combined_with_intensity()
