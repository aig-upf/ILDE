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

dataset_name='handson3'
event_kinds=['views','edits','comments','tools','creations'] #AQUI HAN DESTAR TOTS ELS EVENTS DISPONIBLES
data=dataset(dataset_name=dataset_name, event_kinds=event_kinds)
plot=0
plot_total=0

user_timelines={}
for id in data.IDs:
    #CREEM EL USER TIMELINE, ON NO DISTINGIM ENTRE TIPUS D'EVENT
    user_timeline = []
    # per cada tipus devent, enganxem tots els events daquell tipus a user_timeline
    for kind in data.total_event_kinds:
        try:
            user_timeline += [timeline for timeline in data.data[kind][id]]
        except: pass
    user_timelines[id]=user_timeline

hour_d = data.average_hour_distribution(user_timelines, plot=False)  # daily average of different users
day_d = data.day_distribution(user_timelines, plot=False)  # weekly average of different users

distributions={}


prep=data.prepare_multivariate()

#distribution of session length for each kind of event
sess_len_dist={}
for ek in event_kinds:
    sess_len_dist[ek]={}
    for id in data.IDs:
        sess_len_dist[ek][id]=[]
#distribution of session duration for each user
sess_time_duration_dist={}
for id in data.IDs:
    sess_time_duration_dist[id] = []

N_weeks=data.timedifference.days/7. #number of weeks


#SESS_LEN_DISTRIBURTION
sess_time_duration_dist_combined=[]
for id in data.IDs:
    for sess in prep[id]:
        total_session=[]
        for kind_num in range(len(event_kinds)):
            total_session+=sess[kind_num]
            sess_len_dist[event_kinds[kind_num]][id].append(len(sess[kind_num]))
        total_session.sort()
        T=total_session[-1]-total_session[0] #session duration
        sess_time_duration_dist[id].append(T)
        sess_time_duration_dist_combined.append(T)

#save sess lens
for kind_event in event_kinds:
    sess_len_dist_combined = []
    file = open("Files/sess_len_dist_" + dataset_name + "_" + kind_event+".csv", "wt")
    for id in data.IDs:
        string = id
        for val in sess_len_dist[kind_event][id]:
            string += "," + str(val)
            sess_len_dist_combined.append(val)
        file.write(string + "\n")
    file.close()
    print "sess_len_dist have been saved"

    #SAVE THE COMBINED DIST
    print np.mean(np.asarray(sess_len_dist_combined))
    print np.median(np.asarray(sess_len_dist_combined))
    print np.std(np.asarray(sess_len_dist_combined))


    file = open("Files/sess_len_dist_combined_"+ dataset_name + "_" + kind_event + ".csv", "wt")
    s = ",".join(str(x) for x in sess_len_dist_combined)
    file.write(s + "\n")
    file.close()
    print "sess_len_dist_combined have been saved"

#save sess durations
file = open("Files/sess_duration_dist_" + dataset_name + ".csv", "wt")
for id in data.IDs:
    string = id
    for val in sess_time_duration_dist[id]:
        string += "," + str(val)
    file.write(string + "\n")
file.close()
print "sess_duration_dist have been saved"

file = open("Files/sess_duration_dist_combined_" + dataset_name + ".csv", "wt")
s = ",".join(str(x) for x in sess_time_duration_dist_combined)
file.write(s + "\n")
file.close()
print "sess_duration_dist_combined have been saved"



sess_per_week_acc=[]
distributions_acc=np.zeros(7*24)

#SESS_DISTRIBUTION
for id in data.IDs:
    #Calculem el promig de sessions per setmana
    sess_per_week=len(prep[id])/N_weeks #number of sessions per week
    sess_per_week_acc.append(sess_per_week)

    #CREEM LA DISTRIBUCIO DE SESSIONS HORARIA AL LLARG DE LA SETMANA
    distribution = np.empty([0])
    for d in range(7):
        h = hour_d[id] * day_d[id][d]
        distribution = np.append(distribution, h)
    distribution /= np.sum(distribution)
    distribution*=sess_per_week
    distributions[id] = distribution

    distributions_acc+=np.asarray(distribution) #Distribucio de sessions promig

    if plot:
        #plt.hist(N_sess,bins=max(N_sess)-1)
        #plt.show()
        plt.plot(range(24 * 7), distribution,marker=".")
        print "sessions per week:", sess_per_week
        plt.ylabel("Probability of having one session")
        plt.xticks(np.arange(7) * 24, ["                    Monday", "                    Tuesday", "                    Wednesday", "                    Thursday", "                    Friday", "                    Saturday", "                    Sunday"])
        plt.grid()
        plt.show()


distributions_acc = distributions_acc/np.sum(distributions_acc)*np.mean(np.asarray(sess_per_week_acc))

file=open("Files/sess_dist_"+ dataset_name +".csv","wt")
for id in distributions.keys():
    string=id
    for val in distributions[id]:
        string+= ","+str(val)
    file.write(string+"\n")
file.close()
print "sess_dist have been saved"

file=open("Files/sess_dist_combined_"+ dataset_name +".csv","wt")
s = ",".join(str(x) for x in distributions_acc)
file.write(s+"\n")
file.close()
print "sess_dist_combined have been saved"


if plot_total:
    plt.hist(sess_per_week_acc,bins=20)
    plt.show()
    plt.hist(sess_time_duration_dist_combined,bins=300)
    plt.show()
