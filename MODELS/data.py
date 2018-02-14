from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import os


#In this class is used to load the data and modify it to have the desierable characteristics
class dataset():
    def __init__(self,dataset_name='handson3',event_kinds=['views','edits']):

        users_interval=[-60] #the first more active users that we want to take into account
        self.dataset_name=dataset_name
        self.mindate,self.maxdate=self.dates_interval()

        self.timedifference= self.maxdate-self.mindate
        self.event_kinds=event_kinds
        self.total_event_kinds=['views','edits']#all of the event kinds, the used and the unused


        def to_sec(x):
            return x.total_seconds()
        self.to_sec= np.vectorize(to_sec)

        self.data = {}
        self.data_combined = {}

        for kind in self.total_event_kinds:
            self.data[kind] = self.get_data("/"+kind+".csv")

        self.IDs = self.most_active_users(self.data['views'], False, users_interval=users_interval)


    def get_data(self, filename):
        # create a dictionary with the data of the users. The keys are the IDs.

        directori = os.path.dirname(os.path.realpath(__file__))
        directori = os.path.abspath(os.path.join(directori, os.pardir) + '/DATA/'+self.dataset_name)
        file = open(directori + filename, "rt")
        next(file)
        users={}
        activity = []
        acuser = []
        for line in file:
            w = line.split(";")
            user = w[0]
            acuser.append(int(user))
            action = datetime.strptime(w[3][:-1], '%Y-%m-%d %H:%M:%S')
            if action > self.maxdate or action < self.mindate: continue
            if not user in users.keys():
                users[user] = [action]
            else:
                users[user].append(action)
        file.close()

        return users


    def most_active_users(self, users, plot=False, users_interval=[60]):
        # return the IDs of the most active users
        us_ac = []
        for key in users.keys():
            us_ac.append([int(key), int(len(users[key]))])
        us_ac = np.asarray(us_ac)
        us_ac = us_ac[np.argsort(us_ac[:, 1])]

        if plot:
            plt.hist(us_ac[:, 1],bins=100,  cumulative=True)
            plt.xlabel("Number of events")
            plt.ylabel("Number of users")
            plt.grid(True)
            plt.show()
        if len(users_interval)==1: most_users=us_ac[users_interval[0]:, 0]
        else: most_users = us_ac[users_interval[0]:users_interval[1], 0]
        return map(str, most_users)

    def dates_interval(self):
        interval = {}
        interval['handson2'] =['20/05/2014 01:53:00','20/06/2014 00:00:00']
        interval['handson3'] =['1/11/2014 01:53:00','1/12/2014 00:00:00']
        interval['demo'] =['1/11/2012 01:53:00','1/12/2017 00:00:00']
        return datetime.strptime(interval[self.dataset_name][0], '%d/%m/%Y %H:%M:%S'),datetime.strptime(interval[self.dataset_name][1], '%d/%m/%Y %H:%M:%S')

    def prepare(self, ids="none"):
        users = {}
        if ids == "none": ids = self.IDs
        for k in ids:
            ntl = []  # newtimeline
            tl = self.to_sec(np.asarray(self.data['views'][k]) - self.data['views'][k][0]) / 60.  # timeline
            tl_ant = np.roll(tl, 1)
            dtl = tl - tl_ant
            dtl[0] = 1000  # datetime.timedelta(0,7000)
            max_dt = 80  # datetime.timedelta(0,6000)
            j = -1
            for i in range(len(tl)):
                if dtl[i] < max_dt:
                    ntl[j].append(tl[i])
                else:
                    ntl.append([tl[i]])
                    j += 1
            users[k] = ntl
        return users

    def combine(self,total=1):
        events_combined = {}
        for id in self.IDs:
            events_combined[id] = []
            for ek in self.total_event_kinds:
                try:
                    events_combined[id] += self.data[ek][id]
                except:
                    print id, " have no ", ek
            events_combined[id].sort()
        if not total: return events_combined

        total_combined = {"total": []}
        for id in self.IDs:
            total_combined["total"] += events_combined[id]
        return total_combined

    def prepare_multivariate(self, data="none", ids="none", combined=0, add_start_end_sess=0):
        #data es el diccionari de diccionaris
        users = {}
        if data == "none": data = self.data
        if ids == "none": ids = self.IDs

        #fem una llista amb tots els tipus devents on els dos primers son els dos que ens interessen i despres van els altres
        event_kinds=self.event_kinds[:]
        n_event_kinds=len(event_kinds)
        for e in self.total_event_kinds:
            if not e in event_kinds: event_kinds.append(e)

        for k in ids:
            type = 0
            user_type = []
            user_timeline = []
            #per cada tipus devent, enganxem tots els events daquell tipus a user_timeline i marquem el tipus d'event a user_type
            for kind in event_kinds:
                try:
                    user_type += [type for i in range(len(data[kind][k]))]
                    user_timeline += [timeline for timeline in data[kind][k]]
                except:
                    pass
                type += 1

            #ordenem el user_type a traves del user_timeline i despres ordenem el user timeline:
            # OBTENIM EL USER TIMELINE ORDENAT TEMPORALMENT I EL USER TYPE ENS INDICA QUIN EVENT TENIM EN CADA CAS
            user_type = [y for (x, y) in sorted(zip(user_timeline, user_type))]
            user_timeline.sort()

            ntl = []  # newtimeline
            tl = self.to_sec(np.asarray(user_timeline) - user_timeline[0]) / 60.  #fem que la timeline comenci a 0 i ho convertim a minuts
            tl_ant = np.roll(tl, 1)
            dtl = tl - tl_ant
            dtl[0] = 10000  # minucies del programa, es xq el primer event
            max_dt = 30  # temps maxim a partir del qual ja considerem que es una nova sessio
            sessio = -1

            #es va recorrent el timeline i quan el temps entre event i event es superior a max_dt, es comensa una nova sessio
            for i in range(len(tl)):
                if dtl[i] < max_dt:
                    if user_type[i]< n_event_kinds: ntl[sessio][user_type[i]].append(tl[i])
                    if add_start_end_sess: ntl[-1][-1][-1]=tl[i]
                else:
                    ntl.append([[] for k_ in range(n_event_kinds)])
                    if add_start_end_sess: ntl[-1].append([tl[i],tl[i]]) #afegim un altreapartat dins la sessio on hi posarem linici i el final de la sessio
                    if user_type[i] < n_event_kinds: ntl[-1][user_type[i]].append(tl[i])
                    sessio += 1

            users[k] = ntl

        if not combined:
            return users

        else:
            users_combined = {self.dataset_name: []}
            for id in users.keys():
                for session in users[id]:
                    users_combined[self.dataset_name].append(session)
            return users_combined

    def keys_to_timeseries(self, keylist, users="none"):
        if users=="none": users=self.data["views"]

        ts={}
        for k in keylist:
            ts[k]=users[k]
        return ts

    def dates_to_minutes(self, pre_users="none"):
        if pre_users=="none": pre_users=self.data["views"]
        users={}
        for id in self.IDs:
            try: users[id]=self.to_sec(np.asarray(pre_users[id]) - pre_users[id][0]) / 60.
            except: users[id]=[]
        return users

    def plot_activity_timeline(self, users_list="none", keys="none"):
        #users list es una llista de diccionaris de users, si nomes
        # posem un diccionari cal posar-lo entre brakets ex: [users]
        if users_list=="none": users_list=[self.data["views"]]
        if keys=="none": keys=self.IDs
        c=["red","green","purple","black","yellow"]
        i=0
        n = len(keys)
        #plt.title("Top activity")
        plt.xlabel("Time")
        plt.ylabel("User")
        noms=[]
        for k in range(n,0,-1):
            noms.append("user "+str(k))

        plt.yticks(range(n), noms)
        axes = plt.gca()
        axes.set_ylim([-0.5, n - 0.5])
        for users in users_list:
            usr = []
            yusr = []
            #j = 0
            for id in keys:
                try:
                    usr_ = users[id]
                    yusr_ = [keys.index(id)]*len(usr_)#[j] * len(usr_)

                    usr += usr_
                    yusr += yusr_
                    #j += 1
                except:continue
            plt.plot(usr, yusr, c=c[i], marker='|', ls="")
            i+=1

        plt.show()

    def average_hour_distribution(self, users, ids="none", plot=1, return_dist=0):
        users_hists={}
        if ids=="none": ids=self.IDs
        for id in ids:
            hours=[]
            i=-1
            hind=[]
            dia_=[0]*24
            dant=99
            for date in users[str(id)]:
                h=date.hour
                d=date.day
                hours.append(h)
                if d==dant: hind[i][h]+=1
                else:
                    dant=d
                    dia=dia_[:]
                    hind.append(dia)
                    i+=1
                    hind[i][h] += 1
            hind=np.asarray(hind)
            hind_mean=np.mean(hind,0)
            var=(hind-hind_mean)**2.
            var=(np.mean(var,0))**0.5
            norm=np.sum(hind)/np.shape(hind)[0]
            hist, bins=np.histogram(hours,bins=range(25))
            hist=np.asarray(hist,dtype=float)
            cumhist=np.cumsum(hist)
            hist= hist/norm
            cumhist/=norm
            #plt.bar(np.arange(24),cumhist)
            #plt.colors()
            users_hists[id]=hist
            if plot:
                hist/=np.sum(hist)

                if return_dist:
                    # hist, bins = np.histogram(dts,bins=np.logspace(-5, 28, 29, base=1.5),weights=weights)
                    return hist, bins[1:]

                plt.xlabel("Hour of the day",fontsize=25)
                plt.ylabel("Activity",fontsize=25)
                plt.bar(np.arange(24),hist)
                plt.tick_params(axis='both', which='major', labelsize=20)

                mid = np.arange(24)+0.5
                #plt.errorbar(mid, hist, yerr=var, fmt='none',ecolor="r")
                plt.xlim([0,24])
                plt.show()
        return users_hists


    def day_distribution(self,users="none", ids="none",plot=1,return_dist=0):
        users_hists={}
        if users=="none": users=self.data["views"]
        if ids=="none": ids=self.IDs
        for id in ids:
            day=[]
            dia_=[0]*7
            dinw=[[0,0,0,0,0,0,0]]
            i=0
            we=False
            for date in users[str(id)]:
                d=date.weekday()+1
                day.append(d)
                if d==7:
                    we=True
                if d !=7 and we:
                    we=False
                    dia = dia_[:]
                    dinw.append(dia)
                    i += 1
                    dinw[i][d-1] += 1
                else:
                    dinw[i][d-1] += 1

            #dinw = np.asarray(dinw)
            #print dinw
            #dinw_mean = np.mean(dinw)
            #var = (dinw - dinw_mean) ** 2
            #var = (np.mean(var, 0))**0.5

            #plt.hist(day, bins=7, cumulative=-1)
            weights = np.ones_like(day) / float(len(day))
            if not return_dist: hist, bins, _ = plt.hist(day, bins=np.arange(0.5,8.5,1),weights=weights)
            if return_dist:
                hist, bins= np.histogram(day, bins=np.arange(0.5, 8.5, 1))
                hist=hist.astype(float)
                hist/=np.sum(hist).astype(float)
                print hist
                return hist, bins[1:]
            if plot:

                plt.xticks(np.arange(7)+1, ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
                plt.xlabel("Day of the week",fontsize=20)
                plt.ylabel("Activity",fontsize=20)
                plt.tick_params(axis='both', which='major', labelsize=20)

                #mid = 0.5 * (bins[1:] + bins[:-1])
                #plt.errorbar(mid, hist, yerr=var, fmt='none')
                #plt.grid()
                plt.show()
            users_hists[id]=hist
        plt.clf()
        return users_hists

    def modes(self, users="none",IDlist="none", return_dist=0):
        if users=="none": users=self.data["views"]
        if IDlist=="none": IDlist=self.IDs
        print IDlist
        dts=np.empty([0])
        plot_total=1
        for id in IDlist:
            usr=users[id]
            usr_=np.asarray(usr)
            usr_ant_=np.roll(usr_,1)

            dt=usr_-usr_ant_
            dt=dt[1:]

            dt= self.to_sec(dt)
            dt=dt/60.
            #plt.hist(dt,bins=np.logspace(0, 5, 100),cumulative=-1)
            style=0
            if style==1:
                plt.hist(dt,bins=np.logspace(-1, 28, 29,base=1.5))
                plt.yscale('log', nonposy='clip')
                plt.gca().set_xscale("log",base=2)
                plt.xlabel("dt (minutes)")
                plt.ylabel("Number of repetitions")
                plt.show()

            if style==2:
                plt.hist(dt, bins=np.logspace(-1, 5, 60, base=10))
                plt.yscale('log', nonposy='clip')
                plt.gca().set_xscale("log", base=10)
                plt.xlabel("dt (minutes)")
                plt.show()

            if style==3:
                plt.hist(dt, bins=50)#np.logspace(-1, 5, 60, base=10))
                plt.yscale('log', nonposy='clip')
                plt.gca().set_xscale("log", base=24)
                plt.xlabel("time (minutes)")
                plt.show()
            if plot_total: dts=np.concatenate([dts,dt])

        if plot_total:
            weights = np.ones_like(dts) / len(dts)
            if return_dist:
                #hist, bins = np.histogram(dts,bins=np.logspace(-5, 28, 29, base=1.5),weights=weights)
                hist, bins = np.histogram(dts, bins=np.logspace(-1, 4, 100), weights=weights)
                return hist, bins[1:]

            print dts
            dts=np.asarray(dts)
            plt.hist(dts, bins=np.logspace(-0.3, 28, 29, base=1.5),weights=weights)
            plt.ylabel("Number of repetitions")
            plt.yscale('log', nonposy='clip')
            plt.gca().set_xscale("log", base=2)
            plt.xlabel("Time (minutes)")
            plt.grid()
            plt.show()

            plt.hist(dts, bins=np.arange(0,20,0.05),normed=1)
            plt.ylabel("Number of repetitions")
            plt.xlabel("time (minutes)")
            plt.grid()

            plt.show()
            plt.hist(dts, bins=np.arange(100,6000,50),normed=1)
            plt.ylabel("Number of repetitions")
            plt.xlabel("Time (minutes)")
            plt.grid()

            plt.show()

            decimals=[]
            for i in dts[:3000]:
                decimals.append(i-int(i))
            plt.hist(decimals,bins=60)
            plt.show()


if __name__ == "__main__":
    a=dataset()
    #a.prepare_multivariate()
    #a.average_hour_distribution(a.data["views"])
    a.day_distribution()
    #a.modes(users=a.data["views"])
    #print a.data["creations"]["5482"]
    a.plot_activity_timeline(users_list=[a.data["views"],a.data["comments"],a.data["edits"]])
    exit()