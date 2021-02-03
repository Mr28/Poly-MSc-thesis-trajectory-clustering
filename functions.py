# -*- coding: utf-8 -*-

import datetime, sqlite3, os, os.path, scipy, math, pickle, sys, random, shlex, subprocess, inspect
from termcolor import cprint
from scipy import spatial
import sqlite3 as sqlLib
# import psycopg2 as sqlLib
from pytz import timezone
import numpy as np
from numpy import savetxt, loadtxt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

# from numba import vectorize
from numba import guvectorize

import sklearn, sklearn.cluster, sklearn.mixture


def shCommand(text):
    cmd = shlex.split(text)
    subprocess.run(cmd)


shCommand("pip install dtw-python")
# from dtw import *
shCommand("pip install scikit-learn-extra")
import sklearn_extra
import sklearn_extra.cluster
shCommand("pip install traj-dist")
import traj_dist, traj_dist.distance
# import traj_dist.distance as tdist


def Time(text="time", prnt=True, color='green', on_color='on_grey'):
    tz = timezone('Canada/Eastern')
    dt = datetime.datetime.now(tz)
    if prnt:
        cprint(text+": "+str(dt), color=color, on_color=on_color)
    return dt


def LoadData(dataName="inD1"):
    shCommand("mkdir ./data")
    if dataName.startswith("inD"):
        shCommand("wget -O ./data/inD.zip https://www.dropbox.com/s/7vtgbafa2tj17kk/inD.zip?dl=0")
        shCommand("unzip -o -q ./data/inD.zip -d ./data")
    elif dataName.startswith("NGSIM"):
        shCommand("wget -O ./data/NGSIM.zip https://www.dropbox.com/s/cvg1jvstqu7apga/NGSIM_intersection.zip?dl=0")
        shCommand("unzip -o -q ./data/NGSIM.zip -d ./data")
    t = Time('data loaded and unzipped')


def RecordingList(directory, dataName="inD1"):
    if dataName.startswith("inD1"):
        files = (f for f in os.listdir(directory) if f.endswith('_tracks.csv') and f.startswith(('00', '01', '02', '03', '04', '05', '06')))
    elif dataName.startswith("inD2"):
        files = (f for f in os.listdir(directory) if f.endswith('_tracks.csv') and f.startswith(('07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17')))
    elif dataName.startswith("inD3"):
        files = (f for f in os.listdir(directory) if f.endswith('_tracks.csv') and f.startswith(('18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29')))
    elif dataName.startswith("inD4"):
        files = (f for f in os.listdir(directory) if f.endswith('_tracks.csv') and f.startswith(('30', '31', '31')))
    elif dataName.startswith("NGSIM"):
        files = (f for f in os.listdir(directory) if f.endswith('.txt'))
    return files


def SqlRead(files, dataName="inD1"):
    database = dataName + "_database.db"
    connection = sqlLib.connect(database)
    cursor = connection.cursor()
    cursor.execute("""DROP TABLE IF EXISTS TrackPoints""")
    t = Time('creating database')
    counter = 0
    if dataName.startswith("inD"):
        cursor.execute("""CREATE TABLE IF NOT EXISTS TrackPoints (recordingId INT, trackId INT, frame INT, trackLifetime INT,
                            xCenter REAL, yCenter REAL, heading REAL, width REAL, length REAL, xVelocity REAL, yVelocity REAL,
                            xAcceleration REAL, yAcceleration REAL, lonVelocity REAL, latVelocity REAL, lonAcceleration REAL,
                            latAcceleration)""")
    elif dataName.startswith("NGSIM"):
        cursor.execute("""CREATE TABLE IF NOT EXISTS Trackpoints (recordingId INT, trackId INT, frame INT, totalFrames INT,
                            globalTime REAL, xCenter REAl, yCenter REAL, globalX REAL, globalY REAL, vLength REAL, vWidth REAL,
                            vClass INT, vVel REAL, vAcc REAL, laneId SMALLINT, oZone INT, dZone INT, intId SMALLINT, sectionId SMALLINT,
                            direction SMALLINT, movement SMALLINT, preceding INT, following INT, spaceHeadway REAL, timeHeadway REAL
                            )""")
    for idx, f in enumerate(files):
        try:
            if dataName.startswith("inD"):
                data = pd.read_csv('./data/inD/data/'+f, header=0)
                cursor.executemany("""INSERT INTO TrackPoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                        )""", data.values[:])
            elif dataName.startswith("NGSIM"):
                data = pd.read_csv('./data/NGSIM/data/'+f, header=None, sep='\s+')
                sqlCode = """INSERT INTO TrackPoints VALUES ({}, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                )""".format(idx)
                cursor.executemany(sqlCode, data.values[:])
        except:
            counter += 1
            print("{}- '{}' didn't work".format(counter, f))
    t = Time('database created')
    connection.commit()
    connection.close()


#### limited to 1800 points in trajectory
def TrajList(intId=5, dataName="inD1", maxTrajLength=1800, nSample=None):
    database = dataName + "_database.db"
    connection = sqlLib.connect(database)
    cursor = connection.cursor()
    if dataName.startswith("inD"):
        cursor.execute("""SELECT recordingId, trackId FROM
                            (SELECT recordingId, trackId, count(0) count FROM TrackPoints GROUP BY recordingId, trackId)
                            WHERE count<{}""".format(maxTrajLength))
    elif dataName.startswith("NGSIM"):
        cursor.execute("""SELECT recordingId, trackId, intId FROM
                            (SELECT recordingId, trackId, intId, count(0) count FROM TrackPoints GROUP BY recordingId, trackId, intId)
                            WHERE count<{} AND intId={}""".format(maxTrajLength, intId))
    trajList = np.array(cursor.fetchall(), dtype=int)
    connection.close()
    if nSample!=None:
        trajList = trajList[:min(nSample, trajList.shape[0])]
    return trajList


def Traj(tupleId, dataName="inD1", intXMin=None, intXMax=None, intYMin=None, intYMax=None):
    database = dataName + "_database.db"
    connection = sqlLib.connect(database)
    cursor = connection.cursor()
    if dataName.startswith("inD"):
        cursor.execute("""SELECT xCenter, yCenter FROM TrackPoints WHERE recordingId={} AND trackId={} ORDER BY frame
                            """.format(tupleId[0], tupleId[1]))
    elif dataName.startswith("NGSIM"):
        if intXMin==None or intXMax==None or intYMin==None or intYMax==None:
            cursor.execute("""SELECT xCenter, yCenter FROM TrackPoints WHERE recordingId={} AND trackId={} AND intId={} ORDER BY frame
                                """.format(tupleId[0], tupleId[1], tupleId[2]))
        else:
            cursor.execute("""SELECT xCenter, yCenter FROM TrackPoints WHERE recordingId={} AND trackId={} AND xCenter>{} AND xCenter<{} AND yCenter>{} AND yCenter<{} ORDER BY frame
                                """.format(tupleId[0], tupleId[1], intXMin, intXMax, intYMin, intYMax))
    traj = np.array(cursor.fetchall())
    connection.close()
    return traj


def DatabaseSection(dataName="inD1", resetDatabase=False, pickleInDatabase=False, pickleInAllTrajectories=False, test=True, maxTrajLength=1800, nSample=None, plot=False):

    database = dataName + "_database.db"

    if resetDatabase:
        try:
            os.remove(database)
        except:
            pass

    if not os.path.isfile(database):
        if pickleInDatabase:
            if dataName.startswith("inD"):
                shCommand("wget -O ./inD_database.db https://www.dropbox.com/s/ri67c41wq8kdbrb/inD_database.db?dl=0")
            elif dataName.startswith("NGSIM"):
                shCommand("wget -o ./NGSIM_database.db https://www.dropbox.com/s/uf5gel3a54bo411/NGSIM_database.db?dl=0")

        else:
            # try:
            #     files = RecordingList("./data/inD/data/", "_tracks.csv")
            # except:
            LoadData(dataName)
            if dataName.startswith("inD"):
                files = RecordingList(directory="./data/inD/data/", dataName=dataName)
            elif dataName.startswith("NGSIM"):
                files = RecordingList(directory="./data/NGSIM/data/", dataName=dataName)
            SqlRead(files, dataName)

    intXMin, intXMax = None, None
    intYMin, intYMax = None, None
    if dataName.startswith("inD1"):
        intId = 0
    elif dataName.startswith("inD2"):
        intId = 0
    elif dataName.startswith("inD3"):
        intId = 0
    elif dataName.startswith("inD4"):
        intId = 0
    elif dataName == "NGSIM1":
        intId = 1
        intXMin, intXMax = -35, 35
        intYMin, intYMax = 120, 240
    elif dataName == "NGSIM2":
        intId = 2
    elif dataName == "NGSIM3":
        intId = 3
    elif dataName == "NGSIM4":
        intId = 4
    elif dataName == "NGSIM5":
        intId = 5
        intXMin, intXMax = -40, 40
        intYMin, intYMax = 1980, 2080
    else:
        cprint(text="""*\n*\n*\n Incorrect dataName input. It should be either: inD1, inD2, inD3, inD4, NGSIM1 or NGSIM5.\n*\n*\n*"""
            , color='red', on_color='on_grey')

    trajList = TrajList(intId=intId, dataName=dataName, maxTrajLength=maxTrajLength, nSample=nSample)

    if pickleInAllTrajectories:
        try:
            os.mkdir("./data")
        except:
            pass
        # !mkdir ./data
        if dataName.startswith("inD"):
            shCommand("wget -O ./data/inD/inDTrajectories.pickle https://www.dropbox.com/s/jwkijvw7qdxo7y7/inDTrajectories.pickle?dl=0")
            pickle_in = open("./data/inDTrajectories.pickle", "rb")
        # elif:

        trajectories = pickle.load(pickle_in)
        pickle_in.close()
        trajectories = allTrajectories[startTraj:startTraj+nTraj]
        # pickle_out = open("./data/inDTrajectories.pickle", "wb")
        # pickle.dump(allTrajectories, pickle_out)
        # pickle_out.close()
    else:
        t = Time(text='loading trajectories on RAM')
        trajectories = []
        # for i in range(len(trajList)):
        for i in range(len(trajList)):
            tr = Traj(trajList[i], dataName=dataName, intXMin=intXMin, intXMax=intXMax, intYMin=intYMin, intYMax=intYMax)
            if tr.shape[0]>0:
                trajectories.append(tr)
        t = Time(text='trajectories loaded')
        if dataName.startswith("inD"):
            pickle_out = open("./data/inDTrajectories.pickle", "wb")
        elif dataName.startswith("NGSIM"):
            pickle_out = open("./data/NGSIMTrajectories.pickle", "wb")
        pickle.dump(trajectories, pickle_out)
        pickle_out.close()

    nTraj = len(trajectories)
    if not test:
        try:
            os.mkdir("./data/"+dataName+"_output")
        except:
            pass
        savetxt('./data/'+dataName+"_output/"+dataName+"_trajList.CSV", trajList, delimiter=',')
    
    if plot:
        for tr in trajectories:
            # print(tr.shape)
            plt.plot(tr[:,0], tr[:,1], color='blue', linewidth=0.15)
        plt.show()

    return trajList, trajectories, nTraj


def Lcs(X, Y, M, L, epsilon=1.5): 
	m = X.shape[0] 
	n = Y.shape[0]
	# L = np.zeros((m+1, n+1))
	for i in range(1,m + 1): 
		for j in range(1,n + 1):
			if M[i-1,j-1]<epsilon:
				L[i,j] = L[i-1,j-1]+1
			else: 
				L[i,j] = max(L[i-1,j], L[i,j-1]) 
	return L[m,n]

def Dtw(X, Y, M, L):
    m = X.shape[0] 
    n = Y.shape[0]
    # L = np.full((m+1, n+1), np.inf)
    L[0,0] = 0
    for i in range(1,m + 1): 
        for j in range(1,n + 1):
            L[i,j] = M[i-1,j-1] + min(L[i-1,j-1], L[i-1,j], L[i,j-1])
    return L[m,n]

# @guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1Len,traj2Len), (traj1LenPlus1,traj2LenPlus1), () -> ()"
#     , nopython=True, target='cuda')
# def GuLcs(X, Y, M, L, epsilon, lcs): 
#     m = X.shape[0]
#     n = Y.shape[0]
#     for i in range(1,m + 1):
#         for j in range(1,n + 1):
#             if M[i-1,j-1]<epsilon[0]:
#                 L[i,j] = L[i-1,j-1]+1
#             else: 
#                 L[i,j] = max(L[i-1,j], L[i,j-1])
#     # return L[m,n]
#     lcs[0] = L[m,n]

@guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1LenPlus1,traj2LenPlus1), () -> ()"
    , nopython=True, target='parallel')
def GuLcs(X, Y, L, epsilon, lcs): 
    m = X.shape[0]
    n = Y.shape[0]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if (X[i-1,0]-Y[j-1,0])**2+(X[i-1,1]-Y[j-1,1])**2<epsilon[0]**2:
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1])
    lcs[0] = L[m][n]

# @guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1Len,traj2Len), (traj1LenPlus1,traj2LenPlus1) -> ()"
#     , nopython=True, target='cuda')
# def GuDtw(X, Y, M, L, dtw):
#     m = X.shape[0] 
#     n = Y.shape[0]
#     # L = np.full((m+1, n+1), np.inf)
#     L[0,0] = 0
#     for i in range(1,m + 1): 
#         for j in range(1,n + 1):
#             L[i,j] = M[i-1,j-1] + min(L[i-1,j-1], L[i-1,j], L[i,j-1])
#     # return L[m,n]
#     dtw[0] = L[m,n]

@guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1LenPlus1,traj2LenPlus1) -> ()"
    , nopython=True, target='parallel')
def GuDtw(X, Y, L, dtw):
    m = X.shape[0] 
    n = Y.shape[0]
    # L = np.full((m+1, n+1), np.inf)
    L[0,0] = 0
    for i in range(1, m+1): 
        for j in range(1, n+1):
            L[i,j] = ((X[i-1,0]-Y[j-1,0])**2+(X[i-1,1]-Y[j-1,1])**2)**0.5 + min(L[i-1,j-1], L[i-1,j], L[i,j-1])
    # return L[m,n]
    dtw[0] = L[m,n]

@guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1LenPlus1,traj2LenPlus1), () -> ()"
    , nopython=True, target='parallel')
def GuPf(X, Y, L, delta, pf):
    m = X.shape[0] 
    n = Y.shape[0]
    # L = np.full((m+1, n+1), np.inf)
    L[0,0] = 0
    pf[0] = 0
    # print('m,n', m, n)
    for i in range(0, m+0): 
        winFloor = int((1-delta[0])*i//1+1)
        winCeil = int((1+delta[0])*i//1+2)
        # print('i, f, c',i,  min(winFloor, n), min(winCeil,n))
        n = int(n)
        for j in range(min(winFloor, n), min(winCeil,n)):
            L[i,j] = ((X[i,0]-Y[j,0])**2+(X[i,1]-Y[j,1])**2)**0.5
        if min(winFloor, n)!=min(winCeil,n):
            pf[0] += min(L[i, min(winFloor, n):min(winCeil,n) ])
    # # return L[m,n]
    pf[0] = pf[0]/m


class DistFuncs():
    # def __init__(self, maxTrajLength = 1800):
    #     self.maxTrajLength = maxTrajLength
        # self.zeroMatrix = np.zeros((self.maxTrajLength+1, self.maxTrajLength+1))
        # self.infMatrix = np.full((self.maxTrajLength+1, self.maxTrajLength+1), np.inf)

    @guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1LenPlus1,traj2LenPlus1), () -> ()"
        , nopython=True, target='parallel')
    def GuLcss(X, Y, L, param, lcs):
        m = X.shape[0]
        n = Y.shape[0]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if (X[i-1,0]-Y[j-1,0])**2+(X[i-1,1]-Y[j-1,1])**2<param[0]**2:
                    L[i][j] = L[i-1][j-1]+1
                else: 
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        lcs[0] = 1-((L[m][n])/min(m,n))

    @guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1LenPlus1,traj2LenPlus1), () -> ()"
        , nopython=True, target='parallel')
    def GuDtw(X, Y, L, param, dtw):
        m = X.shape[0] 
        n = Y.shape[0]
        # L = np.full((m+1, n+1), np.inf)
        L[0,0] = 0
        for i in range(1, m+1): 
            for j in range(1, n+1):
                L[i,j] = ((X[i-1,0]-Y[j-1,0])**2+(X[i-1,1]-Y[j-1,1])**2)**0.5 + min(L[i-1,j-1], L[i-1,j], L[i,j-1])
        # return L[m,n]
        dtw[0] = L[m,n]#/min(m, n)

    @guvectorize(["f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:]"],"(traj1Len,dims), (traj2Len,dims), (traj1LenPlus1,traj2LenPlus1), () -> ()"
        , nopython=True, target='parallel')
    def GuPf(X, Y, L, param, pf):
        m = X.shape[0] 
        n = Y.shape[0]
        # print('m,n=',m,n)
        L[0,0] = 0
        pf[0] = 0
        for i in range(0, m+0):
            winFloor = int((1-0.5*param[0])*i//1)
            winCeil = int((1+0.5*param[0])*i//1+1)
            n = int(n)
            for j in range(min(max(0,winFloor), n), min(winCeil,n)):
                L[i,j] = ((X[i,0]-Y[j,0])**2+(X[i,1]-Y[j,1])**2)**0.5
            if min(winFloor, n)!=min(winCeil,n):
                pf[0] += min(L[i, min(winFloor, n):min(winCeil,n) ])
        pf[0] = pf[0]#/min(m, n)

    def Lcss(self, X, Y, L, param):
        return traj_dist.distance.c_e_lcss(X,Y, param)

    def Dtw(self, X, Y, L, param):
        return traj_dist.distance.c_e_dtw(X,Y)

    def Hausdorf(self, X, Y, L, param):
        return traj_dist.distance.c_e_hausdorff(X,Y)

    def Frechet(self, X, Y, L, param):
        return traj_dist.distance.c_frechet(X,Y)

    # def Sowd_grid(self, X, Y, L, param):
    #     return traj_dist.distance.c_sowd_grid(X,Y)

    # def Erp(self, X, Y, L, param):
    #     return traj_dist.distance.c_e_erp(X,Y, )

    def Edr(self, X, Y, L, param):
        return traj_dist.distance.c_e_edr(X,Y, param)

    def Sspd(self, X, Y, L, param):
        return traj_dist.distance.c_e_sspd(X,Y)


def DistMatricesSection(trajList, trajectories, nTraj=None, dataName="inD1",pickleInDistMatrix=False, test=True, maxTrajLength=1800,
                        similarityMeasure=[['GuLcss',[1, 2, 3, 5, 7, 10]], ['GuDtw',[-1]], ['GuPf',[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]],
                                           ['Lcss',[1, 2, 3, 5, 7, 10]], ['Dtw', [-1]], ['Hausdorf', [-1]], ['Edr', [1, 2, 3, 5, 7, 10]], ['Sspd', [-1]]
                                           , ['Frechet', [-1]]
                                           ],
                        lcssParamList=[1, 2, 3, 5, 7, 10], pfParamList=[0.1, 0.2, 0.3, 0.4]):

    try:
        os.mkdir("./data")
    except:
        pass
    # !mkdir ./data

    if nTraj==None:
        nTraj = len(trajectories)

    if pickleInDistMatrix:
        shCommand("wget -O ./data/distMatricesZip.zip")
        shCommand("rm -r -d ./data/distMatricesFolder")
        shCommand("unzip -o ./data/distMatricesZip.zip")

    else:
        shCommand("rm -d -r ./data/distMatricesFolder")
        shCommand("mkdir ./data/distMatricesFolder")
        distMatrices = []
        LZero = np.zeros((maxTrajLength+1,maxTrajLength+1)) ######################
        LInf = np.full((maxTrajLength+1,maxTrajLength+1), 1e6)
        distFuncs = DistFuncs()
        # testTraj0 = np.array([[0,0],[0,1]], dtype=float)
        # testTraj1 = np.array([[0.75,0],[0.5,1],[1,1.5],[0.75,2.5]], dtype=float)

        for (methodName, method) in inspect.getmembers(distFuncs):
            # try:
            #     method(testTraj0, testTraj1, L, 0.5)+0
            for [distName, paramValueList] in similarityMeasure:
                if methodName == distName:
                    if distName in ['Dtw', 'GuDtw', 'GuPf']:
                        LMatrix = LInf.copy()
                    else:
                        LMatrix = LZero.copy()
                    for paramValue in paramValueList:
                        distMatrix = np.zeros((nTraj,nTraj))
                        startTime = Time(prnt=False)
                        for i in range(nTraj):
                            for j in range(i+1):
                                tr1 = trajectories[i]
                                tr2 = trajectories[j]
                                distMatrix[i,j] = method(tr1, tr2, LMatrix, paramValue)
                                distMatrix[j,i] = distMatrix[i,j]
                                # if 'GuLcss' in distName:
                                #     print(method(tr1, tr2, LMatrix, paramValue), distName, tr1.shape, tr2.shape, paramValue, '\n', LMatrix[:3, :3])
                        endTime = Time(prnt=False)
                        print(f'{distName}, paramValue={paramValue}, runtime:{str(endTime-startTime)}')
                        # print(lcssMatrix)
                        ## pickle
                        # pickle_out = open("./data/gulcssMatrix_e{}.pickle".format(e), "wb")
                        # pickle.dump(lcssMatrix, pickle_out)
                        # pickle_out.close()
                        savetxt(f"./data/distMatricesFolder/{dataName}_{distName}Matrix_param{paramValue}.csv", distMatrix, delimiter=',')
                        distMatrices.append((distMatrix, f"{dataName}_{distName}Matrix_param{paramValue}"))
            # except:
            #     # cprint("failed to produce {:s}_{:s}Matrix_param{:02d}".format(dataName, distName, paramValue), color='red', on_color='on_grey')
            #     pass

    shCommand("rm ./data/distMatricesZip.zip")
    shCommand("zip -o -r ./data/distMatricesZip ./data/distMatricesFolder/")

    # files = (f for f in os.listdir("./data/distMatricesFolder/") if "Matrix" in f and '.csv' in f)
    # files = sorted(files)

    if not test:
        pickle_out = open('./data/'+dataName+"_output/"+dataName+'distMatrices.pickle', "wb")
        pickle.dump(distMatrices, pickle_out)
        pickle_out.close()

    return distMatrices


def Silhouette(model, distMatrix, trajIndices=None):#, permuting=True):
    if trajIndices==None:
        trajIndices = list(range(distMatrix.shape[0]))
    subDistMatrix = distMatrix[trajIndices][:,trajIndices]

    shufIndices = [i for i in range(len(trajIndices))]
    random.shuffle(shufIndices)
    shufSubDistMatrix = np.zeros_like(subDistMatrix)
    for i in range(len(shufIndices)):
        for j in range(len(shufIndices)):
            # shufSubDistMatrix[i,j] = subDistMatrix[shufIndices[i], shufIndices[j]]
            shufSubDistMatrix[shufIndices[i], shufIndices[j]] = subDistMatrix[i,j]

    model = model.fit(shufSubDistMatrix)
    # labels = model.labels_#[trajIndices]
    shufLabels = model.labels_#[trajIndices]

    labels = np.zeros_like(shufLabels)
    # print(shufIndices)
    # print(i, np.where(shufIndices==i))
    for i in range(len(shufIndices)):
        labels[np.where(np.array(shufIndices)==i)[0][0]] = shufLabels[i]
    
    clusters = list(set(labels))
    nClus = len(clusters)
    A = np.zeros(len(labels))
    B = np.full(len(labels), np.inf)
    S = np.zeros(len(labels))
    argmins = np.zeros(len(labels))
    # argmins = [0 for i in range(len(labels))]
    closestCluster = labels
    for i in range(len(labels)):
        similarTrajs = [l for l in range(len(labels)) if labels[l]==labels[i]]
        if len(similarTrajs)>1:
            # for k in similarTrajs:
            #     A[i] += distMatrix[i,k]
            A[i] = sum(subDistMatrix[i,similarTrajs])/(len(similarTrajs)-1)
            otherClusters = list(set(labels)-set([labels[i]]))
            b = np.inf
            for j in otherClusters:
                # b = 0
                dissimilarTrajsJ = [l for l in range(len(labels)) if labels[l]==j]
                # for k in dissimilarTrajsJ:
                #     b += distMatrix[i,k]
                b = np.mean(subDistMatrix[i,dissimilarTrajsJ])
                if b<B[i]:
                    B[i] = b
                    argmins[i] = j
                # B[i,j] = np.mean(distMatrix[i,dissimilarTrajsJ])
            # argmins[i] = np.argmin(B[i])
            S[i] = (B[i] - A[i]) / max(B[i], A[i])
            if S[i]<0:
                closestCluster[i] = argmins[i]

    return S, closestCluster, labels, subDistMatrix, shufSubDistMatrix


def Plot(model, distMatrix, trajIndices=None, S=np.array([]), closestCluster=np.array([]), title=None, plotTrajsTogether=False, plotTrajsSeperate=False, plotSilhouette=False, plotSilhouetteTogether=False, darkTheme=True):
    if darkTheme:
        tickColors = 'w'
    else:
        tickColors = 'black'

    if trajIndices==None:
        trajIndices = list(range(distMatrix.shape[0]))
    subDistMatrix = distMatrix[trajIndices][:,trajIndices]
    model = model.fit(subDistMatrix)
    labels = model.labels_
    if closestCluster==np.array([]):
        closestCluster = labels
    clusters = list(set(labels))
    minTrajX = min([min(tr[:, 0]) for tr in trajectories])
    minTrajY = min([min(tr[:, 1]) for tr in trajectories])
    maxTrajX = max([max(tr[:, 0]) for tr in trajectories])
    maxTrajY = max([max(tr[:, 1]) for tr in trajectories])

    try:
        nClus = model.n_clusters
    except:
        nClus = len(set(labels))
    cmap = list(colors.TABLEAU_COLORS)
    colormap = cmap
    repeat = nClus//len(cmap)
    for i in range(repeat):
        colormap = colormap + cmap

    if plotTrajsTogether:
        plt.figure(figsize=(16,12))
        for i, j in enumerate(trajIndices):
            tr = trajectories[j]
            plt.plot(tr[:,0], tr[:,1], c=colormap[labels[i]], linewidth=0.3)#, alpha=1)
            plt.scatter(tr[0,0], tr[0,1], c=100, s=2, marker='o')
        plt.xlim(minTrajX-20,maxTrajX+20)
        plt.ylim(minTrajY-20,maxTrajY+20)
        plt.tick_params(colors=tickColors)
        if title != None:
            plt.title(label=title, color=tickColors)
        plt.show()

    if plotTrajsSeperate:
        nRows = -(-nClus//4)
        plt.figure(figsize=(16,3*nRows), dpi=600)
        for i in range(nClus):
            fig = plt.subplot(nRows, 4, i+1)
            for j, k in enumerate(trajIndices):
                if labels[j] == clusters[i]:
                    tr = trajectories[k]
                    fig.plot(tr[:,0], tr[:,1], c=colormap[closestCluster[j]], linewidth=0.3)
                    fig.scatter(tr[0,0], tr[0,1], c=100, s=2, marker='o')
            fig.set_xlim(minTrajX-20,maxTrajX+20)
            fig.set_ylim(minTrajY-20,maxTrajY+20)
            fig.tick_params(colors=tickColors)
        if title != None:
            plt.suptitle(title, color=tickColors)
        plt.show()

    if plotSilhouette and S!=np.array([]):# and closestCluster!=np.array([]):
        cmap = list(colors.TABLEAU_COLORS)
        colormap = cmap
        repeat = nClus//len(cmap)
        for i in range(repeat):
            colormap = colormap + cmap
        nRows = -(-nClus//4)
        plt.figure(figsize=(16,3*nRows), dpi=600)
        for i in range(nClus):
            clusList = [j for j in range(len(labels)) if labels[j]==clusters[i]]
            sortedClusList = [clusList[i] for i in np.argsort(S[clusList])[::-1]]
            fig = plt.subplot(nRows, 4, i+1)
            # fig.bar(range(len(clusList)), S[clusList], color=colormap[i])
            fig.bar(range(len(sortedClusList)), S[sortedClusList], color=[colormap[closestCluster[j]] for j in sortedClusList])
            fig.set_ylim(-1,1)
            fig.tick_params(colors=tickColors)
        if title != None:
            plt.suptitle(title, color=tickColors)
        plt.show()

    if plotSilhouetteTogether:
        sortedS = [i for i in np.argsort(S)[::-1]]
        plt.bar(range(len(sortedS)), S[sortedS], color=[colormap[closestCluster[j]] for j in sortedS])
        plt.ylim(-1,1)
        plt.tick_params(colors=tickColors)
        if title != None:
            plt.title(label=title, color=tickColors)
        plt.show()


# def SilhouetteStats(model, distMatrix, S, closestCluster=np.array([])):
#     model = model.fit(distMatrix)
#     labels = model.labels_
#     if closestCluster==np.array([]):
#         closestCluster = labels
#     clusters = list(set(labels))
#     try:
#         nClus = model.n_clusters
#     except:
#         nClus = len(set(labels))

#     # sClusters = []
#     # for i in range(nClus):
#     #     clusList = [j for j in range(len(labels)) if labels[j]==clusters[i]]
#     #     sClusters.append(clusList)


def OdClustering(funcTrajectories, nTraj=None, modelList=None, nClusOriginSet=[4], nClusDestSet=[4], modelNames=['average Agglo-Hierarch'], nIter=1, visualize=False, shuffle=False, test=True, darkTheme=False):
    if shuffle or len(nClusOriginSet)>1 or len(nClusDestSet)>1:
        cprint('\n The internal set of trejecories and thus the output labels go out of sync with the input "trajectories" set if shuffle==True or len(nClusOriginSet)>1 or len(nClusDestSet)>1.\n', color='red', on_color='on_grey')

    if darkTheme:
        tickColors = 'white'
    else:
        tickColors = 'black'

    if nTraj==None:
        nTraj = len(funcTrajectories)
    
    startPoints = np.zeros((nTraj,2))
    endPoints = np.zeros((nTraj,2))
    for i in range(nTraj):
        tr = funcTrajectories[i]
        startPoints[i] = tr[0]
        endPoints[i] = tr[-1]

    startAvgDists = np.zeros((len(nClusOriginSet), nIter))
    endAvgDists = np.zeros((len(nClusDestSet), nIter))
    minTrajX = min([min(tr[:, 0]) for tr in funcTrajectories])
    minTrajY = min([min(tr[:, 1]) for tr in funcTrajectories])
    maxTrajX = max([max(tr[:, 0]) for tr in funcTrajectories])
    maxTrajY = max([max(tr[:, 1]) for tr in funcTrajectories])

    if modelList == None:
        modelList = [
                    (sklearn_extra.cluster.KMedoids(metric='euclidean'), 'KMedoids')
                    ,(sklearn.cluster.KMeans(precompute_distances='auto', n_jobs=-1), 'KMeans')
                    ,(sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='ward'), 'ward Agglo-Hierarch')
                    ,(sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='complete'), 'complete Agglo-Hierarch')
                    ,(sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='average'), 'average Agglo-Hierarch')
                    ,(sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='single'), 'single Agglo-Hierarch')
                    ,(sklearn.cluster.Birch(threshold=0.5, branching_factor=50), 'BIRCH')
                    ,(sklearn.mixture.GaussianMixture(covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=200, n_init=5, init_params='kmeans'), 'GMM')
                    ,(sklearn.cluster.SpectralClustering(affinity='rbf', n_jobs=-1), 'Spectral')
                    ,(sklearn.cluster.OPTICS(metric='minkowski', n_jobs=-1), 'OPTICS')
                    ,(sklearn.cluster.DBSCAN(metric='euclidean', n_jobs=-1), 'DBSCAN')
                    ]

    pickedModels = [(model, title) for (model, title) in modelList if title in modelNames]

    for model, title in pickedModels:
        for iter in range(nIter):
            shufIndices = [i for i in range(len(startPoints))]
            if shuffle or len(nClusOriginSet)>1 or len(nClusDestSet)>1:
                random.shuffle(shufIndices)
            shufStartPoints = np.zeros_like(startPoints)
            shufEndPoints = np.zeros_like(endPoints)
            for i in range(len(shufIndices)):
                shufStartPoints[shufIndices[i]] = startPoints[i]
                shufEndPoints[shufIndices[i]] = endPoints[i]

            shufStartPoints = startPoints.copy()
            shufEndPoints = endPoints.copy()
            if shuffle or len(nClusOriginSet)>1 or len(nClusDestSet)>1:
                random.shuffle(shufStartPoints)
                random.shuffle(shufEndPoints)

            # t = Time(text=title)
            for i in range(len(nClusOriginSet)):
                # t = Time("nClus={}".format(nClusOriginSet[i]))
                model1 = model
                try:
                    if title=="GMM":
                        model1.n_components = nClusOriginSet[i]
                    else:
                        model1.n_clusters = nClusOriginSet[i]
                except:
                    pass
                startModel = model1.fit(shufStartPoints)
                try:
                    startLabels = list(startModel.labels_)
                except:
                    startLabels = list(startModel.predict(shufStartPoints))
                startLabelList = list(set(startLabels))
                nClusStart = len(startLabelList)
                try:
                    startCenters = startModel.cluster_centers_
                except:
                    startCenters = np.zeros((nClusStart,2))
                    for k in range(nClusStart):
                        clusPoints = shufStartPoints[startLabels == startLabelList[k]]
                        startCenters[k] = np.average(clusPoints, axis=0)
                startDistSum = 0
                for k in range(nClusStart):
                    clusPoints = shufStartPoints[startLabels == startLabelList[k]]
                    for point in clusPoints:
                        startDistSum += ((point[0]-startCenters[k,0])**2 + (point[1]-startCenters[k,1])**2)**0.5
                startAvgDists[i, iter] = startDistSum/len(startLabels)
                # print(startAvgDists[i])
                if visualize:# and len(nClusOriginSet)>1 and len(nClusDestSet)>1:
                    try:
                        print("Calinski Harabasz: {}".format(sklearn.metrics.calinski_harabasz_score(shufStartPoints, startLabels)))
                    except:
                        pass
                    print("nClusStart={}".format(nClusStart))
                    cmap = list(colors.TABLEAU_COLORS)
                    colormap = cmap
                    repeat = nClusStart//len(cmap)
                    for k in range(repeat):
                        colormap = colormap + cmap
                    plt.figure(figsize=(8,6))
                    for k in range(nTraj):
                        plt.scatter(shufStartPoints[k,0], shufStartPoints[k,1], c=colormap[startLabels[k]])
                    plt.scatter(startCenters[:,0], startCenters[:,1], c='black')
                    plt.tick_params(colors=tickColors)
                
            # t = Time(text=title)
            for i in range(len(nClusDestSet)):
                # t = Time("nClus={}".format(nClusOriginSet[i]))
                model2 = model
                try:
                    if title=="GMM":
                        model2.n_components = nClusDestSet[i]
                    else:
                        model2.n_clusters = nClusDestSet[i]
                except:
                    pass
                endModel = model2.fit(shufEndPoints)
                try:
                    endLabels = list(endModel.labels_)
                except:
                    endLabels = list(endModel.predict(shufEndPoints))
                endLabelList = list(set(endLabels))
                nClusEnd = len(endLabelList)
                try:
                    endCenters = endModel.cluster_centers_
                except:
                    endCenters = np.zeros((nClusEnd,2))
                    for k in range(nClusEnd):
                        clusPoints = shufEndPoints[endLabels == endLabelList[k]]
                        endCenters[k] = np.average(clusPoints, axis= 0)
                endDistSum = 0
                for k in range(nClusEnd):
                    clusPoints = shufEndPoints[endLabels == endLabelList[k]]
                    for point in clusPoints:
                        endDistSum += ((point[0]-endCenters[k,0])**2 + (point[1]-endCenters[k,1])**2)**0.5
                endAvgDists[i, iter] = endDistSum/len(endLabels)
                # print(endAvgDists[i])
                if visualize:# and len(nClusOriginSet)>1 and len(nClusDestSet)>1:
                    try:
                        print("Calinski Harabasz: {}".format(sklearn.metrics.calinski_harabasz_score(shufEndPoints, endLabels)))
                    except:
                        pass
                    print("nClusEnd={}".format(nClusEnd))
                    cmap = list(colors.TABLEAU_COLORS)
                    colormap = cmap
                    repeat = nClusEnd//len(cmap)
                    for k in range(repeat):
                        colormap = colormap + cmap
                    plt.figure(figsize=(8,6))
                    for k in range(nTraj):
                        plt.scatter(shufEndPoints[k,0], shufEndPoints[k,1], c=colormap[endLabels[k]])
                    plt.scatter(endCenters[:,0], endCenters[:,1], c='black')
                    plt.tick_params(colors=tickColors)
                    plt.show()

    meanStartAvgDist = np.mean(startAvgDists, axis=-1)
    stdStartAvgDist = np.std(startAvgDists, axis=-1)#/(nIter**0.5)
    meanEndAvgDist = np.mean(endAvgDists, axis=-1)
    stdEndAvgDist = np.std(endAvgDists, axis=-1)#/(nIter**0.5)
    plotMax = max(np.max(meanStartAvgDist+stdStartAvgDist), np.max(meanEndAvgDist+stdEndAvgDist))

    if len(nClusOriginSet)>1 or len(nClusDestSet)>1:
        plt.figure(figsize=(16,6))
        fig = plt.subplot(1,2,1)
        fig.set_ylim(0,plotMax)
        fig.errorbar(x=nClusOriginSet, y=meanStartAvgDist, yerr=stdStartAvgDist, linestyle='None', fmt='-_', color='orange', ecolor='blue')
        fig.tick_params(colors=tickColors)
        fig.set_title("Origin clusters")
        fig = plt.subplot(1,2,2)
        fig.set_ylim(0,plotMax)
        fig.errorbar(x=nClusDestSet, y=np.mean(endAvgDists, axis=-1), yerr=np.std(endAvgDists, axis=-1)/(nIter**0.5), linestyle='None', fmt='-_', color='orange', ecolor='blue')
        fig.tick_params(colors=tickColors)
        fig.set_title("Destination clusters")
        plt.show()

    if not test:
        try:
            os.mkdir("./data/"+dataName+"_output")
        except:
            pass
        savetxt('./data/'+dataName+"_startLabels.CSV", startLabels, delimiter=',')
        savetxt('./data/'+dataName+"_endLabels.CSV", endLabels, delimiter=',')

    # return startLabels, endLabels, shufStartPoints, shufEndPoints
    unShufStartLabels = np.zeros_like(startLabels)
    unShufEndLabels = np.zeros_like(endLabels)
    for i in range(len(shufIndices)):
        unShufStartLabels[i] = startLabels[shufIndices[i]]
        unShufEndLabels[i] = endLabels[shufIndices[i]]

    return unShufStartLabels, unShufEndLabels, startPoints, endPoints, nClusStart, nClusEnd#, startAvgDists


def OdMajorClusters(trajectories, startLabels=None, endLabels=None, threshold=10, visualize=False, test=True, load=False):

    if load:
        try:
            startLabels = loadtxt('./data/'+dataName+"_startLabels.CSV", delimiter=',')
            endLabels = loadtxt('./data/'+dataName+"_endLabels.CSV", delimiter=',')
        except:
            raise Exception("No such file or directory: ./data/"+dataName+"_endLabels.CSV, "+"./data/"+dataName+"_endLabels.CSV")
    # else:
    #     startLabels, endLabels = startLabels, endLabels

    countOD = np.zeros((len(set(startLabels)), len(set(endLabels))))
    sampleTraj = np.zeros((len(set(startLabels)), len(set(endLabels))))

    startClusterIndices = []
    for i in list(set(startLabels)):
        startClusterIndices.append(list(np.where(startLabels==i)[0]))

    endClusterIndices = []
    for i in list(set(endLabels)):
        endClusterIndices.append(list(np.where(endLabels==i)[0]))

    odTrajLabels = np.full(len(startLabels), -1)
    refTrajIndices = []
    for i in range(countOD.shape[0]):
        for j in range(countOD.shape[1]):
            lst = list(set(startClusterIndices[i]) & set(endClusterIndices[j]))
            countOD[i,j] = len(lst)
            odTrajLabels[lst] = int(i * countOD.shape[1] + j)
            if countOD[i,j]>0:
                sampleTraj[i,j] = lst[0]
            if countOD[i,j]>threshold:
                refTrajIndices.extend(lst)
    refTrajIndices.sort()

    # refDistMatrix = np.zeros((len(refTrajIndices), len(refTrajIndices)))
    # for i in range(len(refTrajIndices)):
    #     refDistMatrix[i] = distMatrix[refTrajIndices[i], refTrajIndices]

    if visualize:
        ### major OD visulaization

        # print(countOD)
        threshold = 10
        plt.figure(figsize=(16,6))
        fig = plt.subplot(1,2,1)
        fig.set_title('examples of major ODs', color='w')
        for i in range(countOD.shape[0]):
            for j in range(countOD.shape[1]):
                if countOD[i,j] > threshold:
                    k = int(sampleTraj[i,j])
                    tr = trajectories[k]
                    fig.plot(tr[:,0], tr[:,1], label=len(tr))
                    fig.scatter(tr[0,0], tr[0,1], c=100, s=2, marker='o')
        fig.legend()
        fig = plt.subplot(1,2,2)
        fig.set_title('examples of minor ODs', color='w')
        for i in range(countOD.shape[0]):
            for j in range(countOD.shape[1]):
                if countOD[i,j] <= threshold:
                    k = int(sampleTraj[i,j])
                    tr = trajectories[k]
                    fig.plot(tr[:,0], tr[:,1])
                    fig.scatter(tr[0,0], tr[0,1], c=100, s=2, marker='o')
        plt.show()

    if not test:
        try:
            os.mkdir("./data/"+dataName+"_output")
        except:
            pass
        savetxt('./data/'+dataName+"_output/"+dataName+"_refTrajIndices.CSV", refTrajIndices, delimiter=',')
        savetxt('./data/'+dataName+"_output/"+dataName+"_odTrajLabels.CSV", odTrajLabels, delimiter=',')

    return refTrajIndices, odTrajLabels


class EvalFuncs():
    def AvgSManual(self, X, odLabels, trajLabels, S):
        return np.mean(S)
    def PosSRatioManual(self, X, odLabels, trajLabels, S):
        return len(np.where(S>0)[0])/len(S)
    def ARI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.adjusted_rand_score(odLabels, trajLabels)
    def MI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.mutual_info_score(odLabels, trajLabels)
    def Homogeneity(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.homogeneity_score(odLabels, trajLabels)
    def Completeness(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.completeness_score(odLabels, trajLabels)
    def V(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.v_measure_score(odLabels, trajLabels)
    def FMI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.fowlkes_mallows_score(odLabels, trajLabels)
    def S(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.silhouette_score(X, trajLabels)
    def CHI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.calinski_harabasz_score(X, trajLabels)
    def DBI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.davies_bouldin_score(X, trajLabels)
    


def Main(distMatrices, trajectories, odTrajLabels, refTrajIndices, nClusStart, nClusEnd, clusRange=list(range(2,15)), nIter=3, modelList=None, dataName="inD1", test=True, evalNameList=None, seed=0.860161037286291):

    t = Time('start')
    random.seed(seed)
    if modelList==None:
        modelList = [
            (sklearn_extra.cluster.KMedoids(metric='precomputed', init='k-medoids++'), 'KMedoids')
            ,(sklearn.cluster.KMeans(precompute_distances='auto'), 'KMeans')
            # (sklearn.cluster.AgglomerativeClustering(affinity='precomputed', linkage='ward'), 'ward Agglo-Hierarch')
            ,(sklearn.cluster.AgglomerativeClustering(affinity='precomputed', linkage='complete'), 'complete Agglo-Hierarch')
            ,(sklearn.cluster.AgglomerativeClustering(affinity='precomputed', linkage='average'), 'average Agglo-Hierarch')
            ,(sklearn.cluster.AgglomerativeClustering(affinity='precomputed', linkage='single'), 'single Agglo-Hierarch')
            # ,(sklearn.cluster.Birch(threshold=0.5, branching_factor=50), 'BIRCH')
            # (sklearn.mixture.GaussianMixture(covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=200, n_init=5, init_params='kmeans'), 'GMM')
            ,(sklearn.cluster.SpectralClustering(affinity='precomputed', n_jobs=-1), 'Spectral')
            ,(sklearn.cluster.OPTICS(metric='precomputed', n_jobs=-1), 'OPTICS')
            ,(sklearn.cluster.DBSCAN(metric='precomputed', n_jobs=-1), 'DBSCAN')
            ]

    cmap = list(colors.TABLEAU_COLORS)
    colormap = cmap
    repeat = len(modelList)//len(cmap)
    for k in range(repeat):
        colormap = colormap + cmap

    randArray = np.random.rand(nIter)      ###### distMatrices refTrajIndices, odTrajLabels
    tempDistMatrices = distMatrices
    if test:
        tempDistMatrices = [(distMatrix, distName) for (distMatrix, distName) in distMatrices if distName in [dataName+"_GuLcssMatrix_param7", dataName+"_GuDtwMatrix_param-1", dataName+"_GuPfMatrix_param0.2"]]

    initEvalMatrix = np.zeros((len(tempDistMatrices), len(modelList), len(clusRange), len(randArray)))
    # print(initEvalMatrix)

    evalFuncs = EvalFuncs()

    if evalNameList==None:
        evalNameList = [methodName for (methodName, method) in inspect.getmembers(evalFuncs, predicate=inspect.ismethod)]

    evalMeasures = []
    for evalName in evalNameList:
        matched=False
        for (methodName, method) in inspect.getmembers(evalFuncs, predicate=inspect.ismethod):
            if methodName == evalName:
                evalMeasures.append((initEvalMatrix.copy(), method, methodName))
                matched=True
                break
            # else:
            #     raise ValueError(f'{evalName} is not a valid name for an evaluation measure! It was skipped.')
                # cprint('{} is not a valid name for an evaluation measure! It was skipped.'.format(evalName), color='red', on_color='on_grey')
        if not matched:
            raise ValueError(f'{evalName} is not a valid name for an evaluation measure! It was skipped.')

    for idxMatrix, (distMatrix, distName) in enumerate(tempDistMatrices):
        t = Time(distName, color='yellow')

        for idxSeed, seed in enumerate(randArray):
            random.seed(a=seed)
            shufTrajectories = trajectories.copy()
            shufTrajIndices = list(range(distMatrix.shape[0]))
            # random.shuffle(shufTrajIndices)

            shufOdTrajLabels = np.zeros(odTrajLabels.shape)
            shufRefTrajIndices = [0 for _ in refTrajIndices]
            shufDistMatrix = np.zeros(distMatrix.shape)
            for i in range(len(shufTrajectories)):
                shufTrajectories[i] = trajectories[shufTrajIndices[i]]
                
            # startLabels, endLabels, startPoints, endPoints = OdClustering(modelList=None, nClusOriginSet=[11], nClusDestSet=[4], visualize=False, funcTrajectories=shufTrajectories)
            # refTrajIndices, odTrajLabels = OdMajorClusters(threshold=10)

            shufOdTrajLabels = odTrajLabels.copy()
            shufRefTrajIndices = refTrajIndices.copy()
            for i in range(len(shufTrajIndices)):
                shufOdTrajLabels[i] = odTrajLabels[shufTrajIndices[i]]
            for i in range(len(refTrajIndices)):
                shufRefTrajIndices[i] = np.where(shufTrajIndices==refTrajIndices[i])[0][0]
            for i in range(shufDistMatrix.shape[0]):
                for j in range(shufDistMatrix.shape[1]):
                    shufDistMatrix[i,j] = distMatrix[shufTrajIndices[i], shufTrajIndices[j]]

            for idxModel, (model, modelName) in enumerate(modelList):
                for idxClus, nClus in enumerate(clusRange):
                    # model1 = model.copy()
                    if modelName=='DBSCAN':
                        if 'dtw' in distName:
                            model.eps = 3
                        elif 'lcss' in distName:
                            model.eps = 0.2
                        elif 'pf' in distName:
                            model.eps = 3
                        else:
                            cprint('Epsilon value not specified yet for {} algorithm in the code. default value 0.5 is used.'.format(distName), color='red', on_color='on_yellow')
                    model.n_clusters = nClus
                    S, closestCluster, labels, subDistMatrix, shufSubDistMatrix = Silhouette(model=model, distMatrix=shufDistMatrix, trajIndices=shufRefTrajIndices)
                    trajLabels = labels
                    for idxEval, (_, evalFunc, evalName) in enumerate(evalMeasures):
                        try:
                            evalMeasures[idxEval][0][idxMatrix, idxModel, idxClus, idxSeed] = evalFunc(subDistMatrix, shufOdTrajLabels[shufRefTrajIndices], trajLabels, S)
                        except:
                            cprint(f'Evaluation metric {evalName} failed to work', color='red', on_color='on_grey')
                    # avgS[idxMatrix, idxModel, idxClus, idxSeed] = np.mean(S)
                    # posSIndices = np.where(S>0)[0]
                    # posSRatio[idxMatrix, idxModel, idxClus, idxSeed] = len(posSIndices)/len(S)
                    # ARI[idxMatrix, idxModel, idxClus, idxSeed] = round(sklearn.metrics.adjusted_rand_score(shufOdTrajLabels[shufRefTrajIndices], trajLabels), 3)
                # t = Time('{} model is done'.format(title))
            t = Time('idxSeed {} is done'.format(idxSeed))

    try:
        os.mkdir('./data/'+dataName+"_output")
    except:
        pass

    if not test:
        for (evalMatrix, evalFunc, evalName) in evalMeasures:
            pickle_out = open(f'./data/{dataName}_output/{dataName}_O{str(nClusStart)}-D{str(nClusEnd)}_{evalName}.pickle', "wb")
            pickle.dump(evalMatrix, pickle_out)
            pickle_out.close()            
        # pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_ARI.pickle", "wb")
        # pickle.dump(ARI, pickle_out)
        # pickle_out.close()
        # pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_avgS.pickle", "wb")
        # pickle.dump(avgS, pickle_out)
        # pickle_out.close()
        # pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_posSRatio.pickle", "wb")
        # pickle.dump(posSRatio, pickle_out)
        # pickle_out.close()

    cprint('\n "tableResults.csv" not saved. Save it manually.\n', color='red', on_color='on_yellow')
    tableResults = []
        # tableResults=[['dataName', 'dist', 'distParam', 'algo', 'algoParam', 'nClus', 'iter', 'ARI', 'avgS', 'posSRatio']]
        # for idxDist, (s, alpha) in enumerate([('dtw', -1), ('lcss', 1), ('lcss', 2), ('lcss', 3), ('lcss', 5), ('lcss', 7), ('lcss', 10), ('pf', 0.1), ('pf', 0.2), ('pf', 0.3), ('pf', 0.4)]):
        #     for idxModel, (A, beta) in enumerate([('kmedoids', 'None'), ('kmeans', 'None'), ('agglo', 'complete'), ('agglo', 'average'), ('agglo', 'single'), ('spectral', 'None'), ('OPTICS', 'None'), ('DBSCAN', 'None')]):
        #         for idxK, k in enumerate(list(range(2,7))):
        #             for iter in (range(3)):
        #                 tableResults.append([dataName, s, alpha, A, beta, k, iter, ARI[idxDist][idxModel][idxK][iter], avgS[idxDist][idxModel][idxK][iter], posSRatio[idxDist][idxModel][idxK][iter]])
        # savetxt('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+'_tableResults.csv', X=tableResults, delimiter=',', fmt ='% s')

    for idxMatrix, (distMatrix, distName) in enumerate(tempDistMatrices):
        cprint(distName, color='green', on_color='on_grey')
        plt.figure(figsize=(24,13))
        for idxEval, (evalMatrix, evalFunc, evalName) in enumerate(evalMeasures):
            fig = plt.subplot(len(evalMeasures)//4+1,4,idxEval+1)
            for i in range(len(modelList)):
                fig.plot(clusRange, np.nanmean(evalMatrix[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
                fig.fill_between(clusRange, np.nanmean(evalMatrix[idxMatrix, i], axis=-1)-np.nanstd(evalMatrix[idxMatrix, i], axis=-1), np.nanmean(evalMatrix[idxMatrix, i], axis=-1)+np.nanstd(evalMatrix[idxMatrix, i], axis=-1), color=colormap[i], alpha=0.3)#, label=modelList[i][1])
            # fig.set_ylim(0,1)
            fig.set_xlim(0,max(clusRange))
            fig.legend(loc='lower left')
            fig.set_title(evalName)

        # fig = plt.subplot(1,3,2)
        # for i in range(len(modelList)):
        #     fig.plot(clusRange, np.nanmean(avgS[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
        #     fig.fill_between(clusRange, np.nanmean(avgS[idxMatrix, i], axis=-1)-np.nanstd(avgS[idxMatrix, i], axis=-1), np.nanmean(avgS[idxMatrix, i], axis=-1)+np.nanstd(avgS[idxMatrix, i], axis=-1), color=colormap[i], alpha=0.3)#, label=modelList[i][1])
        # fig.set_ylim(-1,1)
        # fig.set_xlim(0,max(clusRange))
        # fig.legend(loc='lower left')
        # fig.set_title("average silhouette values")

        # fig = plt.subplot(1,3,3)
        # for i in range(len(modelList)):
        #     fig.plot(clusRange, np.nanmean(posSRatio[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
        #     fig.fill_between(clusRange, np.nanmean(posSRatio[idxMatrix, i], axis=-1)-np.nanstd(posSRatio[idxMatrix, i], axis=-1), np.nanmean(posSRatio[idxMatrix, i], axis=-1)+np.nanstd(posSRatio[idxMatrix, i], axis=-1), alpha=0.3, color=colormap[i])#, label=modelList[i][1])
        # fig.set_ylim(0,1)
        # fig.set_xlim(0,max(clusRange))
        # fig.legend(loc='lower left')
        # fig.set_title("positive sil. value ratio")

        # plt.title(distName, color='w')
        plt.savefig('./data/'+dataName+"_output/"+dataName+'_'+distName[:-4]+"_graphs.pdf", dpi=300)

        plt.show()
    
    return evalMeasures, tableResults
