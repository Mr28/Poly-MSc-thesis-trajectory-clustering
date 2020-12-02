# -*- coding: utf-8 -*-

import datetime, sqlite3, os, os.path, scipy, math, pickle, sys, random, shlex, subprocess
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
from dtw import *
shCommand("pip install scikit-learn-extra")
import sklearn_extra
import sklearn_extra.cluster

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


def SqlRead(files, dataName="inD1", database="inD_database.db"):
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
            if dataName.startswith("inD1"):
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
def TrajList(database="inD_database.db", intId=5, dataName="inD1", maxTrajLength=1800):
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
    return trajList

def Traj(tupleId, database="inD_database.db", dataName="inD1", intXMin=None, intXMax=None, intYMin=None, intYMax=None):
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


def DatabaseSection(dataName="inD1", resetDatabase=False, pickleInDatabase=False, pickleInAllTrajectories=False, test=True, maxTrajLength=1800, plot=False):

    database = dataName+"_database.db"

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
            SqlRead(files, dataName, database)

    intXMin, intXMax = None, None
    intYMin, intYMax = None, None
    if dataName.startswith("inD1"):
        # startTraj = 0
        # nTraj = 300
        intId = 0
    elif dataName.startswith("inD2"):
        # startTraj = 0
        # nTraj = 2033
        intId = 0
    elif dataName.startswith("inD3"):
        # startTraj = 2033
        # nTraj = 7784
        intId = 0
    elif dataName.startswith("inD4"):
        # startTraj = 0
        # nTraj = 300
        intId = 0
    elif dataName == "NGSIM1":
        # startTraj = 0
        # nTraj = 1370
        intId = 1
        intXMin, intXMax = -35, 35
        intYMin, intYMax = 120, 240
    elif dataName == "NGSIM5":
        # startTraj = 0
        # nTraj = 1319
        intId = 5
        intXMin, intXMax = -40, 40
        intYMin, intYMax = 1980, 2080
    else:
        cprint(text="""*\n*\n*\n Incorrect dataName input. It should be either: inD1, inD2, inD3, inD4, NGSIM1 or NGSIM5.\n*\n*\n*"""
            , color='red', on_color='on_grey')

    trajList = TrajList(database, intId=intId, dataName=dataName, maxTrajLength=maxTrajLength)

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
            if trajList[i,0] == 0:
                tr = Traj(trajList[i], database, dataName=dataName, intXMin=intXMin, intXMax=intXMax, intYMin=intYMin, intYMax=intYMax)
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


def DistMatricesSection(trajList, trajectories, nTraj=None, dataName="inD1", similarityMeasure=['lcss', 'dtw', 'pf'], pickleInDistMatrix=False, test=True, lcssParamList=[1, 2, 3, 5, 7, 10], pfParamList=[0.1, 0.2, 0.3, 0.4], maxTrajLength=1800):

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
        L = np.zeros((maxTrajLength+1,maxTrajLength+1)) ######################

        if 'lcss' in similarityMeasure:
            for e in lcssParamList:
                lcssMatrix = np.zeros((nTraj,nTraj))
                # startTime = Time(prnt=False)
                # for i in range(nTraj):
                #     # t = Time('trajectory {}'.format(i))
                #     for j in range(i+1):
                #         X = trajectories[i]
                #         Y = trajectories[j]
                #         M = spatial.distance.cdist(X, Y, metric='euclidean')
                #         # L = np.zeros((X.shape[0]+1, Y.shape[0]+1))
                #         lcssMatrix[i,j] = Lcs(X, Y, M, L, e)
                #         lcssMatrix[j,i] = lcssMatrix[i,j]
                # endTime = Time(prnt=False)
                # print('LCSS, run time: '+str(endTime-startTime))
                # # print(lcssMatrix)
                # ### pickle
                # pickle_out = open("./data/lcssMatrix_e"+str(e)+".pickle", "wb")
                # pickle.dump(lcssMatrix, pickle_out)
                # pickle_out.close()

                startTime = Time(prnt=False)
                for i in range(nTraj):
                    for j in range(i+1):
                        X = trajectories[i]
                        Y = trajectories[j]
                        # M = spatial.distance.cdist(X, Y, metric='euclidean')
                        L = np.zeros((X.shape[0]+1, Y.shape[0]+1))
                        lcssMatrix[i,j] = GuLcs(X, Y, L, e)
                        lcssMatrix[j,i] = lcssMatrix[i,j]
                normLcssMatrix = np.zeros(lcssMatrix.shape)
                for i in range(lcssMatrix.shape[0]):
                    for j in range(lcssMatrix.shape[1]):
                        normLcssMatrix[i,j] = 1 - lcssMatrix[i,j]/(min(len(trajectories[i]),len(trajectories[j])))
                distMatrix = normLcssMatrix
                endTime = Time(prnt=False)
                print('guvec LCSS e{}, run time:{}'.format(e, str(endTime-startTime)))
                # print(lcssMatrix)
                ## pickle
                # pickle_out = open("./data/gulcssMatrix_e{}.pickle".format(e), "wb")
                # pickle.dump(lcssMatrix, pickle_out)
                # pickle_out.close()
                savetxt("./data/distMatricesFolder/gulcssMatrixNorm{:s}_e{:02d}.csv".format(dataName, e), normLcssMatrix, delimiter=',')

        if 'dtw' in similarityMeasure:
            dtwMatrix = np.zeros((nTraj,nTraj))
            # startTime = Time(prnt=False)
            # for i in range(nTraj):
            #     # t = Time('trajectory {}'.format(i))
            #     for j in range(i+1):
            #         X = trajectories[i]
            #         Y = trajectories[j]
            #         M = spatial.distance.cdist(X, Y, metric='euclidean')
            #         L = np.full((X.shape[0]+1, Y.shape[0]+1), np.inf)
            #         dtwMatrix[i,j] = Dtw(X, Y, M, L)
            #         dtwMatrix[j,i] = dtwMatrix[i,j]
            # endTime = Time(prnt=False)
            # print('DTW, run time: '+str(endTime-startTime))
            # # print(dtwMatrix)
            # ### pickle
            # pickle_out = open("./data/dtwMatrix.pickle", "wb")
            # pickle.dump(dtwMatrix, pickle_out)
            # pickle_out.close()

            # startTime = Time('start geuvec-DTW Matrix calculation')
            for i in range(nTraj):
                for j in range(i+1):
                    X = trajectories[i]
                    Y = trajectories[j]
                    # M = spatial.distance.cdist(X, Y, metric='euclidean')
                    L = np.full((X.shape[0]+1, Y.shape[0]+1), np.inf)
                    dtwMatrix[i,j] = GuDtw(X, Y, L)
                    dtwMatrix[j,i] = dtwMatrix[i,j]
            normDtwMatrix = np.zeros(dtwMatrix.shape)
            for i in range(dtwMatrix.shape[0]):
                for j in range(dtwMatrix.shape[1]):
                    normDtwMatrix[i,j] = dtwMatrix[i,j]/(min(len(trajectories[i]),len(trajectories[j])))
            distMatrix = normDtwMatrix
            endTime = Time(prnt=False)
            print('guvec DTW, run time:{}'.format(str(endTime-startTime)))
            # print(dtwMatrix)
            ### pickle
            # pickle_out = open("./data/guDtwMatrix.pickle", "wb")
            # pickle.dump(dtwMatrix, pickle_out)
            # pickle_out.close()
            savetxt("./data/distMatricesFolder/gudtwMatrixNorm{}.csv".format(dataName), normDtwMatrix, delimiter=',')

        if 'pf' in similarityMeasure:
            for r in pfParamList:
                pfMatrix = np.zeros((nTraj,nTraj))
                # startTime = Time('start geuvec-PF Matrix calculation')
                for i in range(nTraj):
                    # print(i)
                    for j in range(nTraj):
                        # print(j)
                        X = trajectories[i]
                        Y = trajectories[j]
                        # M = spatial.distance.cdist(X, Y, metric='euclidean')
                        L = np.full((X.shape[0]+1, Y.shape[0]+1), np.inf)
                        # print(X.shape)
                        # print(Y.shape)
                        # print(L.shape)
                        # print(r)
                        pfMatrix[i,j] = GuPf(X, Y, L, r)
                        # dtwMatrix[j,i] = dtwMatrix[i,j]
                endTime = Time(prnt=False)
                print('guvec PF r{}, run time:{}'.format(r, str(endTime-startTime)))
                # print(dtwMatrix)
                ### pickle
                # pickle_out = open("./data/guPfMatrix.pickle", "wb")
                # pickle.dump(pfMatrix, pickle_out)
                # pickle_out.close()
                savetxt("./data/distMatricesFolder/gupfMatrix{}_r{}.csv".format(dataName, r), pfMatrix, delimiter=',')

    shCommand("rm ./data/distMatricesZip.zip")
    shCommand("zip -o -r ./data/distMatricesZip ./data/distMatricesFolder/")

    files = (f for f in os.listdir("./data/distMatricesFolder/") if "Matrix" in f and '.csv' in f)
    files = sorted(files)

    distMatrices = []
    for f in files:
        for measure in similarityMeasure:
            if measure in f:
                distMatrix = loadtxt("./data/distMatricesFolder/"+f, delimiter=',')
                distMatrices.append((distMatrix, f))

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


def OdClustering(funcTrajectories, nTraj=None, modelList=None, nClusOriginSet=[4], nClusDestSet=[4], visualize=False, test=True, darkTheme=False):

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

    # nClusOriginSet, nClusDestSet = [i for i in range(1,30)], [i for i in range(1,30)]
    # nClusOriginSet, nClusDestSet, visualize = [11], [4], True

    startAvgDists = np.zeros(len(nClusOriginSet))
    endAvgDists = np.zeros(len(nClusDestSet))
    minTrajX = min([min(tr[:, 0]) for tr in funcTrajectories])
    minTrajY = min([min(tr[:, 1]) for tr in funcTrajectories])
    maxTrajX = max([max(tr[:, 0]) for tr in funcTrajectories])
    maxTrajY = max([max(tr[:, 1]) for tr in funcTrajectories])

    if modelList == None:
        modelList = [
            # (sklearn_extra.cluster.KMedoids(metric='euclidean'), 'KMedoids')
            # ,(sklearn.cluster.KMeans(precompute_distances='auto', n_jobs=-1), 'KMeans')
            # ,(sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='ward'), 'ward Agglo-Hierarch')
            # ,(sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='complete'), 'complete Agglo-Hierarch')
            (sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='average'), 'average Agglo-Hierarch')
            # ,(sklearn.cluster.AgglomerativeClustering(affinity='euclidean', linkage='single'), 'single Agglo-Hierarch')
            # ,(sklearn.cluster.Birch(threshold=0.5, branching_factor=50), 'BIRCH')
            # ,(sklearn.mixture.GaussianMixture(covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=200, n_init=5, init_params='kmeans'), 'GMM')
            # ,(sklearn.cluster.SpectralClustering(affinity='rbf', n_jobs=-1), 'Spectral')
            # ,(sklearn.cluster.OPTICS(metric='minkowski', n_jobs=-1), 'OPTICS')
            # ,(sklearn.cluster.DBSCAN(metric='euclidean', n_jobs=-1), 'DBSCAN')
            ]

    for model, title in modelList:
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
            startModel = model1.fit(startPoints[:nTraj])
            try:
                startLabels = list(startModel.labels_)
            except:
                startLabels = list(startModel.predict(startPoints[:nTraj]))

            model2 = model
            try:
                if title=="GMM":
                    model2.n_components = nClusDestSet[i]
                else:
                    model2.n_clusters = nClusDestSet[i]
            except:
                pass
            endModel = model2.fit(endPoints[:nTraj])
            try:
                endLabels = list(endModel.labels_)
            except:
                endLabels = list(endModel.predict(endPoints[:nTraj]))

            startLabelList = list(set(startLabels))
            endLabelList = list(set(endLabels))
            nClusStart = len(startLabelList)
            nClusEnd = len(endLabelList)

            try:
                startCenters = startModel.cluster_centers_
                endCenters = endModel.cluster_centers_
            except:
                startCenters = np.zeros((nClusStart,2))
                endCenters = np.zeros((nClusEnd,2))
                for k in range(nClusStart):
                    clusPoints = startPoints[startLabels == startLabelList[k]]
                    startCenters[k] = np.average(clusPoints, axis=0)
                for k in range(nClusEnd):
                    clusPoints = endPoints[endLabels == endLabelList[k]]
                    endCenters[k] = np.average(clusPoints, axis= 0)    

            startDistSum = 0
            for k in range(nClusStart):
                clusPoints = startPoints[startLabels == startLabelList[k]]
                for point in clusPoints:
                    startDistSum += ((point[0]-startCenters[k,0])**2 + (point[1]-startCenters[k,1])**2)**0.5
            startAvgDists[i] = startDistSum/len(startLabels)
            # print(startAvgDists[i])

            endDistSum = 0
            for k in range(nClusEnd):
                clusPoints = endPoints[endLabels == endLabelList[k]]
                for point in clusPoints:
                    endDistSum += ((point[0]-endCenters[k,0])**2 + (point[1]-endCenters[k,1])**2)**0.5
            endAvgDists[i] = endDistSum/len(endLabels)
            # print(endAvgDists[i])


            if visualize:# and len(nClusOriginSet)>1 and len(nClusDestSet)>1:
                print(nClusStart, nClusEnd)
                try:
                    print("Calinski Harabasz: {} & {}".format(
                        sklearn.metrics.calinski_harabasz_score(startPoints[:nTraj], startLabels)
                        ,sklearn.metrics.calinski_harabasz_score(endPoints[:nTraj], endLabels)))
                except:
                    pass

                print("nClusStart={}, nClusEnd={}".format(nClusStart, nClusEnd))
                cmap = list(colors.TABLEAU_COLORS)
                colormap = cmap
                repeat = max(nClusStart,nClusEnd)//len(cmap)
                for k in range(repeat):
                    colormap = colormap + cmap

                plt.figure(figsize=(16,6))
                fig = plt.subplot(1,2,1)
                for k in range(nTraj):
                    fig.scatter(startPoints[k,0], startPoints[k,1], c=colormap[startLabels[k]])
                fig.scatter(startCenters[:,0], startCenters[:,1], c='black')
                fig.set_title("Origin clusters")
                fig.tick_params(colors=tickColors)
                fig = plt.subplot(1,2,2)
                for k in range(nTraj):
                    fig.scatter(endPoints[k,0], endPoints[k,1], c=colormap[endLabels[k]])
                fig.scatter(endCenters[:,0], endCenters[:,1], c='black')
                fig.tick_params(colors=tickColors)
                plt.show()

    if len(nClusOriginSet)>1 or len(nClusDestSet)>1:
        plt.figure(figsize=(16,6))
        fig = plt.subplot(1,2,1)
        fig.set_ylim(0,50)
        fig.scatter(nClusOriginSet, startAvgDists)
        fig.tick_params(colors=tickColors)
        fig.set_title("Origin clusters")
        fig = plt.subplot(1,2,2)
        fig.set_ylim(0,50)
        fig.scatter(nClusDestSet, endAvgDists)
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

    return startLabels, endLabels, startPoints, endPoints, nClusStart, nClusEnd


def TestOdClustering(funcTrajectories, nTraj=None, modelList=None, nClusOriginSet=[4], nClusDestSet=[4], modelNames=['average Agglo-Hierarch'], nIter=1, funcTrajectories=None, visualize=False, shuffle=False, test=True, darkTheme=False):

    if darkTheme:
        tickColors = 'white'
    else:
        tickColors = 'black'

    if funcTrajectories == None:
        funcTrajectories = trajectories

    startPoints = np.zeros((nTraj,2))
    endPoints = np.zeros((nTraj,2))
    for i in range(nTraj):
        tr = funcTrajectories[i]
        startPoints[i] = tr[0]
        endPoints[i] = tr[-1]

    # nClusOriginSet, nClusDestSet = [i for i in range(1,30)], [i for i in range(1,30)]
    # nClusOriginSet, nClusDestSet, visualize = [11], [4], True

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
            if shuffle:
                random.shuffle(shufIndices)
            shufStartPoints = np.zeros_like(startPoints)
            shufEndPoints = np.zeros_like(endPoints)
            for i in range(len(shufIndices)):
                shufStartPoints[shufIndices[i]] = startPoints[i]
                shufEndPoints[shufIndices[i]] = endPoints[i]
            shufStartPoints = startPoints.copy()
            shufEndPoints = endPoints.copy()
            if shuffle:
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

                startLabelList = list(set(startLabels))
                endLabelList = list(set(endLabels))
                nClusStart = len(startLabelList)
                nClusEnd = len(endLabelList)

                try:
                    startCenters = startModel.cluster_centers_
                    endCenters = endModel.cluster_centers_
                except:
                    startCenters = np.zeros((nClusStart,2))
                    endCenters = np.zeros((nClusEnd,2))
                    for k in range(nClusStart):
                        clusPoints = shufStartPoints[startLabels == startLabelList[k]]
                        startCenters[k] = np.average(clusPoints, axis=0)
                    for k in range(nClusEnd):
                        clusPoints = shufEndPoints[endLabels == endLabelList[k]]
                        endCenters[k] = np.average(clusPoints, axis= 0)    

                startDistSum = 0
                for k in range(nClusStart):
                    clusPoints = shufStartPoints[startLabels == startLabelList[k]]
                    for point in clusPoints:
                        startDistSum += ((point[0]-startCenters[k,0])**2 + (point[1]-startCenters[k,1])**2)**0.5
                startAvgDists[i, iter] = startDistSum/len(startLabels)
                # print(startAvgDists[i])

                endDistSum = 0
                for k in range(nClusEnd):
                    clusPoints = shufEndPoints[endLabels == endLabelList[k]]
                    for point in clusPoints:
                        endDistSum += ((point[0]-endCenters[k,0])**2 + (point[1]-endCenters[k,1])**2)**0.5
                endAvgDists[i, iter] = endDistSum/len(endLabels)
                # print(endAvgDists[i])


                if visualize:# and len(nClusOriginSet)>1 and len(nClusDestSet)>1:
                    print(nClusStart, nClusEnd)
                    try:
                        print("Calinski Harabasz: {} & {}".format(
                            sklearn.metrics.calinski_harabasz_score(shufStartPoints, startLabels)
                            ,sklearn.metrics.calinski_harabasz_score(shufEndPoints, endLabels)))
                    except:
                        pass

                    print("nClusStart={}, nClusEnd={}".format(nClusStart, nClusEnd))
                    cmap = list(colors.TABLEAU_COLORS)
                    colormap = cmap
                    repeat = max(nClusStart,nClusEnd)//len(cmap)
                    for k in range(repeat):
                        colormap = colormap + cmap

                    plt.figure(figsize=(16,6))
                    fig = plt.subplot(1,2,1)
                    for k in range(nTraj):
                        fig.scatter(shufStartPoints[k,0], shufStartPoints[k,1], c=colormap[startLabels[k]])
                    fig.scatter(startCenters[:,0], startCenters[:,1], c='black')
                    fig.set_title("Origin clusters")
                    fig.tick_params(colors=tickColors)
                    fig = plt.subplot(1,2,2)
                    for k in range(nTraj):
                        fig.scatter(shufEndPoints[k,0], shufEndPoints[k,1], c=colormap[endLabels[k]])
                    fig.scatter(endCenters[:,0], endCenters[:,1], c='black')
                    fig.tick_params(colors=tickColors)
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


def Main(distMatrices, trajectories, odTrajLabels, refTrajIndices, nClusStart, nClusEnd, clusRange=list(range(2,15)), nIter=3, modelList=None, dataName="inD1", test=True, seed=0.860161037286291):

    t = Time('start')
    random.seed(seed)
    # test, nIter, clusRange = False, 10, list(range(2,30))
    # test, nIter, clusRange = True, 3, list(range(2,15))
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
        tempDistMatrices = [(distMatrix, f) for (distMatrix, f) in distMatrices if f in ["gulcssMatrixNorm"+dataName+"_e07.csv", "gudtwMatrixNorm"+dataName+".csv", "gupfMatrix"+dataName+"_r0.2.csv"]]

    ARIs = np.zeros((len(tempDistMatrices), len(modelList), len(clusRange), len(randArray)))
    avgSs = np.zeros((len(tempDistMatrices), len(modelList), len(clusRange), len(randArray)))
    posSRatios = np.zeros((len(tempDistMatrices), len(modelList), len(clusRange), len(randArray)))

    for idxMatrix, (distMatrix, f) in enumerate(tempDistMatrices):
        t = Time(f, color='yellow')

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


            # ARIs = []
            # avgSs = []
            # posSRatios = []
            for idxModel, (model, title) in enumerate(modelList):
                for idxClus, nClus in enumerate(clusRange):
                    # model1 = model.copy()
                    model.n_clusters = nClus
                    # model = sklearn.cluster.AgglomerativeClustering(affinity='precomputed', n_clusters=nClus, linkage='complete')
                    S, closestCluster, labels, subDistMatrix, shufSubDistMatrix = Silhouette(model=model, distMatrix=shufDistMatrix, trajIndices=shufRefTrajIndices)
                    # trajLabels = model.labels_
                    trajLabels = labels
                    ari = round(sklearn.metrics.adjusted_rand_score(shufOdTrajLabels[shufRefTrajIndices], trajLabels), 3)
                    # ari = round(sklearn.metrics.adjusted_rand_score(shufOdTrajLabels, trajLabels), 3)
                    ARIs[idxMatrix, idxModel, idxClus, idxSeed] = ari
                    avgSs[idxMatrix, idxModel, idxClus, idxSeed] = np.mean(S)
                    posSIndices = np.where(S>0)[0]
                    posSRatios[idxMatrix, idxModel, idxClus, idxSeed] = len(posSIndices)/len(S)
                # t = Time('{} model is done'.format(title))
            t = Time('idxSeed {} is done'.format(idxSeed))

    try:
        os.mkdir('./data/'+dataName+"_output")
    except:
        pass

    if not test:
        pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_ARIs.pickle", "wb")
        pickle.dump(ARIs, pickle_out)
        pickle_out.close()
        pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_avgSs.pickle", "wb")
        pickle.dump(avgSs, pickle_out)
        pickle_out.close()
        pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_posSRatios.pickle", "wb")
        pickle.dump(posSRatios, pickle_out)
        pickle_out.close()

    for idxMatrix, (distMatrix, f) in enumerate(tempDistMatrices):
        cprint(f, color='green', on_color='on_grey')
        plt.figure(figsize=(24,6))
        fig = plt.subplot(1,3,1)
        for i in range(len(modelList)):
            fig.plot(clusRange, np.nanmean(ARIs[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
            fig.fill_between(clusRange, np.nanmean(ARIs[idxMatrix, i], axis=-1)-np.nanstd(ARIs[idxMatrix, i], axis=-1), np.nanmean(ARIs[idxMatrix, i], axis=-1)+np.nanstd(ARIs[idxMatrix, i], axis=-1), color=colormap[i], alpha=0.3)#, label=modelList[i][1])
        fig.set_ylim(0,1)
        fig.set_xlim(0,max(clusRange))
        fig.legend(loc='lower left')
        fig.set_title("ARI")

        fig = plt.subplot(1,3,2)
        for i in range(len(modelList)):
            fig.plot(clusRange, np.nanmean(avgSs[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
            fig.fill_between(clusRange, np.nanmean(avgSs[idxMatrix, i], axis=-1)-np.nanstd(avgSs[idxMatrix, i], axis=-1), np.nanmean(avgSs[idxMatrix, i], axis=-1)+np.nanstd(avgSs[idxMatrix, i], axis=-1), color=colormap[i], alpha=0.3)#, label=modelList[i][1])
        fig.set_ylim(-1,1)
        fig.set_xlim(0,max(clusRange))
        fig.legend(loc='lower left')
        fig.set_title("average silhouette values")

        fig = plt.subplot(1,3,3)
        for i in range(len(modelList)):
            fig.plot(clusRange, np.nanmean(posSRatios[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
            fig.fill_between(clusRange, np.nanmean(posSRatios[idxMatrix, i], axis=-1)-np.nanstd(posSRatios[idxMatrix, i], axis=-1), np.nanmean(posSRatios[idxMatrix, i], axis=-1)+np.nanstd(posSRatios[idxMatrix, i], axis=-1), alpha=0.3, color=colormap[i])#, label=modelList[i][1])
        fig.set_ylim(0,1)
        fig.set_xlim(0,max(clusRange))
        fig.legend(loc='lower left')
        fig.set_title("positive sil. value ratio")
        # plt.title(f, color='w')
        plt.savefig('./data/'+dataName+"_output/"+dataName+'_'+f[:-4]+"_graphs.pdf", dpi=300)

        plt.show()
    
    return ARIs, avgSs, posSRatios
