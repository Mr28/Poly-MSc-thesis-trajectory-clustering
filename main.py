# -*- coding: utf-8 -*-

dataName = "inD1"                   #### ["inD1_300", "inD1_full", "inD2_full", "NGSIM_1", "NGSIM_5"]

try:
    import functions
    from functions import *
except:
    pass

trajList, trajectories, nTraj = DatabaseSection(dataName=dataName, resetDatabase=False, pickleInDatabase=False, pickleInAllTrajectories=False, maxTrajLength=1800)

distMatrices = DistMatricesSection(trajList=trajList, trajectories=trajectories, nTraj=nTraj, similarityMeasure=['lcss', 'dtw', 'pf'], pickleInDistMatrix=False)

_, _, _, _ = OdClustering(funcTrajectories=trajectories, nClusOriginSet=[i for i in range(1,30)], nClusDestSet=[i for i in range(1,30)], visualize=False)

startLabels, endLabels, startPoints, endPoints = OdClustering(funcTrajectories=trajectories, nClusOriginSet=[15], nClusDestSet=[8], visualize=True)
refTrajIndices, odTrajLabels = OdMajorClusters(startLabels=startLabels, endLabels=endLabels, threshold=10, visualize=True, test=True, load=False)

ARIs, avgSs, posSRatios = Main(clusRange=list(range(2,7)), nIter=3, modelList=None, dataName=dataName, test=True)
