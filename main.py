# -*- coding: utf-8 -*-

dataName = "NGSIM2"                   #### ["inD1_300", "inD1_full", "inD2_full", "NGSIM_1", "NGSIM_5"]
try:
    import functions
    from functions import *
except:
    cprint('file functions.py not found', color='red', on_color='on_grey')

trajList, trajectories, nTraj = DatabaseSection(nSample=500, dataName=dataName, resetDatabase=False, pickleInDatabase=False, pickleInAllTrajectories=False, maxTrajLength=1800, plot=True)
distMatrices = DistMatricesSection(trajList=trajList, trajectories=trajectories, nTraj=nTraj, dataName=dataName,
                            similarityMeasure=[['GuLcss',[1, 2, 3, 5, 7, 10]], ['GuDtw',[-1]], ['GuPf',[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]], ['GuEdr', [1, 2, 3, 5, 7, 10]],
                                                # ['Lcss',[1, 2, 3, 5, 7, 10]], ['Dtw', [-1]], ['Edr', [1, 2, 3, 5, 7, 10]], 
                                                ['Hausdorf', [-1]], ['Sspd', [-1]],
                                                # ['Frechet', [-1]]
                                                ],
                                   pickleInDistMatrix=False)
_, _, _, _, _, _ = OdClustering(funcTrajectories=trajectories, shuffle=True, nIter=10, nClusOriginSet=[i for i in range(1,30)], nClusDestSet=[i for i in range(1,30)], visualize=False, dataName=dataName)

#### Pick "nClusOriginSet" & "nClusDestSet" carefully!
nClusOriginSet, nClusDestSet , threshold= [9], [8], 10
startLabels, endLabels, startPoints, endPoints, nClusStart, nClusEnd = OdClustering(nClusOriginSet=nClusOriginSet, nClusDestSet=nClusDestSet, funcTrajectories=trajectories, shuffle=False, nIter=1, visualize=True, dataName=dataName)
refTrajIndices, odTrajLabels = OdMajorClusters(trajectories=trajectories, startLabels=startLabels, endLabels=endLabels, threshold=threshold, visualize=True, test=True, load=False, dataName=dataName)

clusRange, nIter, test = list(range(2, 30)), 10, False
evalMeasures, tableResults = Main(clusRange=clusRange, nIter=nIter, test=test, distMatrices=distMatrices, trajectories=trajectories, odTrajLabels=odTrajLabels, refTrajIndices=refTrajIndices, nClusStart=nClusStart, nClusEnd=nClusEnd, modelList=None, dataName=dataName,
                                  evalNameList = ['Completeness', 'Homogeneity', 'V', 'ARI', 'AMI', 'FMI', 'S', 'CHI', 'DBI']
                                  )
