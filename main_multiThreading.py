import networkx as nx
import threading
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import asyncio
from random import randint
import sys
import numpy as np
import redis
import json

# node, edge 정보를 담은 csv 파일과 네트워크 performance를 평가하는 정보를 담을 final csv파일
topology = sys.argv[1]
edge = sys.argv[2]
final = sys.argv[3]
# 네트워크 시뮬레이션을 몇 번할지
totalSimulNum=int(sys.argv[4])
# 대충 파일이름 formating해줌
fileNum = topology[12:13]  # 파일이름에 따라 달라질듯
# redis db 연결
rd=redis.StrictRedis(host='localhost', port=6379, db=0)
# 이후에 있을 시뮬레이션에 필요한 정보들을 추출
nodeData = pd.read_csv(topology)  # node 관련 데이터 추출
edgeData = pd.read_csv(edge)  # edge 관련 데이터 추출
finalData = pd.read_csv(final)  # 이번 시뮬레이션의 결과를 저장하기 위한 추출
node = nodeData['node']
xPos = nodeData['xPos']
yPos = nodeData['yPos']
nodeType = nodeData['nodeType']
relatedCache = nodeData['relatedCache']
relatedServer = nodeData['relatedServer']
groupProperty = nodeData['groupProperty']
groupProb = nodeData['groupProb']
nodeList = node.values.tolist()
xPosList = xPos.values.tolist()
yPosList = yPos.values.tolist()
nodeTypeList = nodeType.values.tolist()
relatedCacheList = relatedCache.values.tolist()
relatedServerList = relatedServer.values.tolist()
groupPropertyList = groupProperty.values.tolist()
groupProbList = groupProb.values.tolist()
edgeA = edgeData['nodeA']
edgeB = edgeData['nodeB']
edgeNum = edgeData['edgeNum']
edgeAList = edgeA.values.tolist()
edgeBList = edgeB.values.tolist()
edgeNumList = edgeNum.values.tolist()
x = finalData['xPos']
y = finalData['yPos']
finalRttData = finalData['finalRttData']
xList = x.values.tolist()
yList = y.values.tolist()
finalRttDataList = finalRttData.values.tolist()
currentSimulNum=0   # 현재 simulation을 끝낸 횟수

# 그래프를 만들고 node 관련 csv 파일에서 가져온 node 정보로 node 추가
G = nx.Graph()
G.add_nodes_from(nodeList)
# 각 node들에게 x, y Position을 attribute로 부여한다
for node in nodeList:
    G.nodes[node]['xPos'] = xPosList[nodeList.index(node)]
    G.nodes[node]['yPos'] = yPosList[nodeList.index(node)]

# client node(server node, cache server node, router node를 제외한 노드)를 담아놓는 리스트
clientList = []
for i in nodeList:
    if nodeTypeList[i] == 0:
        clientList.append(i)

# TODO 여기서부터는 한 cache server 위치에 대한 것
xNp=np.arange(0, 501, 100)
yNp=np.arange(0, 401, 100)
for xCoord in xNp:
    for yCoord in yNp:
        currentSimulNum += 1
        print(currentSimulNum, "번째 시뮬레이션 시작")
        # cacheServer의 위치를 랜덤으로 배정하는 부분(위치의 x, y position의 범위는 네트워크 토폴로지의 베이스가 된 사진의 크기와 관련있다.)
        cacheServerNodeNum = 0
        for node in nodeList:
            if nodeTypeList[node] == 1:
                #if currentSimulNum == 1:
                    #xPosList[node] = 401
                    #yPosList[node] = 314
                #else:
                    #xPosList[node] = randint(0, 500)
                    #yPosList[node] = randint(0, 400)
                xPosList[node]=xCoord
                yPosList[node]=yCoord
                cacheServerNodeNum = node
                G.nodes[node]['xPos'] = xPosList[nodeList.index(node)]
                G.nodes[node]['yPos'] = yPosList[nodeList.index(node)]
        # 네트워크 시뮬레이션과 관련된 하이퍼 파라메터(총 시행할 통신의 횟수, 링크의 weight와 관련된 factor, caching server의 size)
        CONST_PACKETSERIES_LIMIT = 10000
        CONST_FACTOR = 0.0001
        CONST_CACHE_SIZE = 10
        CONST_SAMPLE_FOR_ONE_POSITION = 1
        tempSampleList = []  # 한 cache server position에 대한 시뮬레이션 결과들을 잠시 저장했다가 CONST_SAMPLE_FOR_ONE_POSITION 만큼 샘플링 후 평균을 낸다
        currentSampleNum = 0  # 현재까지 모인 한 cache server position에 대한 샘플 개수

        #TODO: cacheList, rttList같은 것들 쓰레드에 넘겨줄때는 deepcopy해서 2차원리스트의 원소인 리스트만 넘겨주고 값은 redis로 받아오기
        while currentSampleNum < CONST_SAMPLE_FOR_ONE_POSITION:
            endedPacketSeries = 0
            currentSampleNum += 1
            # 캐싱 서버의 역할을 할 리스트들의 초기화
            cacheList = []
            for node in nodeList:
                eachCache = []
                for i in range(0, CONST_CACHE_SIZE):
                    if nodeTypeList[node] == 1:
                        eachCache.append(randint(1, 100))
                    else:
                        eachCache.append(0)
                cacheList.append(eachCache)
                if nodeTypeList[node] == 1:
                    # 캐싱 서버 역할의 리스트들을 redis로 보낸다
                    nodeCacheKey = str(node) + " node cache"
                    nodeCacheList = cacheList[node]
                    jsonnodeCacheList = json.dumps(nodeCacheList, indent=4)
                    rd.set(nodeCacheKey, jsonnodeCacheList)
            # 각 노드들의 매 통신마다의 rtt를 기록할 리스트 초기화
            rttList = []
            for i in nodeList:
                eachRtt = []
                rttList.append(eachRtt)

            # 각 node들의 variety(같은 것을 request하는 request가 몇번인지), 총 통신 횟수가 몇번인지 저장하는 리스트들 초기화
            varietyList = []
            trialNumList = []
            for node in nodeList:
                if node not in clientList:
                    varietyList.append(-1)
                    trialNumList.append(1)
                else:
                    varietyList.append(0)
                    trialNumList.append(0)

            # edge들을 node의 x, y position을 바탕으로 weight를 정해준다
            for edgeNum in edgeNumList:
                nodeA = G.nodes[edgeAList[edgeNum]]
                nodeB = G.nodes[edgeBList[edgeNum]]
                xPosEdgeA = nodeA['xPos']
                yPosEdgeA = nodeA['yPos']
                xPosEdgeB = nodeB['xPos']
                yPosEdgeB = nodeB['yPos']
                edgeWeight = math.sqrt(math.pow((xPosEdgeA - xPosEdgeB), 2) + math.pow((yPosEdgeA - yPosEdgeB), 2))
                G.add_edge(edgeAList[edgeNum], edgeBList[edgeNum], weight=edgeWeight)


            # cache server의 값을 update할 필요가 있을 경우 실행하는 함수
            def updateCache(G, dst, realdst, payload):
                global CONST_FACTOR
                # dijkstra 알고리즘을 이용하여 cache server에서 origin server까지의 최적 path 찾기
                optimalPath = nx.dijkstra_path(G, dst, realdst, weight='weight')

                # optimalPath에 있는 node를 travel하기 위한 linkWeight의 합을 구한다
                linkWeight = 0
                for i in range(0, len(optimalPath) - 1):
                    linkWeight += G.edges[optimalPath[i], optimalPath[i + 1]]['weight']
                sleepTime = linkWeight * CONST_FACTOR
                time.sleep(sleepTime)
                time.sleep(sleepTime)

                # cacheList를 업데이트하기 위해 redis에서 가져온다
                nodeCacheKey = str(dst) + " node cache"
                cacheList=rd.get(nodeCacheKey).decode()
                cacheList=json.loads(cacheList)
                del cacheList[0]
                cacheList.append(payload)
                jsoncacheList=json.dumps(cacheList, indent=4)
                rd.set(nodeCacheKey, jsoncacheList)


            # 특정 통신의 절차를 진행하는 함수
            def packet(G, src, dst, realdst, payload, rttList):
                global endedPacketSeries
                global CONST_PACKETSERIES_LIMIT
                global CONST_FACTOR

                # packet 시작 확인용
                print("packet: starting for ", src, " to cache server ", dst, " and origin server ", realdst)

                # dijkstra 알고리즘을 이용하여 cache server까지의 최적 path 찾기
                optimalPath = nx.dijkstra_path(G, src, dst, weight='weight')

                # optimalPath에 있는 node를 travel하기 위한 linkWeight의 합을 구한다
                linkWeight = 0
                for i in range(0, len(optimalPath) - 1):
                    linkWeight += G.edges[optimalPath[i], optimalPath[i + 1]]['weight']

                # sleepTime을 node간 거리*factor로 하여 node들 간의 거리가 너무 멀거나 짧거나 길 경우 시뮬레이션 시간을 줄이거나 늘려준다
                sleepTime = linkWeight * CONST_FACTOR
                start = time.time()
                time.sleep(sleepTime)  # src에서 dst까지 packet 이동

                # 데이터가 cache에 있는지 알아보기 위해 redis에서 cacheList가져온다
                nodeCacheKey = str(dst) + " node cache"
                nodeCacheList=rd.get(nodeCacheKey).decode()
                nodeCacheList=json.loads(nodeCacheList)
                # dst node(cache server)에 찾는 데이터가 있는지 확인한다
                dataexistencebool = False
                if payload in nodeCacheList:
                    dataexistencebool = True
                else:
                    dataexistencebool = False

                time.sleep(sleepTime)  # dst에서 src까지 packet 이동

                # dijkstra 알고리즘을 이용하여 origin server까지의 최적 path 찾기
                optimalPath = nx.dijkstra_path(G, src, realdst, weight='weight')

                # optimalPath에 있는 node를 travel하기 위한 linkWeight의 합을 구한다
                linkWeight = 0
                for i in range(0, len(optimalPath) - 1):
                    linkWeight += G.edges[optimalPath[i], optimalPath[i + 1]]['weight']

                sleepTime = linkWeight * CONST_FACTOR
                # data가 있으면 sleepTime을 0으로 하여 client에서 본 서버까지 데이터 요청하는 시간을 없앤다
                if dataexistencebool:
                    sleepTime = 0
                else:  # updateStart와 updateEnd를 측정하여 updateCache하는동안의 시간을 측정한다
                    thread = threading.Thread(target=updateCache, args=(G, dst, realdst, payload))
                    thread.start()
                # src에서 본 서버에 데이터를 요청하는 시간 지연이지만 cache server에 데이터가 있다면 sleepTime이 0이기 때문에 없는 것과 같다
                time.sleep(sleepTime)
                time.sleep(sleepTime)

                end = time.time()
                rttList.append(end - start)  # 측정한 통신의 rtt를 기록

                # packet 종료 확인용
                print("packet: ending for ", src, " to cache server ", dst, " and origin server ", realdst)


            def main():  # asyncio는 한번에 하나의 run만 할 수 있기에 한번에 asyncio 리스트 객체에서 run을 시킨다
                threads = []
                for node in clientList:
                    thread = threading.Thread(target=nodeThread,
                                              args=(G, node, relatedCacheList[node], relatedServerList[node],
                                                    groupProbList[node], groupPropertyList[node]))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()


            def nodeThread(G, src, dst, realdst, groupProb, groupProperty):
                global endedPacketSeries
                global CONST_PACKETSERIES_LIMIT

                # multiThreading에서 dataframe을 넘겨주면 오류가 생길 수 있기에 varietyList, trialNumList를 대체할 변수 생성
                nodeVariety=0
                nodeTrialNum=0

                # multiThreading에서 dataframe을 넘겨주면 오류가 생길 수 있기에 noderttList를 여기서 생성
                nodeRttList=[]

                # nodeThread multithreading 시작 확인용
                print("nodeThread: multithreading starting for ", src, " to cache server ", dst, " and origin server ",
                      realdst)

                while (endedPacketSeries < CONST_PACKETSERIES_LIMIT):
                    endedPacketSeries += 1  # 현재까지 진행한 통신 횟수 기록
                    nodeTrialNum+=1

                    payload = randint(1, 100)
                    if payload <= groupProb:
                        payload = groupProperty
                        nodeVariety+=1

                    packet(G, src, dst, realdst, payload, nodeRttList)

                jsonnodeRttList=json.dumps(nodeRttList, indent=4)

                # nodeVariety, nodeTrialNum, rttList를 redis로 보냄
                varietyKey=str(src)+" node variety"
                trialNumKey=str(src)+" node trialNum"
                nodeRttKey=str(src)+" node rtt"
                rd.set(varietyKey, nodeVariety)
                rd.set(trialNumKey, nodeTrialNum)
                rd.set(nodeRttKey, jsonnodeRttList)

                # nodeThread multithreading 종료 확인용
                print("nodeThread: multithreading ending for ", src, " to cache server ", dst, " and origin server ",
                      realdst)


            main()
            #redis에 있는 cacheList들을 삭제한다
            for node in nodeList:
                if nodeTypeList[node] == 1:
                    nodeCacheKey = str(node) + " node cache"
                    rd.delete(nodeCacheKey)
            # redis에서 각 client node들의 rtt를 담은 list를 가져온다
            for node in clientList:
                nodeRttKey = str(node) + " node rtt"
                nodeRttList=rd.get(nodeRttKey).decode()
                nodeRttList=json.loads(nodeRttList)
                rttList[node]=nodeRttList
                rd.delete(nodeRttKey)
            # 기록했던 rtt를 바탕으로 각 node의 rtt의 평균을 구한다
            rttMeanList = []
            for nodeRttList in rttList:
                if len(nodeRttList) != 0:
                    rttSum = 0
                    for rtt in nodeRttList:
                        rttSum = rttSum + rtt
                    rttSum = rttSum / len(nodeRttList)
                    rttMeanList.append(rttSum)
                else:
                    rttMeanList.append(-1)

            # redis에 있던 값을 불러와서 List에 저장한다
            for node in clientList:
                varietyKey = str(node) + " node variety"
                trialNumKey = str(node) + " node trialNum"
                varietyList[node]=int(rd.get(varietyKey).decode())
                trialNumList[node]=int(rd.get(trialNumKey).decode())
                rd.delete(varietyKey)
                rd.delete(trialNumKey)

            # 각 노드의 variety를 구한다
            for node in nodeList:
                if node in clientList:
                    if trialNumList[node] != 0:
                        varietyList[node] = varietyList[node] / trialNumList[node]
                    else:
                        varietyList[node] = varietyList[node] / 1

            print(cacheList)
            print(rttMeanList)
            print(varietyList)
            # 네트워크의 performance인 각 노드들의 평균 rtt들의 평균을 구하여 cache server 위치와 함께 미리 만든 리스트에 넣는다
            meanOfRttMean = 0
            for node in nodeList:
                if node in clientList:
                    meanOfRttMean = meanOfRttMean + rttMeanList[node]
            meanOfRttMean = meanOfRttMean / len(clientList)

            tempSampleList.append(meanOfRttMean)

            print("sample: ", meanOfRttMean)
            print(currentSampleNum, "번째 샘플링 종료")

        meanOfSamples = 0
        for sample in tempSampleList:
            meanOfSamples += sample
        meanOfSamples = meanOfSamples / CONST_SAMPLE_FOR_ONE_POSITION

        print("-----------------------------")
        print("------Simulation result------")
        print("-----------------------------")
        print("cacheServerXPos--------------")
        print(xPosList[cacheServerNodeNum])
        print("cacheServerYPos--------------")
        print(yPosList[cacheServerNodeNum])
        print("meanOfRttMean----------------")
        print(meanOfSamples)
        print("-----------------------------")

        xList.append(xPosList[cacheServerNodeNum])
        yList.append(yPosList[cacheServerNodeNum])
        finalRttDataList.append(meanOfSamples)
        print(currentSimulNum, "번째 시뮬레이션 끝")

# 네트워크의 best performance를 가질 것이라 예상되는 지점을 구하는 식의 계수를 구하는 과정
constantTerm = 0
firstXTerm = 0
firstYTerm = 0
secondXTerm = 0
secondYTerm = 0
for node in nodeList:
    if node in clientList:
        constantTerm = constantTerm + ((1 - groupProbList[node] / 100)**2) * (xPosList[node] ** 2) + ((1 - groupProbList[node] / 100)**2) * (yPosList[node] ** 2)
        firstXTerm = firstXTerm + ((-2) * ((1 - groupProbList[node] / 100)**2) * xPosList[node])
        firstYTerm = firstYTerm + ((-2) * ((1 - groupProbList[node] / 100)**2) * yPosList[node])
        secondXTerm = secondXTerm + ((1 - groupProbList[node] / 100)**2)
        secondYTerm = secondYTerm + ((1 - groupProbList[node] / 100)**2)
print("constantTerm---------------- ")
print(constantTerm)
print("firstXTerm-------------------")
print(firstXTerm)
print("firstYTerm-------------------")
print(firstYTerm)
print("secondXTerm------------------")
print(secondXTerm)
print("secondYTerm------------------")
print(secondYTerm)
print("-----------------------------")

#TODO: 만약 필요없으면 꼭 지우기(CONST_FACTOR의 영향을 없애는 부분)
listForPlot=[]
for data in finalRttDataList:
    listForPlot.append(data/CONST_FACTOR)
print(len(listForPlot))
print(len(xNp))
print(len(yNp))

#fig=plt.figure()
#ax=fig.add_subplot(111, projection='3d')
#dx=10
#dy=10
#ax.set_xlabel('xPos')
#ax.set_ylabel('yPos')
#ax.set_zlabel('meanOfRttMeans')
#ax.bar3d(xList, yList, 0.11*np.ones_like(finalRttDataList), dx, dy, listForPlot, shade=True)
#ax.scatter(xList, yList, np.array(finalRttDataList))
#plt.show()

#fig=plt.figure()
#ax=Axes3D(fig)
#dx=10
#dy=10
#ax.set_xlabel('xPos')
#ax.set_ylabel('yPos')
#ax.set_zlabel('meanOfRttMeans')
#ax.set_zlim3d([0.114, 0.123])
#ax.set_zticks(np.linspace(0.114, 0.123, 10))
#ax.bar3d(xList, yList, 0.114*np.ones_like(finalRttDataList), dx, dy, listForPlot)
#plt.show()

X, Y=np.meshgrid(xNp, yNp)
Z=np.array(listForPlot).reshape((len(yNp), len(xNp)))
CS=plt.contourf(X, Y, Z, alpha=0.5, cmap='seismic')
plt.colorbar(CS)
plt.show()

# 필요한 데이터들을 csv파일로 만들어 준다
rttMeanDataFrame = pd.DataFrame({'rttMean': rttMeanList})
rttMeanDataFrame.to_csv(f'./output/rttMean{fileNum}.csv', index=False, header=False)

varianceDataFrame = pd.DataFrame({'variety': varietyList})
varianceDataFrame.to_csv(f'./output/variance{fileNum}.csv', index=False, header=False)
#TODO:현재는 finalRttDataList대신 listForPlot을 쓰고있는 중이다
finalDataFrame = pd.DataFrame({'xPos': xList, 'yPos': yList, 'finalRttData': listForPlot})
finalDataFrame.to_csv(f'./output/final{fileNum}.csv', index=False, header=True)

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#plt.show()
