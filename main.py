import networkx as nx
import threading
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import asyncio
from random import randint
import sys
import numpy as np

# node, edge 정보를 담은 csv 파일과 네트워크 performance를 평가하는 정보를 담을 final csv파일
topology = sys.argv[1]
edge = sys.argv[2]
final = sys.argv[3]
# 네트워크 시뮬레이션을 몇 번할지
simulNum=sys.argv[4]

fileNum = topology[12:13]
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

# 그래프를 만들고 node 관련 csv 파일에서 가져온 node 정보로 node 추가
G = nx.Graph()
G.add_nodes_from(nodeList)
# 각 node들에게 x, y Position을 attribute로 부여한다
for node in nodeList:
    G.nodes[node]['xPos'] = xPosList[nodeList.index(node)]
    G.nodes[node]['yPos'] = yPosList[nodeList.index(node)]

# TODO 여기서부터는 시뮬레이션 1번에 대한 것
# cacheServer의 위치를 랜덤으로 배정하는 부분(위치의 x, y position의 범위는 네트워크 토폴로지의 베이스가 된 사진의 크기와 관련있다.)
cacheServerNodeNum = 0
for node in nodeList:
    if nodeTypeList[node] == 1:
        xPosList[node] = randint(0, 500)
        yPosList[node] = randint(0, 400)
        cacheServerNodeNum = node
# 네트워크 시뮬레이션과 관련된 하이퍼 파라메터(총 시행할 통신의 횟수, 링크의 weight와 관련된 factor, caching server의 size)
endedPacketSeries = 0
CONST_PACKETSERIES_LIMIT = 1000
CONST_FACTOR = 0.000001
CONST_CACHE_SIZE = 10

# 캐싱 서버의 역할을 할 리스트들의 초기화
cacheList = []
for node in nodeList:
    eachCache = []
    for i in range(0, CONST_CACHE_SIZE):
        if nodeTypeList[node] != 0:
            eachCache.append(randint(1, 100))
        else:
            eachCache.append(0)
    cacheList.append(eachCache)
# 각 노드들의 매 통신마다의 rtt를 기록할 리스트 초기화
rttList = []
for i in nodeList:
    eachRtt = []
    rttList.append(eachRtt)
# client node(server node, cache server node를 제외한 노드)를 담아놓는 리스트
clientList = []
for i in nodeList:
    if nodeTypeList[i] == 0:
        clientList.append(i)
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
def updateCache(G, dst, realdst, payload, cachelist):
    global CONST_FACTOR
    linkWeight = G.edges[dst, realdst]['weight']
    sleepTime = linkWeight * CONST_FACTOR
    time.sleep(sleepTime)
    time.sleep(sleepTime)
    del cachelist[dst][0]
    cachelist[dst].append(payload)


# 특정 통신의 절차를 진행하는 함수
async def packet(G, src, dst, realdst, payload, cachelist, rttList):
    global endedPacketSeries
    global CONST_PACKETSERIES_LIMIT
    global CONST_FACTOR
    endedPacketSeries += 1  # 현재까지 진행한 통신 횟수 기록
    if src in clientList:
        trialNumList[src] += 1
    print(endedPacketSeries)

    linkWeight = G.edges[src, dst]['weight']
    # sleepTime을 node간 거리*factor로 하여 node들 간의 거리가 너무 멀거나 짧거나 길 경우 시뮬레이션 시간을 줄이거나 늘려준다
    sleepTime = linkWeight * CONST_FACTOR
    start = time.time()
    await asyncio.sleep(sleepTime)  # src에서 dst까지 packet 이동
    # dst node(cache server)에 찾는 데이터가 있는지 확인한다
    dataexistencebool = False
    if payload in cachelist[dst]:
        dataexistencebool = True
    else:
        dataexistencebool = False

    await asyncio.sleep(sleepTime)  # dst에서 src까지 packet 이동

    linkWeight = G.edges[src, realdst]['weight']
    sleepTime = linkWeight * CONST_FACTOR
    # update cache를 비동기로 돌리기 힘들어서 그 시간동안 wait하되, node의 통신 시간에는 영향을 미치지 않게 하기 위한 측정
    updateStart = 0
    updateEnd = 0
    # data가 있으면 sleepTime을 0으로 하여 client에서 본 서버까지 데이터 요청하는 시간을 없앤다
    if dataexistencebool:
        sleepTime = 0
    else:  # updateStart와 updateEnd를 측정하여 updateCache하는동안의 시간을 측정한다
        updateStart = time.time()
        updateCache(G, dst, realdst, payload, cachelist)
        updateEnd = time.time()
    # src에서 본 서버에 데이터를 요청하는 시간 지연이지만 cache server에 데이터가 있다면 sleepTime이 0이기 때문에 없는 것과 같다
    await asyncio.sleep(sleepTime)
    await asyncio.sleep(sleepTime)

    end = time.time()
    rttList[src].append(end - start - (updateEnd - updateStart))  # 측정한 통신의 rtt를 기록


async def main():  # asyncio는 한번에 하나의 run만 할 수 있기에 한번에 asyncio 리스트 객체에서 run을 시킨다
    randIntList = []
    for node in nodeList:
        payload = randint(1, 100)
        if payload <= groupProbList[node]:
            payload = groupPropertyList[node]
            varietyList[node] += 1
        randIntList.append(payload)
    futures = [asyncio.ensure_future(
        packet(G, node, relatedCacheList[node], relatedServerList[node], randIntList[node], cacheList,
               rttList)) for node in clientList]

    result = await asyncio.gather(*futures)


# 비동기 통신을 하는 main함수를 미리 정한 통신 횟수가 넘을때까지 실행한다
while endedPacketSeries < CONST_PACKETSERIES_LIMIT:
    asyncio.run(main())
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
# 네트워크의 best performance를 가질 것이라 예상되는 지점을 구하는 식의 계수를 구하는 과정
constantTerm = 0
firstXTerm = 0
firstYTerm = 0
secondXTerm = 0
secondYTerm = 0
for node in nodeList:
    if node in clientList:
        constantTerm = constantTerm + groupProbList[node] / 100 * (xPosList[node] ** 2) + groupProbList[
            node] / 100 * (yPosList[node] ** 2)
        firstXTerm = firstXTerm + ((-2) * (groupProbList[node] / 100) * xPosList[node])
        firstYTerm = firstYTerm + ((-2) * (groupProbList[node] / 100) * yPosList[node])
        secondXTerm = secondXTerm + (groupProbList[node] / 100)
        secondYTerm = secondYTerm + (groupProbList[node] / 100)
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
# 네트워크의 performance인 각 노드들의 평균 rtt들의 평균을 구하여 cache server 위치와 함께 미리 만든 리스트에 넣는다
meanOfRttMean = 0
for node in nodeList:
    if node in clientList:
        meanOfRttMean = meanOfRttMean + rttMeanList[node]
meanOfRttMean = meanOfRttMean / (len(rttMeanList) - 2)
print("cacheServerXPos--------------")
print(xPosList[cacheServerNodeNum])
print("cacheServerYPos--------------")
print(yPosList[cacheServerNodeNum])
print("meanOfRttMean----------------")
print(meanOfRttMean)
print("-----------------------------")

xList.append(xPosList[cacheServerNodeNum])
yList.append(yPosList[cacheServerNodeNum])
finalRttDataList.append(meanOfRttMean)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
dx=30
dy=30
ax.set_xlabel('xPos')
ax.set_ylabel('yPos')
ax.set_zlabel('meanOfRttMeans')
ax.bar3d(xList, yList, np.zeros_like(finalRttDataList), dx, dy, finalRttDataList, shade=True)
plt.show()

# 필요한 데이터들을 csv파일로 만들어 준다
rttMeanDataFrame = pd.DataFrame({'rttMean': rttMeanList})
rttMeanDataFrame.to_csv(f'./output/rttMean{fileNum}.csv', index=False, header=False)

varianceDataFrame = pd.DataFrame({'variety': varietyList})
varianceDataFrame.to_csv(f'./output/variance{fileNum}.csv', index=False, header=False)

finalDataFrame = pd.DataFrame({'xPos': xList, 'yPos': yList, 'finalRttData': finalRttDataList})
finalDataFrame.to_csv(f'./output/final{fileNum}.csv', index=False, header=True)

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
