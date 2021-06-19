import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import asyncio
from random import randint

nodeData = pd.read_csv('topology_node.csv')
edgeData = pd.read_csv('topology_edge.csv')
node = nodeData['node']
xPos = nodeData['xPos']
yPos = nodeData['yPos']
nodeType=nodeData['nodeType']
relatedCache=nodeData['relatedCache']
relatedServer=nodeData['relatedServer']
groupProperty=nodeData['groupProperty']
groupProb=nodeData['groupProb']
nodeList = node.values.tolist()
xPosList = xPos.values.tolist()
yPosList = yPos.values.tolist()
nodeTypeList=nodeType.values.tolist()
relatedCacheList=relatedCache.values.tolist()
relatedServerList=relatedServer.values.tolist()
groupPropertyList=groupProperty.values.tolist()
groupProbList=groupProb.values.tolist()
edgeA = edgeData['nodeA']
edgeB = edgeData['nodeB']
edgeNum = edgeData['edgeNum']
edgeAList = edgeA.values.tolist()
edgeBList = edgeB.values.tolist()
edgeNumList = edgeNum.values.tolist()

endedPacketSeries=0
CONST_PACKETSERIES_LIMIT=100
CONST_FACTOR=0.01
CONST_CACHE_SIZE=10

G = nx.Graph()
G.add_nodes_from(nodeList)

for node in nodeList:
    G.nodes[node]['xPos'] = xPosList[nodeList.index(node)]
    G.nodes[node]['yPos'] = yPosList[nodeList.index(node)]

cacheList=[]
for node in nodeList:
    eachCache=[]
    for i in range(0, CONST_CACHE_SIZE):
        if nodeTypeList[node-1]!=0:
            eachCache.append(randint(1, 100))
        else:
            eachCache.append(0)
    cacheList.append(eachCache)

rttList=[]
for i in nodeList:
    eachRtt=[]
    rttList.append(eachRtt)

clientList=[]
for i in nodeList:
    if nodeTypeList[i-1]==0:
        clientList.append(i)

varianceList=[]
trialNumList=[]
for node in nodeList:
    if node not in clientList:
        varianceList.append(-1)
        trialNumList.append(1)
    else:
        varianceList.append(0)
        trialNumList.append(0)

for edgeNum in edgeNumList:
    nodeA = G.nodes[edgeAList[edgeNum]]
    nodeB = G.nodes[edgeBList[edgeNum]]
    xPosEdgeA = nodeA['xPos']
    yPosEdgeA = nodeA['yPos']
    xPosEdgeB = nodeB['xPos']
    yPosEdgeB = nodeB['yPos']
    edgeWeight = math.sqrt(math.pow((xPosEdgeA-xPosEdgeB), 2) + math.pow((yPosEdgeA-yPosEdgeB), 2))
    G.add_edge(edgeAList[edgeNum], edgeBList[edgeNum], weight=edgeWeight)


def updateCache(G, dst, realdst, payload, cachelist):
        global CONST_FACTOR
        linkWeight = G.edges[dst, realdst]['weight']
        sleepTime = linkWeight*CONST_FACTOR
        time.sleep(sleepTime)
        time.sleep(sleepTime)
        del cachelist[dst-1][0]
        cachelist[dst-1].append(payload)


async def packet(G, src, dst, realdst, payload, cachelist, rttList):
    global endedPacketSeries
    global CONST_PACKETSERIES_LIMIT
    global CONST_FACTOR
    endedPacketSeries += 1
    if src in clientList:
        trialNumList[src-1]+=1
    print(endedPacketSeries)

    linkWeight = G.edges[src, dst]['weight']

    sleepTime=linkWeight*CONST_FACTOR
    start = time.time()
    await asyncio.sleep(sleepTime)

    dataexistencebool=False
    if payload in cachelist[dst-1]:
        dataexistencebool=True
    else:
        dataexistencebool=False

    await asyncio.sleep(sleepTime)

    linkWeight = G.edges[src, realdst]['weight']
    sleepTime=linkWeight*CONST_FACTOR

    updateStart=0
    updateEnd=0
    if dataexistencebool:
        sleepTime=0
    else:
        updateStart=time.time()
        updateCache(G, dst, realdst, payload, cachelist)
        updateEnd = time.time()

    await asyncio.sleep(sleepTime)
    await asyncio.sleep(sleepTime)

    end = time.time()
    rttList[src-1].append(end - start - (updateEnd-updateStart))

async def main():
    randIntList=[]
    for node in nodeList:
        payload=randint(1, 100)
        if payload<=groupProbList[node-1]:
            payload=groupPropertyList[node-1]
            varianceList[node-1]+=1
        randIntList.append(payload)
    futures=[asyncio.ensure_future(packet(G, node, relatedCacheList[node-1], relatedServerList[node-1], randIntList[node-1], cacheList, rttList)) for node in clientList]

    result=await asyncio.gather(*futures)

while endedPacketSeries<CONST_PACKETSERIES_LIMIT:
    asyncio.run(main())

rttMeanList=[]
for nodeRttList in rttList:
    if len(nodeRttList)!=0:
        rttSum=0
        for rtt in nodeRttList:
            rttSum=rttSum+rtt
        rttSum=rttSum/len(nodeRttList)
        rttMeanList.append(rttSum)
    else:
        rttMeanList.append(-1)

for node in nodeList:
    if node in clientList:
        if trialNumList[node-1]!=0:
            varianceList[node-1]=varianceList[node-1]/trialNumList[node-1]
        else:
            varianceList[node - 1] = varianceList[node - 1] / 1

print(cacheList)
print(rttMeanList)
print(varianceList)

rttMeanDataFrame = pd.DataFrame({'rttMean': rttMeanList})
rttMeanDataFrame.to_csv('./output/rttMean_output.csv', index=False, header=False)

varianceDataFrame = pd.DataFrame({'variance': varianceList})
varianceDataFrame.to_csv('./output/variance_output.csv', index=False, header=False)

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
