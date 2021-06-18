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
nodeList = node.values.tolist()
xPosList = xPos.values.tolist()
yPosList = yPos.values.tolist()
nodeTypeList=nodeType.values.tolist()
relatedCacheList=relatedCache.values.tolist()
relatedServerList=relatedServer.values.tolist()
edgeA = edgeData['nodeA']
edgeB = edgeData['nodeB']
edgeNum = edgeData['edgeNum']
edgeAList = edgeA.values.tolist()
edgeBList = edgeB.values.tolist()
edgeNumList = edgeNum.values.tolist()

endedPacketSeries=0
CONST_PACKETSERIES_LIMIT=100
CONST_FACTOR=0.1
CONST_CACHE_SIZE=10

G = nx.Graph()
G.add_nodes_from(nodeList)

for node in nodeList:
    G.nodes[node]['xPos'] = xPosList[nodeList.index(node)]
    G.nodes[node]['yPos'] = yPosList[nodeList.index(node)]

cacheList=[]
for i in nodeList:
    eachCache=[]
    for i in range(0, len(nodeList)):
        if nodeTypeList[i-1]!=0:
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
    rttList.append(end - start - (updateEnd-updateStart))

async def main():
    randIntList=[]
    for node in nodeList:
        payload=randint(1, 100)
        randIntList.append(payload)
    futures=[asyncio.ensure_future(packet(G, node, relatedCacheList[node-1], relatedServerList[node-1], randIntList[node-1], cacheList, rttList)) for node in clientList]

    result=await asyncio.gather(*futures)

while endedPacketSeries<CONST_PACKETSERIES_LIMIT:
    asyncio.run(main())

print(cacheList)
print(rttList)

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
