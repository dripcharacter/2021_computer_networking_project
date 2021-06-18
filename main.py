import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import asyncio

nodeData = pd.read_csv('topology_node.csv')
edgeData = pd.read_csv('topology_edge.csv')
node = nodeData['node']
xPos = nodeData['xPos']
yPos = nodeData['yPos']
nodeList = node.values.tolist()
xPosList = xPos.values.tolist()
yPosList = yPos.values.tolist()
edgeA = edgeData['nodeA']
edgeB = edgeData['nodeB']
edgeNum = edgeData['edgeNum']
edgeAList = edgeA.values.tolist()
edgeBList = edgeB.values.tolist()
edgeNumList = edgeNum.values.tolist()

endedPacketSeries=0
CONST_PACKETSERIES_LIMIT=100
CONST_FACTOR=10

G = nx.Graph()
G.add_nodes_from(nodeList)

for node in nodeList:
    G.nodes[node]['xPos'] = xPosList[nodeList.index(node)]
    G.nodes[node]['yPos'] = yPosList[nodeList.index(node)]

for edgeNum in edgeNumList:
    nodeA = G.nodes[edgeAList[edgeNum]]
    nodeB = G.nodes[edgeBList[edgeNum]]
    xPosEdgeA = nodeA['xPos']
    yPosEdgeA = nodeA['yPos']
    xPosEdgeB = nodeB['xPos']
    yPosEdgeB = nodeB['yPos']
    edgeWeight = math.sqrt(math.pow((xPosEdgeA-xPosEdgeB), 2) + math.pow((yPosEdgeA-yPosEdgeB), 2))
    G.add_edge(edgeAList[edgeNum], edgeBList[edgeNum], weight=edgeWeight)


async def updateCache(G, dst, realdst, payload, cachelist):
        global CONST_FACTOR
        linkWeight = G.edges[dst, realdst]['weight']
        sleepTime = linkWeight*CONST_FACTOR
        await asyncio.sleep(sleepTime)
        await asyncio.sleep(sleepTime)
        del cachelist[dst-1][0]
        cachelist[dst-1].append(payload)


async def packet(G, src, dst, realdst, payload, cachelist):
    global endedPacketSeries
    global CONST_PACKETSERIES_LIMIT
    global CONST_FACTOR
    endedPacketSeries += 1

    linkWeight = G.edges[src, dst]['weight']

    sleepTime=linkWeight*CONST_FACTOR
    await asyncio.sleep(sleepTime)

    dataexistencebool=False
    if payload in cachelist[dst-1]:
        dataexistencebool=True
    else:
        dataexistencebool=False

    await asyncio.sleep(sleepTime)

    linkWeight = G.edges[src, realdst]['weight']
    sleepTime=linkWeight*CONST_FACTOR

    if dataexistencebool:
        sleepTime=0
    else:
        asyncio.run(updateCache(G, dst, realdst, payload, cachelist))

    await asyncio.sleep(sleepTime)
    await asyncio.sleep(sleepTime)

    if endedPacketSeries<CONST_PACKETSERIES_LIMIT:
        asyncio.run(packet(G, src, dst, realdst, payload, cachelist)) # 새로운 request니까 payload 바꿔서 보내야됨

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
