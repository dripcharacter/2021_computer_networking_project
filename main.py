import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math
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

G=nx.Graph()
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
    edgeWeight=math.sqrt(math.pow((xPosEdgeA-xPosEdgeB), 2)+math.pow((yPosEdgeA-yPosEdgeB), 2))
    G.add_edge(edgeAList[edgeNum], edgeBList[edgeNum], weight=edgeWeight)

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()