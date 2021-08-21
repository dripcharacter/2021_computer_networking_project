import networkx as nx
import threading
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from random import randint
import sys
import numpy as np
import redis
import json

# node, edge, file, group에 관한 정보를 담은 csv 파일과 네트워크 performance를 평가하는 정보를 담을 final csv파일
topology = sys.argv[1]
edge = sys.argv[2]
file = sys.argv[3]
group = sys.argv[4]
# 대충 파일이름 formating해줌
fileNum = topology[12:13]  # 파일이름에 따라 달라질듯
# redis db 연결
rd = redis.StrictRedis(host='localhost', port=6379, db=0)
# 멀티쓰레딩용 lock
lock = threading.Lock()
# 이후에 있을 시뮬레이션에 필요한 정보들을 추출
nodeData = pd.read_csv(topology)  # node 관련 데이터 추출
edgeData = pd.read_csv(edge)  # edge 관련 데이터 추출
fileData = pd.read_csv(file)  # file 관련 데이터 추출
groupData = pd.read_csv(group)  # group 관련 데이터 추출

node = nodeData['node']
xPos = nodeData['xPos']
yPos = nodeData['yPos']
nodeType = nodeData['nodeType']
relatedCache = nodeData['relatedCache']
relatedServer = nodeData['relatedServer']
groupName = nodeData['groupName']

nodeList = node.values.tolist()
xPosList = xPos.values.tolist()
yPosList = yPos.values.tolist()
nodeTypeList = nodeType.values.tolist()
relatedCacheList = relatedCache.values.tolist()
relatedServerList = relatedServer.values.tolist()
groupNameList = groupName.values.tolist()

edgeA = edgeData['nodeA']
edgeB = edgeData['nodeB']
edgeNum = edgeData['edgeNum']
capacity = edgeData['capacity']

edgeAList = edgeA.values.tolist()
edgeBList = edgeB.values.tolist()
edgeNumList = edgeNum.values.tolist()
capacityList = capacity.values.tolist()

fileName = fileData['fileName']
fileSize = fileData['fileSize']
fileInterval = fileData['fileInterval']
fileLoad = fileData['fileLoad']

fileNameList = fileName.values.tolist()
fileSizeList = fileSize.values.tolist()
fileIntervalList = fileInterval.values.tolist()
fileLoadList = fileLoad.values.tolist()

groupNumber = len(groupData.columns)
groupFileList = [[]]
for i in range(1, groupNumber + 1):
    groupFileList.append(groupData["group" + str(i) + "FileList"].values.tolist())

currentSimulNum = 0  # 현재 simulation을 끝낸 횟수
finalRttDataList = []   # 모든 sample을 저장하는 곳

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

xNp = np.arange(0, 501, 100)
yNp = np.arange(0, 401, 100)
for xCoord in xNp:
    for yCoord in yNp:
        currentSimulNum += 1
        print(currentSimulNum, "번째 시뮬레이션 시작")
        # cacheServer의 위치를 랜덤으로 배정하는 부분(위치의 x, y position의 범위는 네트워크 토폴로지의 베이스가 된 사진의 크기와 관련있다.)
        cacheServerNodeNum = 0
        for node in nodeList:
            if nodeTypeList[node] == 1:
                xPosList[node] = xCoord
                yPosList[node] = yCoord
                cacheServerNodeNum = node
                G.nodes[node]['xPos'] = xPosList[nodeList.index(node)]
                G.nodes[node]['yPos'] = yPosList[nodeList.index(node)]
                break
        # 네트워크 시뮬레이션과 관련된 하이퍼 파라메터(총 시행할 통신의 횟수, 링크의 weight와 관련된 factor, caching server의 size, req packet의 size)
        CONST_PACKETSERIES_LIMIT = 1000
        CONST_FACTOR = 0.00001
        CONST_CACHE_SIZE = 10
        CONST_SAMPLE_FOR_ONE_POSITION = 1
        CONST_REQUEST_PACKET_SIZE = 1
        tempSampleList = []
        # 한 cache server position에 대한 시뮬레이션 결과들을 잠시 저장했다가 CONST_SAMPLE_FOR_ONE_POSITION 만큼 샘플링 후 평균을 낸다
        currentSampleNum = 0  # 현재까지 모인 한 cache server position에 대한 샘플 개수

        # fileInterval에 CONST_FACTOR반영
        for interval in range(len(fileIntervalList)):
            fileIntervalList[interval] = fileIntervalList[interval] * CONST_FACTOR

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
                    nodeCacheKey = str(node) + "node cache"
                    nodeCacheList = cacheList[node]
                    jsonnodeCacheList = json.dumps(nodeCacheList, indent=4)
                    rd.set(nodeCacheKey, jsonnodeCacheList)
            # 각 노드들의 매 통신마다의 rtt를 기록할 리스트 초기화
            rttList = []
            for i in clientList:
                nodeRtt = []
                jsonnodeRtt = json.dumps(nodeRtt, indent=4)
                rttList.append(nodeRtt)
                nodeRttKey = str(i) + "node rtt"
                rd.set(nodeRttKey, jsonnodeRtt)

            # 각 node들의 총 통신 횟수가 몇번인지 저장하는 리스트들 초기화
            trialNumList = []
            for node in nodeList:
                if node not in clientList:
                    trialNumList.append(1)
                else:
                    trialNumList.append(0)

            # 현재 cacheServer에서 originServer에 요청중인 파일리스트
            currentReqFileList = []
            currentReqFileKey = "currentReqFileKey"
            jsoncurrentReqFileList = json.dumps(currentReqFileList, indent=4)
            rd.set(currentReqFileKey, jsoncurrentReqFileList)

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
                G.edges[edgeAList[edgeNum], edgeBList[edgeNum]]['capacity'] = capacityList[edgeNum]
                # edge들의 현재 load를 기록하기 위한 변수를 미리 redis에 초기화 시켜놓는다
                # 작은수 to 큰수 edge current load라는 key를 가진다
                if edgeAList[edgeNum] < edgeBList[edgeNum]:
                    edgeCurrentLoadKey = str(edgeAList[edgeNum]) + "to" + str(
                        edgeBList[edgeNum]) + "edge current load"
                else:
                    edgeCurrentLoadKey = str(edgeBList[edgeNum]) + "to" + str(
                        edgeAList[edgeNum]) + "edge current load"
                rd.set(edgeCurrentLoadKey, str(0))

            # packet이 optimalPath를 따라 이동하도록 하는 함수
            def packetMoving(optimalPath, dataBool, payload, fileLoadToEdge):
                global CONST_REQUEST_PACKET_SIZE
                global CONST_FACTOR

                print("packetMoving: starting function with optimalPath", optimalPath, "to move", payload,
                      "with fileLoadToEdge", fileLoadToEdge)
                for i in range(len(optimalPath) - 1):
                    print("start Moving from", optimalPath[i], "to", optimalPath[i + 1], "with payload", payload)
                    if dataBool:
                        leftFileSize = fileSizeList[payload]
                    else:
                        leftFileSize = CONST_REQUEST_PACKET_SIZE
                    if optimalPath[i] < optimalPath[i + 1]:
                        edgeCurrentLoadKey = str(optimalPath[i]) + "to" + str(
                            optimalPath[i + 1]) + "edge current load"
                    else:
                        edgeCurrentLoadKey = str(optimalPath[i + 1]) + "to" + str(
                            optimalPath[i]) + "edge current load"
                    with lock:
                        edgeCurrentLoad = float(rd.get(edgeCurrentLoadKey).decode())
                        edgeCurrentLoad = edgeCurrentLoad + fileLoadToEdge
                        rd.set(edgeCurrentLoadKey, str(edgeCurrentLoad))
                    print(edgeCurrentLoadKey, "has increased:", str(edgeCurrentLoad))
                    while leftFileSize > 0:
                        edgeCurrentLoad = float(rd.get(edgeCurrentLoadKey).decode())
                        print(edgeCurrentLoadKey, "now:", str(edgeCurrentLoad))
                        try:
                            leftFileSize = leftFileSize - (G.edges[optimalPath[i], optimalPath[i + 1]][
                                                               'capacity'] / edgeCurrentLoad) * fileLoadToEdge
                        except ZeroDivisionError as e:
                            print("------------------------------------------------------------------")
                            print("capacity:", G.edges[optimalPath[i], optimalPath[i + 1]]['capacity'])
                            print(edgeCurrentLoadKey, "edgeCurrentLoad:", edgeCurrentLoad)
                            print("fileLoadToEdge:", fileLoadToEdge)
                            print("------------------------------------------------------------------")

                        time.sleep(G.edges[optimalPath[i], optimalPath[i + 1]]['weight'] * CONST_FACTOR)
                        if leftFileSize > 0:
                            print("payload", payload, "needs to move", leftFileSize, "to complete moving from",
                                  optimalPath[i], "to", optimalPath[i + 1])
                        else:
                            print("payload", payload, "completed moving from",
                                  optimalPath[i], "to", optimalPath[i + 1])
                    with lock:
                        edgeCurrentLoad = float(rd.get(edgeCurrentLoadKey).decode())
                        edgeCurrentLoad = edgeCurrentLoad - fileLoadToEdge
                        rd.set(edgeCurrentLoadKey, str(edgeCurrentLoad))
                    print(edgeCurrentLoadKey, "has decreased:", str(edgeCurrentLoad))
                    print("ending Moving from", optimalPath[i], "to", optimalPath[i + 1], "with payload", payload)
                print("packetMoving: ending function with optimalPath", optimalPath, "to move", payload,
                      "with fileLoadToEdge", fileLoadToEdge)

            # cache server의 값을 update할 필요가 있을 경우 실행하는 함수
            def updateCache(G, dst, realdst, payload, fileLoadToEdge):
                global CONST_FACTOR
                print("updateCache: starting function from", realdst, "to", dst, ",", payload,
                      "will be cached in cacheServer")
                # dijkstra 알고리즘을 이용하여 cache server에서 origin server까지의 최적 path 찾기
                optimalPath = nx.dijkstra_path(G, dst, realdst, weight='weight')

                # dst에서 realdst로 이동
                packetMoving(optimalPath, False, payload, fileLoadToEdge)
                # realdst에서 dst로 이동
                packetMoving(optimalPath, True, payload, fileLoadToEdge)

                # cacheList를 업데이트하기 위해 redis에서 가져온다
                with lock:
                    nodeCacheKey = str(dst) + "node cache"
                    cacheList = rd.get(nodeCacheKey).decode()
                    cacheList = json.loads(cacheList)
                    del cacheList[0]
                    cacheList.append(payload)
                    jsoncacheList = json.dumps(cacheList, indent=4)
                    rd.set(nodeCacheKey, jsoncacheList)
                    currentReqFileKey = "currentReqFileKey"
                    jsoncurrentReqFileList = rd.get(currentReqFileKey).decode()
                    currentReqFileList = json.loads(jsoncurrentReqFileList)
                    try:
                        currentReqFileList.remove(payload)
                    except ValueError as e:
                        print(e)
                    jsoncurrentReqFileList = json.dumps(currentReqFileList, indent=4)
                    rd.set(currentReqFileKey, jsoncurrentReqFileList)

                print("updateCache: ending function from", realdst, "to", dst, ",", payload,
                      "will be cached in cacheServer")

            # 특정 통신의 절차를 진행하는 함수
            def packet(G, src, dst, realdst, payload, fileLoadToEdge, groupFileList):
                # packet 시작 확인용
                print("packet: starting function to get", payload, "from cache server", dst, "to", src,
                      "and origin server", realdst)
                # 시간 측정 시작
                start = time.time()
                # dijkstra 알고리즘을 이용하여 cache server까지의 최적 path 찾기
                optimalPath = nx.dijkstra_path(G, src, dst, weight='weight')

                # src에서 dst로의 이동
                packetMoving(optimalPath, False, payload, fileLoadToEdge)

                # 데이터가 cache에 있는지 알아보기 위해 redis에서 cacheList가져온다
                nodeCacheKey = str(dst) + "node cache"
                nodeCacheList = rd.get(nodeCacheKey).decode()
                nodeCacheList = json.loads(nodeCacheList)
                # dst node(cache server)에 찾는 데이터가 있는지 확인한다
                dataexistencebool = False
                if payload in nodeCacheList:
                    dataexistencebool = True
                else:
                    dataexistencebool = False

                # dst에서 src까지 packet 이동
                optimalPath.reverse()
                packetMoving(optimalPath, dataexistencebool, payload, fileLoadToEdge)

                if not dataexistencebool:
                    # cache server 업데이트
                    currentReqFileKey = "currentReqFileKey"
                    with lock:
                        jsoncurrentReqFileList = rd.get(currentReqFileKey).decode()
                        currentReqFileList = json.loads(jsoncurrentReqFileList)
                        if len(currentReqFileList) < CONST_CACHE_SIZE and payload not in currentReqFileList:
                            currentReqFileList.append(payload)
                            jsoncurrentReqFileList = json.dumps(currentReqFileList, indent=4)
                            rd.set(currentReqFileKey, jsoncurrentReqFileList)
                            thread = threading.Thread(target=updateCache,
                                                      args=(G, dst, realdst, payload, fileLoadToEdge))
                            thread.start()

                    # dijkstra 알고리즘을 이용하여 origin server까지의 최적 path 찾기
                    optimalPath = nx.dijkstra_path(G, src, realdst, weight='weight')

                    # src에서 realdst까지 이동
                    packetMoving(optimalPath, False, payload, fileLoadToEdge)

                    # realdst에서 src까지 이동
                    optimalPath.reverse()
                    packetMoving(optimalPath, True, payload, fileLoadToEdge)

                end = time.time()

                # 파일 전송이 끝난 것을 반영하기
                with lock:
                    fileSendingBoolKey = str(src) + "fileSendingBoolKey"
                    jsonfileSendingBoolList = rd.get(fileSendingBoolKey).decode()
                    fileSendingBoolList = json.loads(jsonfileSendingBoolList)
                    fileSendingBoolList[groupFileList.index(payload)] = False
                    jsonfileSendingBoolList = json.dumps(fileSendingBoolList, indent=4)
                    rd.set(fileSendingBoolKey, jsonfileSendingBoolList)

                # 파일 전송 끝난 시간 기록하기
                fileLastReqTimeKey = str(src) + "node file number" + str(payload) + "last req time"
                rd.set(fileLastReqTimeKey, str(end))

                # redis에서 rtt를 기록할 nodeRttList 가져오기
                with lock:
                    nodeRttKey = str(src) + "node rtt"
                    nodeRttList = rd.get(nodeRttKey).decode()
                    nodeRttList = json.loads(nodeRttList)
                    nodeRttList.append(end - start)  # 측정한 통신의 rtt를 기록

                    # redis로 nodeRttList 보내기
                    jsonnodeRttList = json.dumps(nodeRttList, indent=4)
                    rd.set(nodeRttKey, jsonnodeRttList)

                # packet 종료 확인용
                print("packet: ending function to get", payload, "from cache server", dst, "to", src,
                      "and origin server", realdst)


            def main():  # asyncio는 한번에 하나의 run만 할 수 있기에 한번에 asyncio 리스트 객체에서 run을 시킨다
                threads = []
                for node in clientList:
                    thread = threading.Thread(target=nodeThread,
                                              args=(G, node, relatedCacheList[node], relatedServerList[node],
                                                    groupFileList[groupNameList[node]]))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                currentThreadList = threading.enumerate()
                currentThreadList.remove(threading.current_thread())
                for thread in currentThreadList:
                    if thread.is_alive():
                        thread.join()


            def nodeThread(G, src, dst, realdst, groupFileList):
                global endedPacketSeries
                global CONST_PACKETSERIES_LIMIT
                global CONST_FACTOR

                # multiThreading에서 dataframe을 넘겨주면 오류가 생길 수 있기에 trialNumList를 대체할 변수 생성
                nodeTrialNum = 0

                # multiThreading에서 dataframe을 넘겨주면 오류가 생길 수 있기에 noderttList를 여기서 생성
                nodeRttList = []
                # redis에 nodeRttList보내기(여러 종류의 파일 packet에서 공유하기 때문)
                jsonnodeRttList = json.dumps(nodeRttList, indent=4)
                nodeRttKey = str(src) + "node rtt"
                rd.set(nodeRttKey, jsonnodeRttList)

                # thread들의 join 타이밍을 나중으로 하기 위해 저장하는 리스트
                threads = []

                # nodeThread multithreading 시작 확인용
                print("nodeThread: multithreading starting from cache server", dst, "to", src, "and origin server",
                      realdst)

                # 마지막 파일 전송 시간으로부터 일정 시간이 지났으며 아직 CONST_PACKETSERIES_LIMIT이 아닐 경우 파일 request
                for groupFile in groupFileList:  # redis에 node의 각 file의 lastReqTime을 보낸다
                    fileLastReqTimeKey = str(src) + "node file number" + str(groupFile) + "last req time"
                    with lock:
                        rd.set(fileLastReqTimeKey, str(time.time() + randint(1, 100) * CONST_FACTOR))

                fileSendingBoolList = []
                for _ in groupFileList:
                    fileSendingBoolList.append(False)
                jsonfileSendingBoolList = json.dumps(fileSendingBoolList, indent=4)
                fileSendingBoolKey = str(src) + "fileSendingBoolKey"
                with lock:
                    rd.set(fileSendingBoolKey, jsonfileSendingBoolList)

                while True:
                    for groupFile in groupFileList:
                        fileLastReqTimeKey = str(src) + "node file number" + str(groupFile) + "last req time"
                        currentLastReqTime = float(rd.get(fileLastReqTimeKey).decode())
                        jsonfileSendingBoolList = rd.get(fileSendingBoolKey).decode()
                        fileSendingBoolList = json.loads(jsonfileSendingBoolList)
                        if (time.time() - currentLastReqTime >= fileIntervalList[
                            groupFile] or time.time() >= currentLastReqTime) and not fileSendingBoolList[
                            groupFileList.index(groupFile)]:  # file.csv에는 fileName이 index 번호로 되어 있다
                            # 현재 src 노드에서 groupFile은 전송스케줄에 따라 전송을 시작했다는 것을 반영한다
                            fileSendingBoolList[groupFileList.index(groupFile)] = True
                            jsonfileSendingBoolList = json.dumps(fileSendingBoolList, indent=4)
                            fileSendingBoolKey = str(src) + "fileSendingBoolKey"
                            rd.set(fileSendingBoolKey, jsonfileSendingBoolList)
                            # groupFile의 전송을 관리할 함수 호출 or CONST_PACKETSERIES_LIMIT을 넘었으니 nodeThread를 종료
                            if endedPacketSeries < CONST_PACKETSERIES_LIMIT:
                                endedPacketSeries += 1  # 현재까지 진행한 통신 횟수 기록
                                nodeTrialNum += 1
                                print(endedPacketSeries, "번째 packetSeries 시작")
                                thread = threading.Thread(target=packet, args=(
                                    G, src, dst, realdst, groupFile, fileLoadList[groupFile], groupFileList))
                                thread.start()
                                threads.append(thread)
                            else:
                                break
                    if not endedPacketSeries < CONST_PACKETSERIES_LIMIT:
                        break
                    time.sleep(1 * CONST_FACTOR)

                # 앞의 while문에서 만들어진 thread들 중 아직 끝나지 않은 thread들을 join()하여 결과가 나올때까지 기다린다.
                for thread in threads:
                    if thread.is_alive():
                        thread.join()

                # redis에 저장되있던 fileLastReqTime 삭제하기
                for groupFile in groupFileList:
                    fileLastReqTimeKey = str(src) + "node file number" + str(groupFile) + "last req time"
                    rd.delete(fileLastReqTimeKey)

                # nodeTrialNum, rttList를 redis로 보냄
                with lock:
                    trialNumKey = str(src) + "node trialNum"
                    rd.set(trialNumKey, str(nodeTrialNum))

                # redis에 있는 fileSendingBoolList를 삭제한다
                with lock:
                    fileSendingBoolKey = str(src) + "fileSendingBoolKey"
                    rd.delete(fileSendingBoolKey)

                # nodeThread multithreading 종료 확인용
                print("nodeThread: multithreading ending from cache server", dst, "to", src, "and origin server",
                      realdst)


            main()
            # redis에 있는 cacheList들을 삭제한다
            for node in nodeList:
                if nodeTypeList[node] == 1:
                    nodeCacheKey = str(node) + "node cache"
                    rd.delete(nodeCacheKey)
            # redis에서 각 client node들의 rtt를 담은 list를 가져온다
            with lock:
                for node in clientList:
                    nodeRttKey = str(node) + "node rtt"
                    print("After main's key:", nodeRttKey)
                    nodeRttList = rd.get(nodeRttKey).decode()
                    nodeRttList = json.loads(nodeRttList)
                    print(node, "node's nodeRttList:", nodeRttList)
                    rttList[node] = nodeRttList
                    print("rttList[", node, "]'s nodeRttList:", nodeRttList)
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
                    print("why it is -1:", nodeRttList)
                    rttMeanList.append(-1)
            # redis에 있던 edgeCurrentLoad들을 삭제한다
            for edgeNum in edgeNumList:
                if edgeAList[edgeNum] < edgeBList[edgeNum]:
                    edgeCurrentLoadKey = str(edgeAList[edgeNum]) + "to" + str(
                        edgeBList[edgeNum]) + "edge current load"
                else:
                    edgeCurrentLoadKey = str(edgeBList[edgeNum]) + "to" + str(
                        edgeAList[edgeNum]) + "edge current load"
                rd.delete(edgeCurrentLoadKey)

            # redis에 있던 currentReqFileList를 삭제한다
            currentReqFileKey = "currentReqFileKey"
            rd.delete(currentReqFileKey)

            # redis에 있던 값을 불러와서 List에 저장한다
            for node in clientList:
                trialNumKey = str(node) + "node trialNum"
                trialNumList[node] = int(rd.get(trialNumKey).decode())
                rd.delete(trialNumKey)

            print(cacheList)
            print(rttMeanList)
            # 네트워크의 performance인 각 노드들의 평균 rtt들의 평균을 구하여 cache server 위치와 함께 미리 만든 리스트에 넣는다
            meanOfRttMean = 0
            for node in nodeList:
                if node in clientList:
                    meanOfRttMean = meanOfRttMean + rttMeanList[node]
            meanOfRttMean = meanOfRttMean / len(clientList)

            tempSampleList.append(meanOfRttMean)

            print("sample:", meanOfRttMean)
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

        finalRttDataList.append(meanOfSamples)
        print(currentSimulNum, "번째 시뮬레이션 끝")

# TODO: 만약 필요없으면 꼭 지우기(CONST_FACTOR의 영향을 없애는 부분)
listForPlot = []
for data in finalRttDataList:
    listForPlot.append(data / CONST_FACTOR)
print(len(listForPlot))
print(len(xNp))
print(len(yNp))

X, Y = np.meshgrid(xNp, yNp)
Z = np.array(listForPlot).reshape((len(yNp), len(xNp)))
CS = plt.contourf(X, Y, Z, alpha=0.5, cmap='seismic')
plt.colorbar(CS)
plt.show()

# 필요한 데이터들을 csv파일로 만들어 준다
rttMeanDataFrame = pd.DataFrame({'rttMean': rttMeanList})
rttMeanDataFrame.to_csv(f'./output/rttMean{fileNum}.csv', index=False, header=False)

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
