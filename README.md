# 2021_computer_networking_project

main.py 실행 command line: python main.py nodecsvfile edgecsvfile datacsvfile

ex) python main.py ./input/node3.csv ./input/edge3.csv ./output/final3.csv

input file 설명
1. node관련 csv파일 header 설명
  node:node name
  xPos: x Position of node
  yPos: y Position of node
  nodeType: type of node(0: client node, 1: cache node, 2: server node)
  relatedCache: node 와 연결된 cache node의 name(없으면 0)
  relatedServer: node 와 연결된 server node의 name(없으면 0)
  groupProperty: node가 속한 group의 이름
  groupProb: node가 같은 request를 보낼 확률 %(같은 group에 있으면 같은 groupProb를 갖는다)

2. edge관련 csv파일 header 설명
  edgeNum: edge name
  nodeA: edge에 연결된 node A
  nodeB: edge에 연결된 node B

output file 설명
1. final csv파일 관련 설명
  xPos: x Position of cache node
  yPos: y Position of cache node
  finalRttData: 위의 Position을 바탕으로 돌린 시뮬레이션에서 나온 rtt결과값
  +)각 final파일의 첫번째 data는 anticipated result를 따른다고 가정한 optimal position을 바탕으로 산출된 시뮬레이션 결과
2. final png파일 관련 설명
  시뮬레이션에서 모은 final csv파일의 값들을 3차원 막대 그래프로 나타낸 결과

대본 설명: 영상의 대본
