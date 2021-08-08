import redis
import json

rd=redis.StrictRedis(host='localhost', port=6379, db=0)

rd.set("Hello", "World")
print(rd.get("Hello"))
print(rd.get("Hello").decode())
a=rd.get("Hello").decode()
print(a)
rd.set("Hello", "Sex")
print(rd.get("Hello"))
print(rd.get("Hello").decode())
a=rd.get("Hello").decode()
print(a)
rd.delete("Hello")
testList=[1, 2, 3, 4, 5]
jsontestList=json.dumps(testList, indent=4)
print(jsontestList)
rd.set("testList", jsontestList)
testList2=rd.get("testList").decode()
testList2=json.loads(testList2)
print(testList2)
print(type(testList2))
rd.delete("testList")
for num in testList2:
    print(type(num))
    times=str(num)+" times sex"
    print(times)
    print(type(times))
rttList = []
for i in testList2:
    eachRtt = []
    rttList.append(eachRtt)
for i in testList2:
    rttList[i - 1] = testList2
print(rttList)
oneKey="one"
rd.set(oneKey, 1)
print(rd.get(oneKey))
print(type(rd.get(oneKey)))
print(rd.get(oneKey).decode())
print(type(rd.get(oneKey).decode()))
print(int(rd.get(oneKey).decode()))
print(type(int(rd.get(oneKey).decode())))