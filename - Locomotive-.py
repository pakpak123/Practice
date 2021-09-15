class SinglyLinkedList:
    class Node:
        def __init__(self, data, next = None):
            self.data = data
            if next is None:
                self.next = None
            else:
                self.next = next

    def __init__(self):
        self.head = None
        self.size = 0

    def __str__(self):
        s = "Before : "
        p = self.head
        c = 0
        while p != None:
            if c == self.__len__()-1 :
                s += str(p.data) + " "
                p = p.next
            else:
                s += str(p.data) + " <- "
                p = p.next
            c += 1
        return s

    def STRafter(self):
        s = "After : "
        p = self.head
        c = 0
        while p != None:
            if c == self.__len__() - 1:
                s += str(p.data) + " "
                p = p.next
            else:
                s += str(p.data) + " <- "
                p = p.next
            c += 1
        return s

    def __len__(self):
        return self.size

    def isEmpty(self):
        return self.size == 0

    def indexOf(self, data):  # หา อินเด็กของข้อมูลว่าอยู่ที่ตำแหน่งใด
        p = self.head
        for i in range(len(self)):
            if p.data == data:
                return i
            p = p.next
        return -1

    def isIn(self, data):
        return self.indexOf(data) >= 0

    def nodeAt(self, i):
        p = self.head
        for j in range(0, i):
            p = p.next
        return p

    def append(self, data):
        if self.head == None:
            self.head = self.Node(data, None)
            self.size += 1
        else:
            self.insertAfter(len(self) - 1, data)  # len(self) = จำนวนสมาชิก - 1 คือ index

    def insertAfter(self, i, data):
        p = self.nodeAt(i)
        p.next = self.Node(data, p.next)
        self.size += 1

    def deleteAfter(self, i):
        p = self.nodeAt(i)
        p.next = p.next.next
        self.size -= 1

    def delete(self, i):
        if i == 0:
            self.head = self.head.next
            self.size -= 1
        else:
            self.deleteAfter(i)

    def removeData(self, data):
        if self.isIn(data):
            self.deleteAfter(self.indexOf(data) - 1)


print(" *** Locomotive ***")
inp = input("Enter Input : ").split()
L = SinglyLinkedList()
LB = []
for e in range(0,len(inp)) :
    L.append(inp[e])
    LB.append(inp[e])
print(L)

LA = []

for e in range(L.indexOf("0"), L.__len__()):
    LA.append(LB[e])
if len(LA) != len(L) :
    for e in range(L.indexOf("0")):
        LA.append(LB[e])

L2 = SinglyLinkedList()
for e in range(0,len(LA)) :
    L2.append(LA[e])
print(L2.STRafter())