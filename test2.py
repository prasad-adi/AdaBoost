a = [1,2,1,2,1,1,1,1,1,2,3,4,5]
b = {}
import operator
import numpy as np
for i in a:
    if(i not in b):
        temp = [1]
        b[i] = temp
    elif(i in b):
        b[i].append(1)
a = list(b.values())
print(sum(a,[]))
c = {1:2,2:4,4:5,5:6,6:7}
print(np.array(sorted(c.items(), key=lambda x:-x[1])[:3]).T[0])
print(b)
