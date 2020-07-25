import psutil
import numpy as np
import random

number = 1000000

threshold = psutil.virtual_memory().percent
print("threshold = {}".format(threshold))
x = []

for i in range(number):
    x.append(np.zeros((int(random.random()*10), int(random.random()*10)), dtype=np.float32))
    if i % 100000 == 0:
        print(i)
        print(psutil.virtual_memory().percent)
    if(psutil.virtual_memory().percent > threshold+0.5 and i >0):
        print("condition met at iteration {}".format(i))
        break
        