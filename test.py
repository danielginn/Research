import numpy as np


array = np.zeros(20)
copy = np.zeros(10)
array_width = 20
copy_width = 10

for i in range(20):
    array[i] = i

x = 15

for x in range(15,20):
    diff = array_width - x
    print("diff: " + str(diff))
    copy[0:diff] = array[x:array_width]
    copy[diff:copy_width] = array[0:copy_width-diff]

print(copy)