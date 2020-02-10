import sys
import numpy

def mean(arr):
    n=len(arr)
    sum=0
    for x in arr:
        sum=sum+x
    mean=sum/n
    return mean

def variance(arr):
    n_minus=len(arr)-1
    mean=mean(arr)
    sums_squared=0
    for x in arr:
        diff=x-mean
        diff_squared=diff**2
        sums_squared=sums_squared+diff_squared
    variance=sums_squared/n_minus

def standard_deviation(arr,mean):
    n_minus=len(arr)-1
    mean=mean(arr)
    sums_squared=0
    for x in arr:
        diff=x-mean
        diff_squared=diff**2
        sums_squared=sums_squared+diff_squared
    variance=sums_squared/n_minus
    standard_deviation=sqrt(variance)


with open(sys.argv[1]) as f:
    first_line = f.readline()
print(first_line)
spaces=0
word_check=1
for c in first_line:
    if c==' ':
        while c==' ' and word_check==1:
            spaces+=1
            word_check=0
            break
    else:
        word_check=1
# print(spaces)
data = numpy.loadtxt(sys.argv[1],usecols=range(0,spaces ))

classes = numpy.loadtxt(sys.argv[1],usecols=range(spaces-1,spaces))
# print(classes)

unique_classes = []
for x in classes:
    if x not in unique_classes:
        unique_classes.append(x)
for x in unique_classes:
    print(x)

classes=numpy.sort(unique_classes)
storage=[]
for x in classes:
    for y in range(0,spaces-1):
        storage.append([])
    i=0
    for row in data:
        if data[i,spaces-1]==x:
            for attr in range(0,spaces-1):
                storage[int(x*attr)].append(data[i,attr])
        i+=1
    for attr in range(0,spaces-1):
        print("Class: ",x," attribute: ",attr+1," mean: ",numpy.mean(storage[int(x*attr)])," std: ",numpy.std(storage[int(x*attr)]))
