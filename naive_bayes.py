#Abdul Rafay Mohammed UTA ID: 1001331625
import sys
import numpy
import math

def mean_func(arr):
    n=len(arr)
    sum=0
    for x in arr:
        sum=sum+x
    if n==0:
        n=0.01
    mean=sum/n
    return float(mean)


def gaussian(x, u, sigma):
    if sigma<0.01:
        sigma=0.01
    exponent = math.exp(-(math.pow(x-u,2)/(2*math.pow(sigma,2))))
    return (1 / (math.sqrt(2*math.pi) * sigma)) * exponent

def prob(arr,i):
    p_c=0
    for occ in arr:
        if occ==i:
            p_c=p_c+1
    p_c=p_c/len(arr)
    return p_c


def standard_deviation(arr):
    n_minus=len(arr)-1
    mean=mean_func(arr)
    sums_squared=0
    for x in arr:
        diff=x-mean
        diff_squared=diff**2
        sums_squared=sums_squared+diff_squared
    variance=sums_squared/n_minus
    standard_deviation=numpy.sqrt(variance)
    if standard_deviation<=0.01:
        standard_deviation=0.01
    return float(standard_deviation)

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
data = numpy.loadtxt(sys.argv[1],usecols=range(0,-1))

classes = numpy.loadtxt(sys.argv[1])

data_test= numpy.loadtxt(sys.argv[2],usecols=range(0,-1))

classes_test = numpy.loadtxt(sys.argv[2])


classes=classes[:,-1]
classes_test=classes_test[:,-1]
print(classes)
unique_classes = []
for x in classes:
    if x not in unique_classes:
        unique_classes.append(x)

unique_classes=numpy.sort(unique_classes)

x=0

storage_mean = [[0 for x in range(spaces)] for y in range(len(unique_classes)+1)]
storage=[[[] for x in range(spaces-1)] for y in range(len(unique_classes)+1)]
storage_sd=[[0 for x in range(spaces)] for y in range(len(unique_classes)+1)]




for x in unique_classes:
    i=0
    for row in data:
        if data[i,spaces-1]==x:
            for attr in range(0,spaces-1):
                storage[int(x)][attr].append(data[i,attr])
        i+=1
    for attr in range(0,spaces-1):
        mean="{:.2f}".format(mean_func(storage[int(x)][attr]))
        std="{:.2f}".format(standard_deviation(storage[int(x)][attr]))
        storage_mean[int(x)][attr+1]=mean
        storage_sd[int(x)][attr+1]=std


counter=0
for x in unique_classes:
    for attr in range(1,9):
        print("Class {:d},".format(int(x)),"attribute {:d},".format(attr),"mean = {:.2f},".format(float(storage_mean[int(x)][attr])),"std = {:.2f}".format(float(storage_sd[int(x)][attr])))



iteration=0
match_store=[]
for z in data_test[:,:-1]:
    iteration+=1
    counter=0
    pxc_store=[]
    pcx_store=[]
    p_x=0
    for i in unique_classes:
        p_x_c=1
        count=0
        p_c=prob(classes,float(i))
        for x in z:
            count+=1
            p_x_c*=gaussian(float(x),float(storage_mean[int(i)][count]),float(storage_sd[int(i)][count]))
        pxc_store.append(p_x_c)
    for value in pxc_store:
        counter+=1
        p_x+=value*prob(classes,counter)
    counter=0
    for value in pxc_store:
        counter+=1
        pcx_store.append((value*prob(classes,counter))/(p_x))
    counter=1
    for val in pcx_store:
        if max(pcx_store)==val:
            break
        counter+=1
    match=0
    if counter==classes_test[iteration-1]:
        match=1
    match_store.append(match)
    print("ID={:5d},".format(iteration),"predicted={:3d},".format(counter),"probability={:.4f},".format(max(pcx_store)),"true={:3d},".format(int(classes_test[iteration-1])),"accuracy={:4.2f},".format(match))

classifcation_accuracy=sum(match_store)/len(classes_test)*100

print("classification accuracy={:6.4f}%".format(classifcation_accuracy))
