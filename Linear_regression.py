#Abdul Rafay Mohammed 1001331625

import numpy
import sys

def linear_regression(training_file,degree,lambda_value,test_file):

    data=training_file[:,:-1]
    data2=test_file[:,:-1]
    classes_training=training_file[:,-1]
    classes_test=test_file
    classes_test=classes_test[:,-1]
    phi_collection=[]
    phi_collection2=[]

    for row in data:
        small_phi=[1]
        for val in row:
            for deg in range(degree):
                intern=val**(deg+1)
                small_phi.append(intern)
        phi_collection.append(small_phi)

    phi_collection=numpy.array(phi_collection)

    weight=numpy.linalg.pinv((lambda_value*numpy.identity(deg))+phi_collection.T@phi_collection)@(phi_collection.T@classes_training)

    count=0
    for x in weight:
        print(count," : ",x)
        count+=1

    for row in data2:
        small_phi=[1]
        for val in row:
            for deg in range(degree):
                intern=val**(deg+1)
                small_phi.append(intern)
        phi_collection2.append(small_phi)

    count=1
    for x in phi_collection2:
        print("ID = {:d} ".format(count),", output = {:14.4f}".format(weight.T@x),", target value = {:10.4f}".format(classes_test[count-1]),", squared error = {:.4f}".format(((classes_test[count-1])-(weight.T@x))**2))
        count+=1

linear_regression(numpy.loadtxt(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),numpy.loadtxt(sys.argv[4]))
