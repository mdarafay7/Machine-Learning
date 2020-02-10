#Abdul Rafay Mohammed Assignment 1
#1001331625

def rec_factorial(n):
    if(n>1):
        return n*rec_factorial(n-1)
    else:
        return n

print(rec_factorial(10))
