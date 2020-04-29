#Abdul Rafay Mohammed 1001331625
import numpy as np
import sys
import random
import math
from collections import deque


class tree_class:
  def __init__(self):
      self.left_child = None
      self.right_child = None
      self.level =None

def print_lvl(tree):
    print(tree.left_child.best_attribute)
    print(tree.right_child.best_attribute)


def print_tree(tree):
    print_lvl(tree.best_attribute)
    test_tree = tree
    if tree.left_child != None:
        print_lvl(test_tree)
    elif tree.right_child != None:
        print_lvl(test_tree)
    if tree.left_child != None:
        print_tree(tree.left_child)
    elif tree.right_child != None:
        print_tree(tree.right_child)



def printer(node):
    node.level = 1
    tree = deque([node])
    output = []
    current_level = node.level
    gain_store = []
    thres_store = []
    while len(tree)>0:
          ptr = tree.popleft()
          if(ptr.level > current_level):
              output.append("\n")
              current_level += 1
          val = str(ptr.best_attribute)
          if val == "- 1":
              val = "-1"
          output.append(val)
          gain_store.append(str(ptr.gain))
          thres_store.append(str(ptr.best_threshold))

          if ptr.left_child != None:
             ptr.left_child.level = current_level + 1
             tree.append(ptr.left_child)

          if ptr.right_child != None:
             ptr.right_child.level = current_level + 1
             tree.append(ptr.right_child)
    return output, gain_store, thres_store




def DISTRIBUTION(examples):
    size=np.size(examples,0)
    return_storage=[]
    for class_ in classes:
        class_examples=list(filter(
            lambda x: x  == class_,
            examples[:,-1]))
        return_storage.append(len(class_examples)/size)
    return return_storage

def ENTROPY(examples):
    size=np.size(examples, 0)
    H=0
    for class_ in classes:
        amt_class_examples=len(list(filter(
            lambda x: x  == class_,
            examples[:,-1])))
        if amt_class_examples==0:
            H -= 0
        else:
            H -= (amt_class_examples/size) * (np.log2(amt_class_examples/size))
    return H


def INFORMATION_GAIN(examples, A, threshold):
    head_entropy = ENTROPY(examples)
    greater_than, less_than = [], []
    pointer=0
    for example in examples:
        if example[A] >= threshold:
            greater_than.append(pointer)
        else:
            less_than.append(pointer)
        pointer+=1
    greater_than= np.delete(examples, greater_than, 0)
    less_than= np.delete(examples, less_than, 0)
    size = np.size(examples,0)
    sigma = (len(less_than)/size) * ENTROPY(less_than)
    sigma += (len(greater_than)/size) * ENTROPY(greater_than)
    I = head_entropy - sigma
    return I

def CHOOSE_ATTRIBUTE(examples, attributes):
    max_gain, best_attribute, best_threshold=-1,-1,-1
    if option_todo == "optimized":
        for A in attributes:
            attribute_values=examples[:,A]
            L=min(attribute_values)
            M=max(attribute_values)
            # print(examples)
            for K in range(1,52):
                # print(A,K)
                threshold = L + (K * (M-L)/51)
                gain = INFORMATION_GAIN(examples, A, threshold)
                # print(gain)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = A
                    best_threshold = threshold
    elif option_todo == "random" or "forest3" or "forest15":
            A = random.choice(attributes)
            attribute_values=examples[:,A]
            L=min(attribute_values)
            M=max(attribute_values)
            best_attribute = A
            for K in range(1,52):
                # print(A,K)
                threshold = L + (K * (M-L)/51)
                gain = INFORMATION_GAIN(examples, A, threshold)
                # print(gain)
                if gain > max_gain:
                    max_gain = gain
                    best_threshold = threshold

    return(best_attribute, best_threshold, max_gain)

def DTL(examples, attributes, default, pruning_thr):
    tree = tree_class()
    # print("x")
    if(np.size(examples,0)<pruning_thr):
        tree.distribution = default
        tree.gain = 0
        tree.best_threshold = -1
        tree.best_attribute = -1
    elif((examples[:,-1]==examples[0,-1]).all()):
        tree.best_threshold = -1
        tree.best_attribute = -1
        dist = np.zeros(len(classes))
        class_ = int(examples[0,-1])
        dist[class_] = 1
        tree.distribution = dist
        tree.gain = 0
    else:
        (tree.best_attribute,tree.best_threshold,tree.gain) = CHOOSE_ATTRIBUTE(examples, attributes)
        examples_right, examples_left = [], []
        pointer=0
        for example in examples:
            if example[tree.best_attribute] >= tree.best_threshold:
                examples_left.append(pointer)
            else:
                examples_right.append(pointer)
            pointer+=1
        examples_left= np.delete(examples, examples_left, 0)
        examples_right= np.delete(examples, examples_right, 0)
        tree.distribution = DISTRIBUTION(examples)
        tree.left_child = DTL(examples_left, attributes, tree.distribution, pruning_thr)
        tree.right_child = DTL(examples_right, attributes, tree.distribution, pruning_thr)
    # print(tree.best_attribute,tree.best_threshold)
    return tree



def DTL_TopLevel(examples, pruning_thr):
    attributes=list(range(0,np.size(examples[:,:-1],1)))
    default=DISTRIBUTION(examples)
    return DTL(examples, attributes, default, pruning_thr)


def decision_tree(training_file, test_file, option, pruning_thr):
    examples=training_file
    test_examples = test_file
    training_data=training_file[:,:-1]
    test_data=test_file[:,:-1]
    classes_training=training_file[:,-1]
    classes_test=test_file
    classes_test=classes_test[:,-1]
    unique_classes = []
    for x in classes_training:
        if x not in unique_classes:
            unique_classes.append(int(x))
    amt_classes=len(unique_classes)
    unique_classes = sorted(unique_classes)
    global classes
    classes = unique_classes
    global option_todo
    option_todo = option
    accurate = 0
    lowest = int(min(unique_classes))
    tree_cnt = 1
    if(option == "optimized"):
        tree = DTL_TopLevel(examples, pruning_thr)
        print_opt, gain_store, thres_store=(printer(tree))
        level = 1
        gain_cnt, thres_cnt = 0, 0
        node = 0
        for opt in print_opt:
            if opt=="\n":
                level+=1
                continue
            print("tree={:2d}".format(tree_cnt),"node={:3d}".format(int(node)),"feature={:2d}".format(int(opt)),"thr={:6.2f}".format(float(thres_store[thres_cnt])),"gain={:f}".format(float(gain_store[gain_cnt])))
            gain_cnt+=1
            thres_cnt+=1
            node = node + 1

        test_tree = tree
        count = 0
        for example in test_examples:
            count += 1
            tree = test_tree
            while tree.left_child!=None and tree.right_child!=None:
                attribute = tree.best_attribute
                if example[attribute] < tree.best_threshold:
                    tree = tree.left_child
                elif example[attribute] >= tree.best_threshold:
                    tree = tree.right_child
            accuracy = 0
            if example[-1] == np.argmax(tree.distribution)+lowest:
                accuracy = 1
                accurate += 1
            # if np.argmax(tree.distribution) == 0:
            #     print("ID = {:5d}".format(count) ,"Predicted={:3d}".format(int(np.argmax(tree.distribution))) ,"true={:3d}".format(int(example[-1])) ,"accuracy={:4.2f}".format(accuracy))
            # else:
            print("ID = {:5d}".format(count) ,"Predicted={:3d}".format(int(np.argmax(tree.distribution))+lowest) ,"true={:3d}".format(int(example[-1])) ,"accuracy={:4.2f}".format(accuracy))
            # print(tree.distribution)
        print("classification accuracy={:6.4f}".format(float((accurate)/(count))))
    elif(option == "randomized"):
        tree = DTL_TopLevel(examples, pruning_thr)

        print_opt, gain_store, thres_store=(printer(tree))
        level = 1
        gain_cnt, thres_cnt = 0, 0
        node = 0
        for opt in print_opt:
            if opt=="\n":
                level+=1
                continue
            print("tree={:2d}".format(tree_cnt),"node={:3d}".format(int(node)),"feature={:2d}".format(int(opt)),"thr={:6.2f}".format(float(thres_store[thres_cnt])),"gain={:f}".format(float(gain_store[gain_cnt])))
            gain_cnt+=1
            thres_cnt+=1
            node = node + 1




        test_tree = tree
        count = 0
        for example in test_examples:
            count += 1
            tree = test_tree
            while tree.left_child!=None and tree.right_child!=None:
                attribute = tree.best_attribute
                if example[attribute] < tree.best_threshold:
                    tree = tree.left_child
                elif example[attribute] >= tree.best_threshold:
                    tree = tree.right_child
            accuracy = 0
            if example[-1] == np.argmax(tree.distribution)+lowest:
                accuracy = 1
                accurate += 1
            print("ID = {:5d}".format(count) ,"Predicted={:3d}".format(int(np.argmax(tree.distribution))+lowest) ,"true={:3d}".format(int(example[-1])) ,"accuracy={:4.2f}".format(accuracy))
        print("classification accuracy={:6.4f}".format(float((accurate)/(count))))
    elif(option == "forest3" or option == "forest15"):
        if option == "forest3":
            amt = 3
        else:
            amt = 15
        for x in range(amt):
            tree_store = []
            tree = DTL_TopLevel(examples, pruning_thr)
            print_opt, gain_store, thres_store=(printer(tree))
            level = 1
            gain_cnt, thres_cnt = 0, 0
            node = 0
            for opt in print_opt:
                if opt=="\n":
                    level+=1
                    continue
                print("tree={:2d}".format(tree_cnt),"node={:3d}".format(int(node)),"feature={:2d}".format(int(opt)),"thr={:6.2f}".format(float(thres_store[thres_cnt])),"gain={:f}".format(float(gain_store[gain_cnt])))
                gain_cnt+=1
                thres_cnt+=1
                node = node + 1
            tree_cnt+=1
            tree_store.append(tree)

        count = 0
        for example in test_examples:
            count += 1
            dist_store = []
            for tree_rand in tree_store:
                test_tree = tree_rand
                tree = test_tree
                while tree.left_child!=None and tree.right_child!=None:
                    attribute = tree.best_attribute
                    if example[attribute] < tree.best_threshold:
                        tree = tree.left_child
                    elif example[attribute] >= tree.best_threshold:
                        tree = tree.right_child
                dist_store = dist_store+list(tree.distribution)
            accuracy = 0
            if example[-1] == np.argmax(dist_store)+lowest:
                accuracy = 1
                accurate += 1
            print("ID = {:5d}".format(count) ,"Predicted{:3d}".format(int(np.argmax(dist_store)+lowest)) ,"true={:3d}".format(int(example[-1])) ,"accuracy={:4.2f}".format(float(accuracy)))
        print("classification accuracy={:6.4f}".format(float((accurate)/(count))))






decision_tree(np.loadtxt(sys.argv[1]),np.loadtxt(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
