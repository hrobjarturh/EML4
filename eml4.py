#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:47:21 2021

@author: hrobjarturhoskuldsson
"""

### Import Pytorch and other relevant packages
import torch
import torch.nn as nn
### Import MNIST dataset 
from torchvision.datasets import MNIST
### Load Numpy and Matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

#************************************ PART 1 *************************************************

print('Starting ...')
### Get to know your data

# 1: Download MNIST data 
print('\nDownloading MNIST data ...')
train_set = MNIST('.',  download=False)
test_set = MNIST('.', train=False)
xTrain, yTrain = torch.load('MNIST/processed/training.pt')
xTest, yTest = torch.load('MNIST/processed/test.pt')

xTrainNp, yTrainNp = xTrain.numpy(), yTrain.numpy()
xTestNp, yTestNp = xTest.numpy(), yTest.numpy()

# 2: Check how many data points are provided in the training and test sets
print('\nChecking length of data points ...')
print(type(xTrain))
print('Length of training data :',len(xTrain),' Length of training labels :',len(yTrain))
print('Length of test data :',len(xTest),' Length of test labels :',len(yTest))

# 3: While we will be assuming this is unlabelled data, MNIST provides 10 class labels. 
# We can use these to evaluate the performance of our unsupervised models. 
# Make a histogram of the different classes and see if the data distribution is balanced or not.
def create_histogram():
    print('\nCreating histogram of class labels ...')
    yTrainNP = yTrain.numpy()
    counts, bins, bars = plt.hist(yTrainNP, bins = np.arange(11) - 0.5,ec='black')
    plt.xticks(range(10))
    ax = plt.gca()
    p = ax.patches
    # print(p[0].get_height())
    plt.title("histogram")
    plt.xlabel('MNIST 10 class labels')
    plt.ylabel('Frequency')
    plt.show()
    
    heights = [patch.get_height() for patch in p]
    avg = 0
    for i in range(len(heights)):
        avg += heights[i]
    print(avg/len(heights))
    print('Values of each label in order : ',heights)

# 4:  Visualize a few samples

def visualize_sample(index):
    print('\nVisualising sample :',index,' ...')
    sample_image = xTrain[index]
    print(sample_image.size())
    plt.imshow(sample_image,cmap='gray_r')

# 5:  Reshape 28x28 to 784
def reshape_tensor(t):
    print('\nReshaping tensor ...')
    t_reshaped = torch.flatten(t, 1 , 2)
    print('Reshaped size: ', t_reshaped.shape)
    return t_reshaped

print('Original size: ', xTrain.shape)
reshape_tensor(xTrain)
#*************************************** PART 2 **********************************************

### Principal component analysis (PCA) on MNIST

# 1: Create a smaller dataset comprising a subset of data corresponding to classes [0, 1, 2, 3, 4]

def create_subset():
    print('\nCreating a data subset ...')
    subclasses = [0,1,2,3,4]
    indices = []
    indices_test = []
    for i in range(len(yTrain)):
        if yTrain[i] in subclasses:
            indices.append(i)
            
    for i in range(len(yTest)):
        if yTest[i] in subclasses:
            indices_test.append(i)
            
    xTrain_subset = xTrain[indices]
    yTrain_subset = yTrain[indices]
    xTest_subset = xTest[indices_test]
    yTest_subset = yTest[indices_test]
    return xTrain_subset, yTrain_subset, xTest_subset, yTest_subset

# 2 : You are allowed to use the PCA package in Scikit-Learn

# 3 - 5 : Perform PCA on the training data with D=200 components

xTrain_subset, yTrain_subset, xTest_subset, yTest_subset = create_subset()
xTrain_subset = reshape_tensor(xTrain_subset)
xTest_subset = reshape_tensor(xTest_subset)

def perform_pca(data, D):
    print('\nPerforming PCA with D = ',D,' ...')
    if type(D) is not list:
        pca = PCA(n_components=D)
        pca.fit(data)
        explained_variance = pca.explained_variance_ratio_
        print('Total variance explained : ',sum(explained_variance),' where D = ',D)
        plt.plot(explained_variance, label = 'D = ' + str(D), linewidth=2)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.legend(loc='upper left')
        plt.show()
    else:
        total_variances = []
        for d in D:
            pca = PCA(n_components=d)
            pca.fit(data)
            explained_variance = pca.explained_variance_ratio_
            total_variances.append(sum(explained_variance))
            print('Total variance explained : ',sum(explained_variance),' where D = ',d)
            plt.plot(explained_variance, label = 'D = ' + str(d) , linewidth=2)
            plt.xlabel('Number of components')
            plt.ylabel('Explained variance')
            plt.legend(loc='upper left')
            plt.show()
        plt.plot(D,total_variances, linewidth = 2)
        plt.xlabel('Number of components')
        plt.ylabel('Total variance explained')     
        plt.show()
        
D =   [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# D = 200
# perform_pca(xTrain_subset, D)

# 6 : Using D = 2 perform PCA on the training data and obtain the low dimensional representation of the data.

def visualise_training_2d():
    print('\nVisualize training data in a 2D scatter plot ...')
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(xTrain_subset)
    
    x = principal_components[:,0]
    y = principal_components[:,1]
    labels = yTrain_subset.numpy()
    
    colors = ['b','g','r','m','c']
    label_colors = [colors[i] for i in labels]
    
    plt.scatter(x,y, c=label_colors, s=0.1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
  
# 7: Transform and visualize the test data
def visualise_test_2d():    
    print('\nVisualize test data in a 2D scatter plot ...')
    pca = PCA(n_components=2)
    pca.fit_transform(xTrain_subset)
    xTest_transformed = pca.transform(xTest_subset)
    
    x = xTest_transformed[:,0]
    y = xTest_transformed[:,1]
    labels = yTest_subset.numpy()

    colors = ['b','g','r','m','c']
    label_colors = [colors[i] for i in labels]
    
    plt.scatter(x,y, c=label_colors, s=0.1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# visualise_test_2d()
# visualise_training_2d()


# 8 : Perform k-Means clustering with k = 5 

def kmeans():
    pca = PCA(n_components=2)
    
    principal_components = pca.fit_transform(xTrain_subset, yTrain_subset)
    xTest_transformed = pca.transform(xTest_subset)
    
    kmeans = KMeans(n_clusters=5, random_state=0).fit(principal_components)
    
    colors = ['b','g','r','m','c']
    
    training_labels = kmeans.predict(principal_components)
    test_labels = kmeans.predict(xTest_transformed)
    
    x = principal_components[:,0]
    y = principal_components[:,1]
    
    training_labels_transformed = []
    test_labels_transformed = []
    
    for i in range(len(training_labels)):
        if training_labels[i] == 0:
            training_labels_transformed.append(3)
        elif training_labels[i] == 1:
            training_labels_transformed.append(4)
        elif training_labels[i] == 2:
            training_labels_transformed.append(0)
        elif training_labels[i] == 3:
            training_labels_transformed.append(1)
        else:
            training_labels_transformed.append(2)
            
    for i in range(len(test_labels)):
        if test_labels[i] == 0:
            test_labels_transformed.append(3)
        elif test_labels[i] == 1:
            test_labels_transformed.append(4)
        elif test_labels[i] == 2:
            test_labels_transformed.append(0)
        elif test_labels[i] == 3:
            test_labels_transformed.append(1)
        else:
            test_labels_transformed.append(2)
            
            
    colors = ['b','g','r','m','c']
    label_colors = [colors[i] for i in training_labels_transformed]
    
    plt.scatter(x,y, c=label_colors, s=0.1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')    
    plt.show()
    
    print('Accuracy of training data : ',accuracy_score(yTrain_subset, training_labels_transformed))
    print('Accuracy of test data : ',accuracy_score(yTest_subset, test_labels_transformed))
    



kmeans()


print('Ending ...')
