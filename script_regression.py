from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1) 

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    #trainsize = 1000
    #testsize = 5000
    LinearMetaParameter=[0]
    regularizationMetaParameter=[1]

    numparams = 1
    numruns = 1
    p = 0
    r = 0
    nFoldSplit=10
    params = {}
    

    print "The Following Algorithms Will Run:"
    print "1)FSLinearRegression"
    print "2)RidgeRegression"
    print "3)GradientDescentRidgeRegression"
    print "4)MPLinearRegression"
    print "5)LassoRegression"
    print "6)BatchGradientDescentRegression"
    print "7)StochasticGradientDescentRegression"
    print "-Lambda Parameter May Not Be Used In All Models Though It Will Be Displayed For All Algorithms-"
    print ""
    

    paramList=list(range(1,11,1))
    trainTest=raw_input("How would you like to Seperate Your Data For Training and Testing On CT Scan. Enter your choice as 1 or 2.\n1) 10-Fold\n2) 70%-30% (Random Train-Test Split)\nEnter Value : ")
    if int(trainTest) not in [1,2]:
        trainTest=2
    numruns=raw_input("How many iterations over Data You Want To Go. Max 10.\nEnter Value : ")
    if int(numruns) not in paramList:
        numruns=1
    numparams=raw_input("For How Many Lambda Regularization Parameters You Want The Code To Run. Max 10.\nEnter Value : ")
    if int(numparams) not in paramList:
        numparams=1
    print "Enter Lambda Parameters. Example Enter In Form of : {1,10,100} without curly brackets."
    regularizationMetaParameter=map(float,raw_input('Enter List Values : ').strip().split(','))
    if len(regularizationMetaParameter)!=int(numparams):
        regularizationMetaParameter=[1]
        print "Something Went Wrong Lambda Default has Been Set To 1"
        print ""

    print ""
    print "-----Your Choices-----"
    seventyThirtyCrossValidation=True
    if int(trainTest) == 1:
        seventyThirtyCrossValidation=False
        print "10-Fold Data Split Selected"
    else:
        print "70-30 Random Data Split Selected"


    print "Number Of Iterations :",numruns
    print "Lambda Values : "
    print regularizationMetaParameter
    print ""
    print ""
    #Load The DataSet In The Memory
    print "Fetching Data ..."
    dataset=dtl.load_Nctscan()
    print "CT Scan DataSet Loaded with Data Samples",dataset.shape[0],"and Features",dataset.shape[1]-1
    

    #Which Algorithms To Run
    regressionalgs = {
                'FSLinearRegression': algs.FSLinearRegression({'features': range(dataset.shape[1]-1)}),
                'RidgeRegression': algs.RidgeRegression({'features': range(dataset.shape[1]-1),'lambda':1}),
                'GradientDescentRidgeRegression': algs.BatchGradientDescentRidgeRegressor({'features': range(dataset.shape[1]-1),'alpha':1,'iterations':500,'lambda':1,'epsilon':.00001}),
                'MPLinearRegression': algs.MPLinearRegression({'features': range(dataset.shape[1]-1),'epsilon':.1}),
                'LassoRegression': algs.LassoRegression({'features': range(dataset.shape[1]-1),'alpha':1,'iterations':500,'lambda':10,'epsilon':.00001}),
                'BatchGradientDescentRegression': algs.BatchGradientDescent({'features': range(dataset.shape[1]-1),'alpha':1,'iterations':500,'epsilon':.00001}),
                'StochasticGradientDescentRegression': algs.StochasticGradientDescent({'features': range(dataset.shape[1]-1),'alpha':0.01,'epoch':25})
             }       
    numalgs = len(regressionalgs)


    #Storing Errors With Respect To Algorithms
    errors = {}
    numparams=int(numparams)
    numruns=int(numruns)
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))
        #paramAlg[learnername]
    

    batchRunTime,stocRunTime,batchErrorRT,stocErrorRT,batchFlag,stocFlag=[],[],[],[],False,False
    
    saveTrainSet,saveTestSet,mpRun=None,None,False
    for nfold in xrange(numruns):
        print ""
        print "---****##### RUN (",(nfold+1),") #####****---"
        print ""
        raw_input("Press Enter To Start The New Run Cycle")
        trainset, testset = dtl.nFoldSplit(dataset,fold=(nfold+1),regularization=seventyThirtyCrossValidation,MPlinearRegression=False)
        print "***New DataSet Selected. For MPLinearRegression The DataSet Would Be Selected Differently Due To Normalization***"
        print ""
        for learnername, learner in regressionalgs.iteritems():
            raw_input("Press Enter To Start The Algorithm Run for : "+str(learnername))
            #trainset, testset = dtl.nFoldSplit(dataset,fold=(nfold+1),Nfold=nFoldSplit,regularization=True)
            #params['weights']=np.random.rand(dataset.shape[1]-1)
            #We don't want Normalization upfront Of DataSet if The algo is MPlinearRegression, Normalization is done in the algo itself
            if mpRun:
                mpRun=False
                print "#####"
                print "----------------Old Data Restored After MPLinearRegression----------------"
                print "#####"
                trainset,testset=saveTrainSet[:],saveTestSet[:]
            if learnername=='MPLinearRegression':
                saveTrainSet,saveTestSet=trainset[:],testset[:]
                trainset, testset = dtl.nFoldSplit(dataset,fold=nfold,regularization=seventyThirtyCrossValidation,MPlinearRegression=True)  
                mpRun=True
            #Restore DataSet After MP Run
            p,r=0,0
            for nParam in xrange(numparams):
                print ""
                print "----- Run",(nfold+1),"Parameter",(nParam+1),"for",learnername,"-----"
                print('Running {2} on trainSet={0} and testSet={1} and lambdaParameter={3}').format(trainset[0].shape[0], testset[0].shape[0],learnername,regularizationMetaParameter[nParam])
                params['lambda']=regularizationMetaParameter[nParam]
                learner.reset(params)
            	# Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                Mprediction=learner.predict(trainset[0])
                error = geterror(testset[1], predictions)
                print 'Test Prediction Error : ' + learnername + ': ' + str(error)
                errors[learnername][nParam,nfold] = error
                print 'Model Error : ' + str(geterror(trainset[1],Mprediction))
                if learnername=='StochasticGradientDescentRegression':
                    stocRunTime=learner.returnRunTime()
                    errVals=learner.returnErrorGraph()
                    stocErrorRT=learner.returnErrorRT()
                    epochs=list(range(1,len(errVals)+1,1))
                    plt.clf()
                    plt.plot(epochs,errVals,'ro')
                    plt.plot(epochs,errVals,'r')
                    plt.xlabel("Epochs")
                    plt.ylabel("l2err_squared")
                    plt.show()
                    stocFlag=True
                p=p+1
                if learnername=='BatchGradientDescentRegression':
                    batchRunTime=learner.returnRunTime()
                    batchErrorRT=learner.returnErrorRT()
                    batchFlag=True
                print ""
            r=r+1
    

    for key in errors:
        print ""
        print "***#####***"
        print "Some Intuition About",key
        print "Errors For This Algorithm"
        print errors[key]
        print ""
        print ""
        print "More Explanation As Follow:"
        for x in xrange(numparams):
            print "Regularization Parameter is :",regularizationMetaParameter[x]
            print "Mean Error for",numruns,"runs : ",np.mean(np.array(errors[key][x,:]))
            print "Standard Deviation Error for",numruns,"runs : ",np.std(np.array(errors[key][x,:]))
            print ""
        print "***#####***"
        raw_input("Enter To Continue:")


    if stocFlag and batchFlag:
        print ""
        print "Stochastic RunTime VS L2_SquaredError"
        print "\nStochastic Run Time:"
        print stocRunTime
        print "\nStochastic Error:"
        print stocErrorRT
        print ""
        print "Batch RunTime VS L2_SquaredError"
        print "\nBatch Run Time:"
        print batchRunTime
        print "\nBatch Error:"
        print batchErrorRT
        plt.clf()
        plt.plot(stocRunTime,stocErrorRT,'ro')
        plt.plot(batchRunTime,batchErrorRT,'bs')
        plt.xlabel("RunTime")
        plt.ylabel("l2err_squared")
        red_patch = mpatches.Patch(color='red', label='Stochastic GD')
        blue_patch = mpatches.Patch(color='blue', label='Batch GD')
        plt.legend(handles=[red_patch,blue_patch],loc=2)
        plt.show()
