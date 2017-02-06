from __future__ import division  # floating point division
import numpy as np
import math
import random
import time

import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.weights = None
        self.params = {}
        
    def reset(self, params):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,params)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
        # Could also add re-initialization of weights, so that does not use previously learned weights
        # However, current learn always initializes the weights, so we will not worry about that
        
    def getparams(self):
        return self.params

    def getweights():
        return self.weights
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """        
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        self.params = {}
                
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest
        
class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params={} ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean
        

class FSLinearRegression(Regressor):
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        #Changed to pinv generalized inverse for non-singular matrix problem.
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest


class RidgeRegression(Regressor):

    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5],'lambda':1}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        #Closed Form Ridge Regression
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        lambdaVal=self.params['lambda']
        print "Ridge Regression With",Xless.shape[1]," features and Samples : ",numsamples,"and lambda",lambdaVal
        self.weights = np.dot(np.dot(np.linalg.pinv(np.subtract(np.dot(Xless.T,Xless),(np.identity(Xless.shape[1])*lambdaVal))), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

class BatchGradientDescent(Regressor):
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5],'alpha':1,'iterations':250,'epsilon':.00001}
        self.reset(params)
        self.runtime=[]
        self.errorRT=[]
        
    def learn(self, Xtrain, ytrain):
        runTime=time.time()
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        samples=Xless.shape[0]
        tempw=np.random.rand(Xless.shape[1])
        alpha=self.params['alpha']
        for x in xrange(int(self.params['iterations'])):
            h=np.subtract(np.dot(Xless,tempw),ytrain)
            errOld=np.sum(np.square(h))/samples
            deltaW=(np.dot(Xless.T,h)/samples)
            tAlpha=alpha
            #Backtracking line Search To Choose Alpha
            errNew=np.sum(np.square(np.subtract(np.dot(Xless,(tempw-(tAlpha*deltaW))),ytrain)))/samples
            while errNew>errOld:
                tAlpha=tAlpha/2
                errNew=np.sum(np.square(np.subtract(np.dot(Xless,(tempw-(tAlpha*deltaW))),ytrain)))/samples
            tempw=tempw-(tAlpha*deltaW)          
            #To show how batch gradient step is doing after each step
            if x%100==0:
                print "Step ",(x+1),") Old Cost : ",errOld," New Cost : ",errNew," Step Size : ",tAlpha
            if abs(errOld-errNew)<self.params['epsilon']:
                print "Tolerance Level Reached Exiting Gradient Descent At Step",(x+1)
                break
        runTime=abs(time.time()-runTime) 
        self.runtime.append(runTime)
        self.errorRT.append(np.sum(np.square(np.subtract(np.dot(Xless,tempw),ytrain)))/samples)
        print "Weights Learned By BatchGradientDescent on " + str(len(self.params['features'])) + " features."
        self.weights = tempw
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

    def returnRunTime(self):
        return self.runtime

    def returnErrorRT(self):
        return self.errorRT


class BatchGradientDescentRidgeRegressor(Regressor):

#Extra Implementation Ridge Regression With Gradient Descent

    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5],'alpha':1,'iterations':250,'lambda':1,'epsilon':.00001}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        samples=Xless.shape[0]
        tempw=np.random.rand(Xless.shape[1])
        alpha=self.params['alpha']
        lambdaVal=self.params['lambda']
        for x in xrange(int(self.params['iterations'])):
            regularizationCostVal=(lambdaVal*(np.sum(np.square(tempw[1:]))))/samples
            regularizationDeltaVal=((np.insert(tempw[1:],0,0))*(lambdaVal))/samples
            h=np.subtract(np.dot(Xless,tempw),ytrain)
            errOld=(np.sum(np.square(h))/samples)+regularizationCostVal
            deltaW=(np.dot(Xless.T,h)/samples)+regularizationDeltaVal
            tAlpha=alpha
            #backtracking line Search To Choose Alpha
            errNew=(np.sum(np.square(np.subtract(np.dot(Xless,(tempw-(tAlpha*deltaW))),ytrain)))/samples)+regularizationCostVal
            while errNew>errOld:
                tAlpha=tAlpha/2
                errNew=(np.sum(np.square(np.subtract(np.dot(Xless,(tempw-(tAlpha*deltaW))),ytrain)))/samples)+regularizationCostVal
            tempw=tempw-(tAlpha*deltaW)          
            if x%100==0:
                print "Step ",(x+1),") Old Cost : ",errOld," New Cost : ",errNew," Step Size : ",tAlpha, "Lambda : ",lambdaVal
            if abs(errOld-errNew)<self.params['epsilon']:
                print "Tolerance Level Reached Exiting Gradient Descent At Step",(x+1)
                break
                
        print "Weights Learned By BatchGradientDescentRidgeRegressor on " + str(len(self.params['features'])) + " features."
        self.weights = tempw
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest



class LassoRegression(Regressor):

    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5],'alpha':1,'iterations':250,'lambda':1,'epsilon':.00001}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        #print len(self.params['features'])
        samples=Xless.shape[0]
        tempw=np.random.rand(Xless.shape[1])
        alpha=self.params['alpha']
        lambdaVal=self.params['lambda']

        for x in xrange(int(self.params['iterations'])):
            regularizationCostVal=(lambdaVal*(np.sum(abs(tempw[1:]))))/samples
            #regularizationDeltaVal=((np.divide(abs(tempw[1:]),tempw))*(lambdaVal))/samples
            h=np.subtract(np.dot(Xless,tempw),ytrain)
            errOld=(np.sum(np.square(h))/samples)+regularizationCostVal
            deltaW=(np.dot(Xless.T,h)/samples)#+regularizationDeltaVal
            tAlpha=alpha
            #line Search To Choose Alpha
            errNew=(np.sum(np.square(np.subtract(np.dot(Xless,(tempw-(tAlpha*deltaW))),ytrain)))/samples)+regularizationCostVal
            while errNew>errOld:
                tAlpha=tAlpha/2
                errNew=(np.sum(np.square(np.subtract(np.dot(Xless,(tempw-(tAlpha*deltaW))),ytrain)))/samples)+regularizationCostVal
            tempw=tempw-(tAlpha*deltaW)          
            for k in xrange(tempw.shape[0]):
                if tempw[k]>0:
                    tempw[k]=tempw[k]-(lambdaVal/samples)
                    if tempw[k]<0:
                        tempw[k]=0
                elif tempw[k]<0:
                    tempw[k]=tempw[k]+(lambdaVal/samples)
                    if tempw[k]>0:
                        tempw[k]=0
            if x%100==0:
                print "Step ",(x+1),") Old Cost : ",errOld," New Cost : ",errNew," Step Size : ",tAlpha, "Lambda : ",lambdaVal
                #print tempw
            if abs(errOld-errNew)<self.params['epsilon']:
                print "Tolerance Level Reached Exiting Gradient Descent At Step",(x+1)
                break
        print "Weights Learned By LassoRegression on " + str(len(self.params['features'])) + " features."
        zeroCounter=0
        for x in tempw:
            if x==float(0):
                zeroCounter=zeroCounter+1
        print "By Using Lasso with lambda",lambdaVal,",",zeroCounter,"Features weight were reduced to 0."
        self.weights = tempw
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest


class MPLinearRegression(Regressor):
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5],'iterations':250,'epsilon':.1}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        nNXless=Xless[:,:]
        #print len(self.params['features'])
        for x in xrange(Xless.shape[1]):
            normval = np.linalg.norm(Xless[:,x])
            if normval > 0:
                Xless[:,x] = np.divide(Xless[:,x], normval)
        for ii in range(nNXless.shape[1]):
            maxval = np.max(np.abs(nNXless[:,ii]))
            if maxval > 0:
                nNXless[:,ii] = np.divide(nNXless[:,ii], maxval)
        samples=Xless.shape[0]
        featureList=list(xrange(Xless.shape[1]))
        tempw=np.random.rand(Xless.shape[1])
        featureS,featureW=[],[]
        featureS.append(random.randint(1,Xless.shape[1]))
        featureW.append(tempw[featureS[0]])
        #del featureList[featureS[0]]
        #tempw=tempw[featureList]
        for k in xrange(Xless.shape[1]):
            R=np.subtract(np.dot(nNXless[:,np.asarray(featureS)],np.array(featureW)),ytrain)
            maxValRE=""
            maxValREIndx=""
            for x in featureList:
                if x not in featureS:
                    if maxValRE=="":
                        maxValRE=abs(np.dot(Xless[:,x].T,R))
                        maxValREIndx=x
                    else:
                        tempVal=abs(np.dot(Xless[:,x].T,R))
                        if tempVal>maxValRE:
                            maxValRE=tempVal
                            maxValREIndx=x
            if maxValRE<self.params['epsilon']:
                print "Tolerance Reached With",k,"features"
                break
            featureS.append(maxValREIndx)
            XtrainS = nNXless[:,np.asarray(featureS)]
            featureW = np.dot(np.dot(np.linalg.pinv(np.dot(XtrainS.T,XtrainS)), XtrainS.T),ytrain)
            #print "Residual Error with",k,"features :",maxValRE
            #print featureS
            #print featureW
            #if k==100:
                #break
        tempw=[]
        for x in xrange(Xless.shape[1]):
            if x in featureS:
                tempw.append(featureW[featureS.index(x)])
            else:
                tempw.append(0)
        print "Weights Learned By MPLinearRegression on the above mentioned features."
        self.weights=tempw

        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest


class StochasticGradientDescent(Regressor):
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5],'alpha':1,'epoch':250}
        self.reset(params)
        self.errorGraph=[]
        self.runtime=[]
        self.errorRT=[]
        
    def learn(self, Xtrain, ytrain):
        runTime=time.time()
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        samples=Xless.shape[0]
        tempw=np.random.rand(Xless.shape[1])
        epoch=self.params['epoch']
        alpha=self.params['alpha']
        for i in xrange(epoch):
            print "On Epoch : ",(i+1)
            randomSample=np.random.permutation(samples)
            for x in xrange(samples):
                h=np.subtract(np.dot(Xless[x,:],tempw),ytrain[x])
                errOld=np.square(h)
                deltaW=(np.dot(Xless[x,:].T,h))
                tAlpha=alpha/(x+1)
                tempw=tempw-(tAlpha*deltaW)          
                if x%10000==0:
                    print "Step ",(x+1),") Old Cost : ",errOld," Step Size : ",tAlpha
            self.errorGraph.append(np.sum(np.square(np.subtract(np.dot(Xless,tempw),ytrain)))/samples)
        runTime=abs(time.time()-runTime) 
        self.runtime.append(runTime)
        self.errorRT.append(np.sum(np.square(np.subtract(np.dot(Xless,tempw),ytrain)))/samples)
        print "Weights Learned By StochasticGradientDescent on " + str(len(self.params['features'])) + " features."
        self.weights = tempw
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

    def returnErrorGraph(self):
        return self.errorGraph

    def returnRunTime(self):
        return self.runtime

    def returnErrorRT(self):
        return self.errorRT