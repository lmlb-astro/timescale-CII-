import numpy as np
import sys
import matplotlib.pyplot as plt

#### DEFINITIONS TO FIT GAUSSIANS TO THE SPECTRA #####

## the 1d gaussian for fitting (x0: central velocity, sigma: width)
def gaussian1d(x,A,x0,sigma):
	return A*np.exp(-(x-x0)**2/(2.*sigma**2))

## a double 1d gaussian for fitting
def doubleGaussian1d(x,A1,x1,sigma1,A2,x2,sigma2):
	return A1*np.exp(-(x-x1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-x2)**2/(2.*sigma2**2))

## a double 1d gaussian for fitting
def tripleGaussian1d(x,A1,x1,sigma1,A2,x2,sigma2,A3,x3,sigma3):
	return A1*np.exp(-(x-x1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-x2)**2/(2.*sigma2**2)) + A3*np.exp(-(x-x3)**2/(2.*sigma3**2))

## get the result of a function (specified by 'func')
## works for gaussian1d, doubleGaussian1d, and tripleGaussian1d so far
def getGaussianFunction(func,x,inputVals):
	if(func == gaussian1d):
		return [func(xi,inputVals[0],inputVals[1],inputVals[2]) for xi in x]
	elif(func == doubleGaussian1d):
		return [func(xi,inputVals[0],inputVals[1],inputVals[2],inputVals[3],inputVals[4],inputVals[5]) for xi in x]
	elif(func == tripleGaussian1d):
		return [func(xi,inputVals[0],inputVals[1],inputVals[2],inputVals[3],inputVals[4],inputVals[5],inputVals[6],inputVals[7],inputVals[8]) for xi in x]
	else:
		sys.exit('You asked for a guassian that does not exist, so the program can not continue')


## plots the components of the double and triple gaussians
def plotGaussianComponents(x,inputVals):
	if(len(inputVals)%3 == 0):
		for a in range(0,len(inputVals),3):
			gaussArr = [gaussian1d(xi,inputVals[a],inputVals[a+1],inputVals[a+2]) for xi in x]
			plt.plot(x,gaussArr,'g-')
	else:
		sys.exit('No correct input was given to plot the gaussian components')
