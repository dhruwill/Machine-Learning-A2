import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, params):
    return np.round(sigmoid(np.dot(X, params)))

def error(y_pred,target):
    return np.mean((-target*np.log(y_pred)-(1-target)*np.log(1-y_pred)))



if __name__ == "__main__":

	data = pd.read_csv('./data_banknote_authentication.csv')
	X = data.iloc[:, 0:4].values    	#Features
	T = data.iloc[:, 4:5].values 	 		#Result


	X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)

	epoch = 1
	np.random.seed(0)
	W = np.random.uniform(0,1,size=(X_train.shape[1],1)) 	#intial weights
	b=0.5                                           		#bias

	for i in range(5):
		Z = np.dot(X_train, W) + b
		y_predicted = sigmoid(Z)
		error = (y_predicted, T_train)
		print np.shape(error)
		#print("------------->",error)
		grad = y_predicted - T_train
		grad_weight= np.dot(np.transpose(X_train),grad)/(X_train.shape[0])
	   	grad_bias = np.average(grad)
	   	W=W-.01*grad_weight
	  	b=b-.01*grad_bias
