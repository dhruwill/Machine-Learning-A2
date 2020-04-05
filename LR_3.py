import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    return np.round(sigmoid(np.dot(X, w)+ b))

def loss(y_pred,t):
	return np.mean(-t*np.log(y_pred)-(1-t)*np.log(1-y_pred))

def accuracy(y_pred, test):
	acc = 0
	not_acc = 0

	for i in range(0, y_pred.shape[0]):
		if y_pred[i] == test[i]:
			acc = acc + 1
		else:
			not_acc = not_acc + 1
	return (float(acc)/float(acc + not_acc))

def stan(X):
	mean = np.mean(X, axis=0)			# axis = 0 for along the column
	std = np.std(X, axis=0, ddof=1)     #ddof = 1 for sample standard deviation
 	
 	X = X.astype(float)

	for i in range(0, X.shape[0]):
		for j in range(0, X.shape[1]):
			X[i][j] = (float(X[i][j] - mean[j])/float(std[j]))

	return X


if __name__ == "__main__":

	data = pd.read_csv('./data_banknote_authentication.csv')
	X = data.iloc[:, 0:4].values    	#Features
	T = data.iloc[:, 4:5].values 	 		#Result


	X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)

	X_train_standard = stan(X_train)
	X_test_standard = stan(X_test)
	
	epoch = 5000
	lr = 0.5
	L2_coeff = 0.0001
	np.random.seed(0)
	W = np.random.uniform(0,1,size=(X_train.shape[1],1)) 	#intial weights
	b = 1                                        			    #bias

	for i in range(epoch):
		Z = np.dot(X_train, W) + b
		y_predicted = sigmoid(Z)
		error = loss(y_predicted, T_train)
		print("------------->",error)                     	
		grad = y_predicted - T_train
		gradient= np.dot(np.transpose(X_train),grad)/(X_train.shape[0])
		grad_bias = np.average(grad)
		W = W - lr*gradient + L2_coeff*2*(W)
		b = b - lr*grad_bias + L2_coeff*2*(b)

	print(W)
	print(b)

	T_pred = predict(X_test, W, b)
	a = accuracy(T_pred, T_test)
	print("accuracy",a)
	

	






		

