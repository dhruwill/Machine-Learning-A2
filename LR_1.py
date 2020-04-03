import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":

	data = load_data("data_banknote_authentication.txt", None)
	X = data.iloc[:, :-1]    #Features
	Y = data.iloc[:, -1]	 #Result
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
	
	

		

