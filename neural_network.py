import numpy as np
import pandas as pd
import random
import math

def get_data(filename):
    import csv
    data=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count=0
        for row in csv_reader:
            if line_count!=0:
                a=[]
                for i in range(len(row)-1):
                    a.append(float(row[i]))
                a.append(int(row[len(row)-1]))
                data.append(a)
            line_count+=1
    return standardize_data(data)

def standardize_data(data):
    a={}
    for i in range(len(data[0])):
        a[i]=[]
    for i in range(len(data)):
        for j in range(len(data[i])):
            a[j].append(data[i][j])
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=(data[i][j]-min(a[j]))/(max(a[j])-min(a[j]))
    return data


class NN:
    def __init__(self,data):
        self.data=data
        self.columnlabel=len(self.data[0])-1
        self.input_layer_dim=self.columnlabel
        random.shuffle(self.data)
        self.training_data=self.data[:int(len(self.data)*0.8)]
        self.testing_data=self.data[int(len(self.data)*0.8):]
        self.layers={}
        self.shape_layers={}
        self.ct_layers=2
        self.ct_weights=1
        self.layers[1]=np.zeros([self.input_layer_dim,1])
        self.layers[2]=np.zeros([1,1])
        self.activation_functions={}
        self.activation_functions[2]='sigmoid'
        self.weights={}
        self.weights[1]=np.zeros([1,self.input_layer_dim])
        self.shape_layers[1]=self.input_layer_dim
        self.shape_layers[2]=1

    def add_layer(self,neurons,activation_function):
        self.layers[self.ct_layers+1]=self.layers[self.ct_layers]
        self.layers[self.ct_layers]=np.zeros([neurons,1])
        self.activation_functions[self.ct_layers+1]=self.activation_functions[self.ct_layers]
        self.activation_functions[self.ct_layers]=activation_function
        self.shape_layers[self.ct_layers+1]=self.shape_layers[self.ct_layers]
        self.shape_layers[self.ct_layers]=neurons
        self.ct_weights+=1
        self.ct_layers+=1
        for i in range(1,self.ct_weights+1):
            self.weights[i]=np.random.rand(self.shape_layers[i+1],self.shape_layers[i])
            # self.weights[i]=np.zeros([self.shape_layers[i+1],self.shape_layers[i]])


    def sigmoid(self,x):
        z = 1/(1 + np.exp(-x))
        return z

    def d_sigmoid(self,x):
        return (self.sigmoid(x))*(1-self.sigmoid(x))

    def tanh(self,x):
        z=np.tanh(x)
        return z

    def d_tanh(self,x):
        return 1-(self.tanh(x))**2

    def feedforward(self,input,expected_output):
        for i in self.layers:
            if i==1:
                self.layers[i]=np.array(input)
                self.layers[i].reshape([self.shape_layers[i],1])
            else:
                if self.activation_functions[i]=='sigmoid':
                    self.layers[i]=self.sigmoid(self.weights[i-1].dot(self.layers[i-1]))
                elif self.activation_functions[i]=='tanh':
                    self.layers[i]=self.tanh(self.weights[i-1].dot(self.layers[i-1]))
        predicted=self.layers[self.ct_layers]
        expected_output=np.array([expected_output])
        expected_output=np.array([expected_output])
        return predicted,expected_output

    def backprop(self,predicted,expected_output):
        deriv_w={}
        for i in range(self.ct_weights,0,-1):
            if i==self.ct_weights:
                if self.activation_functions[i+1]=='sigmoid':
                    deriv_w[i]=self.layers[i].dot(2*(expected_output-predicted).dot(self.d_sigmoid(self.layers[i+1])))
                elif self.activation_functions[i+1]=='tanh':
                    deriv_w[i]=self.layers[i].dot(2*(expected_output-predicted).dot(self.d_tanh(predicted)))
                deriv_w[i]=deriv_w[i].transpose()
            else:
                if self.activation_functions[i+1]=='sigmoid':
                    last=self.d_sigmoid(self.layers[i+1])
                    mid=deriv_w[i+1].transpose().dot(self.weights[i+1])
                    term=mid.dot(last)
                    term=term.transpose()
                    deriv_w[i]=self.layers[i].dot(term)
                    deriv_w[i]=deriv_w[i].transpose()
                elif self.activation_functions[i]=='tanh':
                    last=self.d_tanh(self.layers[i+1])
                    mid=deriv_w[i+1].transpose().dot(self.weights[i+1])
                    term=mid.dot(last)
                    term=term.transpose()
                    deriv_w[i]=self.layers[i].dot(term)
                    deriv_w[i]=deriv_w[i].transpose()
        return deriv_w

    def train_model(self,epochs=10000,batch_size=256,sample=100,learning_rate=0.002):
        for epoch in range(1,epochs+1):
            train_data=[]
            loss=0.
            ct=0.
            tc=0.
            for i in range(batch_size):
                idx=random.randrange(0,len(self.training_data))
                train_data.append(self.training_data[idx])
            for i in train_data:
                input=i[:len(i)-1]
                input=np.array(input)
                input=input.reshape([len(input),1])
                expected_output=i[len(i)-1]
                predicted,expected_output=self.feedforward(input,expected_output)
                # print(epoch,predicted,expected_output)
                if(predicted<=0.5):
                    predicted=0
                else:
                    predicted=1
                loss+=(predicted-expected_output)**2
                tc+=1
                if predicted==expected_output:
                    ct+=1
                deriv_w=self.backprop(predicted,expected_output)
                for i in self.weights:
                    self.weights[i]+=learning_rate*deriv_w[i]
            if epoch%sample==0:
                print("Epoch = ",epoch,end='\t')
                print("Loss = ",loss,end='\t')
                print("Accuracy = ",(ct/tc)*100)
                # print(self.weights)

    def test_model(self):
        tp=0.
        fp=0.
        tn=0.
        fn=0.
        for i in self.testing_data:
            input=i[:len(i)-1]
            input=np.array(input)
            input=input.reshape([len(input),1])
            expected_output=i[len(i)-1]
            predicted,expected_output=self.feedforward(input,expected_output)
            # print(epoch,predicted,expected_output)
            if(predicted<=0.5):
                predicted=0
            else:
                predicted=1
            if expected_output==1:
                if predicted==1:
                    tp+=1
                else:
                    fn+=1
            else:
                if predicted==1:
                    fp+=1
                else:
                    tn+=1
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def stats(self):
        self.precision=self.tp/(self.tp+self.fp)
        self.recall=self.tp/(self.tp+self.fn)
        self.f_score=(2*self.precision*self.recall)/(self.precision+self.recall)
        self.accuracy=(self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
        print("Accuracy = ",self.accuracy)
        print("F-Score = ",self.f_score)

    def check_model(self):
        print(self.ct_layers)
        print(self.shape_layers)
        print(self.layers)
        print(self.activation_functions)
        print(self.ct_weights)
        print(self.weights)

if __name__=='__main__':
    filename='housepricedata.csv'
    data=get_data(filename)
    model=NN(data)
    model.add_layer(5,'sigmoid')
    model.add_layer(3,'sigmoid')
    # model.add_layer(10,'sigmoid')
    model.train_model()
    model.test_model()
    model.stats()
