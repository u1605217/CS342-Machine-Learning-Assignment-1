import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as kNN
from sklearn import metrics
import csv

#read probeA.csv, classA.csv and probeB.csv
probeA = pd.read_csv("../probeA.csv", header = 0)
classA = pd.read_csv("../classA.csv", header = 0)
probeB = pd.read_csv("../probeB.csv", header = 0)

#create sorted probeA 
df = probeA.copy()
	
with open("probeA_sorted.csv","wb") as f:
	thewriter = csv.writer(f)
	thewriter.writerow(['c1','c2','c3','m1','m2','m3','n1','n2','n3','p1','p2','p3'])
		
	for i in range(0,1000):
		c = [df.loc[i,'c1'],df.loc[i,'c2'],df.loc[i,'c3']]
		newc = sorted(c)
		m = [df.loc[i,'m1'],df.loc[i,'m2'],df.loc[i,'m3']]
		newm = sorted(m)
		n = [df.loc[i,'n1'],df.loc[i,'n2'],df.loc[i,'n3']]
		newn = sorted(n)
		p = [df.loc[i,'p1'],df.loc[i,'p2'],df.loc[i,'p3']]
		newp = sorted(p)
		thewriter.writerow(newc+newm+newn+newp)

probeAs = pd.read_csv("probeA_sorted.csv",header=0)

#create sorted probeB
df2 = probeB.copy()
	
with open("probeB_sorted.csv","wb") as f:
	thewriter = csv.writer(f)
	thewriter.writerow(['c1','c2','c3','m1','m2','m3','n1','n2','n3','p1','p2','p3'])
		
	for i in range(0,1000):
		c = [df2.loc[i,'c1'],df2.loc[i,'c2'],df2.loc[i,'c3']]
		newc = sorted(c)
		m = [df2.loc[i,'m1'],df2.loc[i,'m2'],df2.loc[i,'m3']]
		newm = sorted(m)
		n = [df2.loc[i,'n1'],df2.loc[i,'n2'],df2.loc[i,'n3']]
		newn = sorted(n)
		p = [df2.loc[i,'p1'],df2.loc[i,'p2'],df2.loc[i,'p3']]
		newp = sorted(p)
		thewriter.writerow(newc+newm+newn+newp)

probeBs = pd.read_csv("probeB_sorted.csv",header=0)

#create function to standardise attributes
def scaledata(dataframe, tlabel):
	
	df = dataframe.copy()
	df[tlabel]= (df[tlabel] - df[tlabel].mean())/df[tlabel].std()
	
	return df[tlabel]

#create function to return r2 and predictions of tna from probeB data
def Rsquared(model,dataframe,target,dataframe_predict):
	
	#scale the attributes
	attributes = ["c1","c2","c3","m1","m2","m3","n1","n2","n3","p1","p2","p3"]
	scaled_attributes = scaledata(dataframe, attributes)
	scaled_predict = scaledata(dataframe_predict, attributes)
        
	#fitting training data
	model.fit(scaled_attributes, target)
	
	#calculate predictions according to fitted model
	probeB_tna = model.predict(scaled_predict)
	
	#calculate Rsquared
	r2 = metrics.r2_score(target,model.predict(scaled_attributes))
    
	return r2, probeB_tna
	
#kNNreg model 
kNNregModel = kNN(n_neighbors=6)

#Rsquared of kNNreg
print "kNNreg:",Rsquared(kNNregModel,probeAs,probeA["tna"],probeBs)[0]

#output probeB_tna to csv file "tnaB.csv"
with open("tnaB.csv","wb") as f:
	thewriter = csv.writer(f)
	thewriter.writerow(["tna"])
	for val in Rsquared(kNNregModel,probeAs,probeA["tna"],probeBs)[1]:
		thewriter.writerow([val])