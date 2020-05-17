#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


#loading the database
boston=load_boston()
print(boston.DESCR)

#access the data attributes
dataset=boston.data
for index,name in enumerate(boston.feature_names):
    print(index,name)

#reshaping the data
data=dataset[:,12].reshape(-1,1)

#shape of the data
np.shape(dataset)

#target value
target=boston.target.reshape(-1,1)

#shape of the target
np.shape(target)

#ensuring that matplotlib working in the notebook
%matplotlib inline
plt.scatter(data,target,color='green')
plt.xlabel("lower income population")
plt.ylabel("cost of house")
plt.show()


#regression
from sklearn.linear_model import LinearRegression


#creating a regression model
model=LinearRegression()

#fit the model
model.fit(data,target)



#prediction
pred=model.predict(data)

#using matplotib
%matplotlib inline
plt.scatter(data,target,color="green")
plt.plot(data,pred,color="red")
plt.xlabel("lower income population")
plt.ylabel("cost of house")
plt.show()

#circumventing curve issue using polynomial model
from sklearn.preprocessing import PolynomialFeatures

#to allow merging of models
from sklearn.pipeline import make_pipeline
model2=make_pipeline(PolynomialFeatures(3),model)


#fit the data
model2.fit(data,target)

#predicting
pred=model2.predict(data)


#using matplotlib
%matplotlib inline
plt.scatter(data,target,color="green")
plt.plot(data,pred,color="red")
plt.xlabel("lower income population")
plt.ylabel("cost of house")
plt.show()

#r_2metric to judge performance of our model
from sklearn.metrics import r2_score

#predict
print(r2_score(pred,target))

