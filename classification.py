#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score






#loading database
data=pd.read_csv(r'C:\Users\MUTHU PS\Downloads\train_mnist.csv')

#extracting data from the dataset and viewing them up close
a=data.iloc[3,1:]
print(a)

#reshaping the extracted data into resonable size
a=a.values.reshape(28,28).astype('uint8')
plt.imshow(a)

#preparing the data
#seperating label and data values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


#creating test and train batches
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)


#check data
y_train.head()

#call rf indicator
rf=RandomForestClassifier(n_estimators=100)

#fit the model
rf.fit(x_train,y_train)


#prediction on test data
pred=rf.predict(x_test)
pred

y=y_test.values

#calculate the number of correctly predicted values
count=0
for i in range(len(pred)):
    if pred[i]==a.any():
        count=count+1
print(count)
print(len(pred))   

type(a)

type(pred)

a.head()

print(pred)

