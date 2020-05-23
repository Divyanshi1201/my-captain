import sys
from matplotlib import pyplot as plt
import scipy
import pandas 
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier




#loading dataset using pandas
df=read_csv(r"C:\Users\Kishore\Downloads\iris.csv")

print(df.shape)

df.drop(["Id"],axis=1,inplace=True)

df

print(df.head(20))



print(df.describe())


print(df.groupby("Species").size())

#visualization of data
df.plot(kind="box", color="b")
plt.show()

df.hist()
plt.show()

#multivariate plots
scatter_matrix(df)
plt.show()

#splitting data to create a validation dataset
x=df.iloc[:,0:4]
y=df.iloc[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101, shuffle=True)


#building models
models=[]
models.append(("LR",LogisticRegression(solver="liblinear",multi_class="ovr")))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("NB",GaussianNB()))
models.append(("SVM",SVC(gamma="auto")))





#evaluate the created models
results=[]
names=[]
for name,model in models:
    kfold=StratifiedKFold(n_splist=10,rand,random_state=1)
    cv_results=cross_val_score(models,x_train,y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print(name,cv_results.mean(),cv_results.std())



 
