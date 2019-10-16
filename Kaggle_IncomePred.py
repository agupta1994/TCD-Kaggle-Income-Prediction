import pandas as pd
import numpy as np

df = pd.read_csv('D:/TCD/Study/Machine Learning/Kaggle/Data/tcd ml 2019-20 income prediction training (with labels).csv')
df = df.rename(index=str, columns={"Income in EUR": "Income"})
Test_Data=pd.read_csv('D:/TCD/Study/Machine Learning/Kaggle/Data/tcd ml 2019-20 income prediction test (without labels).csv')
df=pd.concat([df,Test_Data], sort=False)

df = df.drop("Instance", axis=1)     
df['University Degree'] = df['University Degree'].replace('0', "No")
df['Hair Color'] = df['Hair Color'].replace('0', "Unknown")
age_mean = df['Age'].mean()
Record_mean=df['Year of Record'].mean()
df['Age'] = df['Age'].replace('#N/A',age_mean )
df['Year of Record'] = df['Year of Record'].replace('#N/A',Record_mean )
df['Gender'] = df['Gender'].replace('0', "Other")
df['Gender'] = df['Gender'].replace('#N/A', "Other")
data2 = pd.get_dummies(df, columns=["Gender"])
data2 = pd.get_dummies(data2, columns=["Country"])
data2 = pd.get_dummies(data2, columns=["Profession"])
data2 = pd.get_dummies(data2, columns=["University Degree"])
data2 = pd.get_dummies(data2, columns=["Hair Color"])

train = data2[0:111994]
train = train.dropna()

Y = train.Income  
X = train.drop("Income", axis=1)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
model = LR.fit(X_train, Y_train)
 
Y_pred = LR.predict(X_test)

test = data2[111993:]
X_test = test.drop("Income", axis=1)

age_mean = X_test['Age'].mean()
Record_mean=X_test['Year of Record'].mean()
X_test['Age'] = X_test['Age'].replace(pd.np.nan,age_mean )
X_test['Year of Record'] = X_test['Year of Record'].replace(pd.np.nan,Record_mean )

y1 = LR.predict(X_test)