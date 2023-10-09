# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values
3. Import linear regression from sklearn.
4. Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DIVYA.A
RegisterNumber: 212222230034 
*/
#df.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

#df.tail()
df.tail()

#Array values of X
X=df.iloc[:,:1].values
X

#Array  values of Y
Y=df.iloc[:,-1].values
Y

#Values of Y prediction
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

#Array values of Y test
Y_test

#Training Set Graph
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Testing Set Graph
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,regressor.predict(X_test),color="pink")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Values of MSE,MAE and RMSE
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
# df.head()
![EX 02 (1)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/255a1aae-0913-4f0c-85fd-316e13ea7acd)

# df.tail()
![EX 02 (2)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/04704f85-db38-4e9b-899d-4e80f945fa2c)

# Array values of X
![EX 02 (3)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/db613a50-f1e5-4a90-8bb8-e60720b3f423)

# Array values of Y
![EX 02 (4)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/c91c4b1c-5dbd-4166-93d0-3f5febdf5e42)

# Values of Y prediction
![EX 02 (5)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/883bc75a-9c98-4315-8a44-8895f442beb1)

# Array values of Y test
![EX 02 (6)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/0ece98ac-e71d-4c50-9d88-48cc1878b39f)

# Training Set Graph
![EX 02 (7)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/0eeab6e6-d1ee-496b-a1a0-dbc1af087720)

# Testing Set Graph
![EX 02 (8)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/2b04514f-c73d-4481-9a4d-a8282c769245)

# Values of MSE,MAE and RMSE
![EX 02 (9)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/2ffc5aa5-abb6-463d-92cc-6d570dce8f75)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
