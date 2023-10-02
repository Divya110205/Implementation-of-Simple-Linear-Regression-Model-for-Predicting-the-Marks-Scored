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
![EX 02 (1)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/ce6743b6-cd9b-41d5-9660-10a62dc01362)

# df.tail()
![EX 02 (2)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/9d7f7fa0-ac2d-4770-897e-b3806d69c36d)

# Array values of X
![EX 02 (3)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/fe5c5267-1351-474b-a6b9-fec743b68b64)

# Array values of Y
![EX 02 (4)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/80c5a4de-42a5-4feb-b37d-f8ae3670a36f)

# Values of Y prediction
![EX 02 (5)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/ea0add51-f811-4dd8-b26d-007a6a5f4c24)

# Array values of Y test
![EX 02 (6)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/209b51e9-a503-4b0f-8e05-0e0ff12f1a41)

# Training Set Graph
![EX 02 (7)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/bd103356-02bc-4dbc-a5ea-d98170f7cfb9)

# Testing Set Graph
![EX 02 (8)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/5a0e8904-bfae-4def-80a2-51d500634e1b)

# Values of MSE,MAE and RMSE
![EX 02 (9)](https://github.com/Divya110205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404855/5a885949-f602-45cb-842b-1fdb4c042c86)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
