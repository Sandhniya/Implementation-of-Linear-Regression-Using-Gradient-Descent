# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SANDHIYA SREE B
RegisterNumber:212223220093
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta


data=pd.read_csv("/content/50_Startups.csv")
data.head()


X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)


theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![linear regression using gradient descent](sam.png)

DATA INFORMATION:

<img width="908" height="341" alt="image" src="https://github.com/user-attachments/assets/f187dd1a-870f-426a-8f85-e08a85b574f5" />

VALUE OF X:


<img width="399" height="883" alt="image" src="https://github.com/user-attachments/assets/82a788bb-9619-4045-a007-28980fa33ce3" />

VALUE OF X1_SCALED:

<img width="486" height="688" alt="image" src="https://github.com/user-attachments/assets/001f4946-4dd9-4e46-8079-1b8f9fa643f0" />

<img width="489" height="433" alt="image" src="https://github.com/user-attachments/assets/eda33fb5-2856-41bf-bc2a-a502befaa5e7" />

PREDICTED VALUE:


<img width="375" height="41" alt="image" src="https://github.com/user-attachments/assets/5338708f-9554-4919-8b53-8eb0661b2638" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
