import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_test_sc = scaler.fit_transform(X_test)  # Scale the test features


X_test_m1 = X_test_sc[:, (2,)]  
y_test_m1 = X_test_sc[:, (3,)]  

model = LinearRegression()
model.load('Reg_model3_parameters.npz')

y_pred = model.predict(X_test_m1)
mse = model.score(X_test_m1, y_test_m1)

# Print the mean squared error
print("Mean Squared Error for model 3:", mse)


