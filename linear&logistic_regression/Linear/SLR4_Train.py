import numpy as np
from LinearRegression import LinearRegression  
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the Iris dataset
iris = load_iris()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, stratify=iris.target, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)



X_train_m1 = X_train_sc[:, (0,)]
Y_train_m1 = X_train[:, (3,)]

model = LinearRegression()
model.fit(X_train_m1, Y_train_m1)

plt.plot(model.loss_step)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step for predicting petal width using sepal length')
plt.grid(True)
plt.savefig('model4_loss.png')
plt.show()

print('Non-Regularized model:')
print('Weights: ',model.weights)
print('Bias:', model.bias)

model.save('model4_parameters.npz')
print("Model parameters saved as model4_parameters.npz in the current directory!!!")

model_reg = LinearRegression(regularization=0.1)
model_reg.fit(X_train_m1, Y_train_m1)

# Plot loss history
plt.plot(model_reg.loss_step)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step for Regularized model 4')
plt.grid(True)
plt.savefig('Reg_model4_loss.png')
plt.show()

print('Regularized model:')
print('Weights: ',model_reg.weights)
print('Bias:', model_reg.bias)

model.save('Reg_model4_parameters.npz')
print("Model parameters saved as Reg_model4_parameters.npz in the current directory!!!")

print('Weight differnce:',(model.weights-model_reg.weights))
print('Bias difference: ', (model.bias - model_reg.bias))
