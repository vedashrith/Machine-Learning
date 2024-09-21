import numpy as np
from LinearRegression import LinearRegression 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the Iris dataset
iris = load_iris()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)



X_train_m2 = X_train_sc[:, (2,)]
Y_train_m2 = X_train[:, (1,)]

model = LinearRegression()
model.fit(X_train_m2, Y_train_m2)

    
plt.plot(model.loss_step)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step for predicting petal length using sepal width')
plt.grid(True)
plt.savefig('model2_loss.png')
plt.show()

print('Non-Regularized model:')
print('Weights: ',model.weights)
print('Bias:', model.bias)

model.save('model2_parameters.npz')
print("Model parameters saved as model2_parameters.npz in the current directory!!!")

model_reg = LinearRegression(regularization=0.1)
model_reg.fit(X_train_m2, Y_train_m2)

# Plot loss history
plt.plot(model_reg.loss_step)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step for Regularized model 2')
plt.grid(True)
plt.savefig('Reg_model2_loss.png')
plt.show()

print('Regularized model:')
print('Weights: ',model_reg.weights)
print('Bias:', model_reg.bias)

model_reg.save('Reg_model2_parameters.npz')
print("Model parameters saved as Reg_model2_parameters.npz in the current directory!!!")

print('Weight differnce:',(model.weights-model_reg.weights))
print('Bias difference: ', (model.bias - model_reg.bias))
