import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = load_iris()

X = iris.data[:, :2]
y = iris.data[:, 2:]  

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

plt.plot(model.loss_step)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step for predicting sepal length using sepal width')
plt.grid(True)
plt.savefig('model0_loss.png')
plt.show()

print('Non-Regularized model:')
print('Weights: ',model.weights)
print('Bias:', model.bias)

model.save('model1_parameters.npz')
print("Model parameters saved as modelo_parameters.npz in the current directory!!!")

model_reg = LinearRegression(regularization= 0.1)
model_reg.fit(X_train, y_train)

# Plot loss history
plt.plot(model_reg.loss_step)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step for Regularized model 1')
plt.grid(True)
plt.savefig('Reg_model0_loss.png')
plt.show()

print('Regularized model:')
print('Weights: ',model_reg.weights)
print('Bias:', model_reg.bias)

model_reg.save('Reg_model1_parameters.npz')
print("Model parameters saved as Reg_modelo_parameters.npz in the current directory!!!")

print('Weight differnce:',(model.weights-model_reg.weights))
print('Bias difference: ', (model.bias - model_reg.bias))

# Evaluate the model
mse = model.score(X_test, y_test)
print("Mean Squared Error:", mse)

# Save the trained model parameters
model.save('petal_predictor_parameters.npz')
print("Model parameters saved as petal_predictor_parameters.npz")
