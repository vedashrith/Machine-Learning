import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
iris = load_iris()

#X = iris.data[:2,]  # Use only sepal length and width
#y = iris.data[2:]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_test_sc = scaler.fit_transform(X_test)  # Scale the test features


X_test_m1 = X_test_sc[:, (3,)]  # Select the features for model 1
y_test_m1 = X_test_sc[:, (0,)]  

model = LinearRegression()
model.load('Reg_model1_parameters.npz')

y_pred = model.predict(X_test_m1)
mse = model.score(X_test_m1, y_test_m1)

# Print the mean squared error
print("Mean Squared Error for model 1:", mse)

# Calculate R^2
'''r2 = r2_score(y_test_m1, y_pred)
print("R^2 Score for model 1:", r2)
'''

plt.scatter(X_test_m1, y_test_m1, color='black')
plt.plot(X_test_m1, y_pred, color='blue', linewidth=3)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Linear Regression: Sepal Length vs Petal Width')
plt.show()