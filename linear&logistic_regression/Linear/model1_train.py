import numpy as np
from LinearRegression import LinearRegression  # Assuming LinearRegression.py is in the same directory
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


combinations = [
        ((0,), (1,)),  
        ((1,), (2,)),    
        ((3,), (1,)),  
        ((1,), (0,))   
    ]
models =[]
for i,o in combinations:
    X_train_comb = X_train_sc[:, i]
    Y_train_comb = X_train[:, o]

    model = LinearRegression()
    model.fit(X_train_comb, Y_train_comb)
    models.append(model)

    # Plot loss history for each model
    plt.plot(model.loss_step)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss vs Step')
    plt.grid(True)
    plt.show()
