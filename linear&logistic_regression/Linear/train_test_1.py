import numpy as np
from LinearRegression import LinearRegression  # Assuming LinearRegression.py is in the same directory
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Define regression models
models = [
    ("SepalWidth", [0]),  # Predict sepal width using sepal length
    ("PetalLength", [3]),  # Predict petal length using petal width
    ("PetalWidth", [0, 1]),  # Predict petal width using sepal length and width
    ("PetalLengthFromSepal", [0, 1])  # Predict petal length using sepal length and width
]

# Train and evaluate regression models
for name, features in models:
    # Extract features for training and testing
    X_train_feat = X_train[:, features]
    X_test_feat = X_test[:, features]

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train_feat, y_train[:len(X_train_feat)])  # Ensure correct size of y_train

    # Make predictions
    y_pred = model.predict(X_test_feat)

    # Evaluate the model
    mse = model.score(X_test_feat, y_test)
    print(f"Model '{name}' Mean Squared Error: {mse:.4f}")

    # Save the trained model
    model.save(f"model_{name}.npz")
