import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the Iris dataset
iris = load_iris()

X = iris.data[:, (2,3)]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train, X_val, y_val)

train_accuracy = model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
plot_decision_regions(X_train, y_train, clf=model)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Decision Regions for sepal Length/Width')
plt.savefig('log2.png')
plt.show()
