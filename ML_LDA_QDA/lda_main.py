from utils import load_and_prepare_data
from models import LDAModel
import numpy as np

train_data_rgb, train_labels_rgb, test_data_rgb, test_labels_rgb = (
    load_and_prepare_data(False)
)
train_data_grey, train_labels_gray, test_data_gray, test_labels_gray = (
    load_and_prepare_data(True)
)
model = LDAModel()
model.fit(train_data_rgb, train_labels_rgb)
y_pred_rgb = model.predict(test_data_rgb)
print("Accuracy of RGB data is ", np.mean(y_pred_rgb == np.array(test_labels_rgb)))
model.fit(train_data_grey, train_labels_gray)
y_pred_grey = model.predict(test_data_gray)
print(
    "Accuracy of Gray Scale data is ",
    np.mean(y_pred_grey == np.array(test_labels_gray)),
)
