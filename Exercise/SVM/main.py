# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
# model = SVC()
model = SVC(kernel='rbf', gamma=30)

# TODO: Fit the model.
model.fit(X,y)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)




x1,  y1  = X[y==0][:,0],      X[y==0][:,1]
x2,  y2  = X[y==1][:,0],      X[y==1][:,1]
x1p, y1p = X[y_pred==0][:,0], X[y_pred==0][:,1]
x2p, y2p = X[y_pred==1][:,0], X[y_pred==1][:,1]

x1f, y1f = X[y_pred!=y][:,0], X[y_pred!=y][:,1]

import matplotlib.pyplot as plt

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x1f, y1f, marker='x', color='tab:red')
plt.show()


# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, axs = plt.subplots(1, 2)
axs[0].scatter(x1, y1)
axs[0].scatter(x2, y2)

axs[1].scatter(x1p, y1p)
axs[1].scatter(x2p, y2p)
fig.suptitle('SVM')
plt.show()
