# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv', header=None)
data = data.to_numpy()

X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier()

# TODO: Fit the model
model.fit(X,y)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred) #sum(y_pred == y)/len(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state = 42,
                                                    test_size = 0.25)

for p1 in range(5,15,2):
    for p2 in range(2,10,2):
        for p3 in range(1,6,1):
            print(p1,p2,p3)
            model = DecisionTreeClassifier(max_depth=p1, min_samples_split=p2, min_samples_leaf=p3)
            model.fit(X_train, y_train)
            # Making predictions
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)
            # Calculate the accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy  = accuracy_score(y_test,  y_test_pred)
            print('The training accuracy is', train_accuracy)
            print('The test accuracy is', test_accuracy)


