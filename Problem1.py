import numpy as np
from numpy.linalg import inv

def calculateObjectiveFunction(a, X, y):
    totalObjectiveFunction = 0;
    for i in range(0, m):  # For all datapoints
        individualObjectiveFunction = np.dot(a.T, X[i]) - y[i];
        individualObjectiveFunction = 0.5 * individualObjectiveFunction * individualObjectiveFunction;
        totalObjectiveFunction += individualObjectiveFunction;
    return totalObjectiveFunction;

n = 100 # dimensions of data
m = 1000 # number of data points
X = np.random.normal(0,1, size=(m,n))
a_true = np.random.normal(0,1, size=(n,1))
y = X.dot(a_true) + np.random.normal(0,0.1,size=(m,1))

## 1.A

# Calculate the most optimal a
closedFormA = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y);
closedFormAObjectiveValue = calculateObjectiveFunction(closedFormA, X, y);

print "Found value of closedFormA " + str(closedFormA)
print "Error with that a is " + str(closedFormAObjectiveValue);

## 1.B

