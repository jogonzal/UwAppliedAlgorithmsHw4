import numpy as np
from numpy.linalg import inv

n = 100 # dimensions of data
m = 1000 # number of data points
X = np.random.normal(0,1, size=(m,n))
a_true = np.random.normal(0,1, size=(n,1))
y = X.dot(a_true) + np.random.normal(0,0.1,size=(m,1))

# Calculate the most optimal a
a = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y);

# Now calculate the value of the objective function with a
totalObjectiveFunction = 0;
for i in range(0, m): # For all datapoints
    individualObjectiveFunction = np.dot(a.T, X[i]) - y[i];
    individualObjectiveFunction = 0.5 * individualObjectiveFunction * individualObjectiveFunction;
    totalObjectiveFunction += individualObjectiveFunction;

print "Found value of a " + str(a)
print "Error with that a is " + str(totalObjectiveFunction);