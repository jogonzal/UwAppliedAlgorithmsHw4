import numpy as np
import random

def calculateObjectiveFunction(a, X, y, m):
    totalObjectiveFunction = 0;
    for i in range(0, m):  # For all datapoints
        individualObjectiveFunction = np.dot(a.T, X[i]) - y[i];
        individualObjectiveFunction = 0.5 * individualObjectiveFunction * individualObjectiveFunction;
        totalObjectiveFunction += individualObjectiveFunction;
    return totalObjectiveFunction[0];

def calculatePartialDerivative(currentA, X, y, m, j):
    totalDerivative = 0;
    for i in range(0, m):  # For all datapoints
        individualError = np.dot(currentA.T, X[i]) - y[i];
        totalDerivative += individualError * X[i][j];
    return totalDerivative;

def calculateNextEstimate(currentA, X, y, m, n, stepSize):
    nextA = np.zeros((n, 1));
    for j in range(0, n):
        nextA[j] = currentA[j] - stepSize * calculatePartialDerivative(currentA, X, y, m, j);
    return nextA;

def calculateSteps(initialA, X, y, m, n, stepSize, stepsToExecute):
    stepsExecuted = [];
    stepsExecuted.append(calculateObjectiveFunction(initialA, X, y, m));
    currentA = initialA;
    for s in range(0, stepsToExecute):
        currentA = calculateNextEstimate(currentA, X, y, m, n, stepSize);
        newObjValue = calculateObjectiveFunction(currentA, X, y, m);
        stepsExecuted.append(newObjValue);
    return stepsExecuted;

def calculateNextStochasticEstimate(currentA, X, y, m, n, stepSize, stepsExecuted):
    currentA = currentA.copy();
    smallStepSize = stepSize;
    r = list(range(1000))
    random.shuffle(r)
    for i in r:  # For all datapoints
        for j in range(0, n): # For all dimensions
            individualError = np.dot(currentA.T, X[i]) - y[i];
            totalDerivative = individualError * X[i][j];
            currentA[j] -= smallStepSize * totalDerivative;
            stepsExecuted.append(calculateObjectiveFunction(currentA, X, y, m));
    return currentA;

def calculateStepsStochastic(initialA, X, y, m, n, stepSize, stepsToExecute):
    stepsExecuted = [];
    stepsExecuted.append(calculateObjectiveFunction(initialA, X, y, m));
    currentA = initialA;
    for s in range(0, stepsToExecute):
        currentA = calculateNextStochasticEstimate(currentA, X, y, m, n, stepSize, stepsExecuted);
        #newObjValue = calculateObjectiveFunction(currentA, X, y, m);
        #stepsExecuted.append(newObjValue);
    return stepsExecuted;

n = 100 # dimensions of data
m = 1000 # number of data points
X = np.random.normal(0,1, size=(m,n))
a_true = np.random.normal(0,1, size=(n,1))
y = X.dot(a_true) + np.random.normal(0,0.1,size=(m,1))

## 1.A

# Calculate the most optimal a
closedFormA = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y);
closedFormAObjectiveValue = calculateObjectiveFunction(closedFormA, X, y, m);

print "1.A: Found value of closedFormA " + str(closedFormA)
print "1.A: Objective value with that 'closedFormA' is " + str(closedFormAObjectiveValue);

## 1.B
stepSizes = [0.0001, 0.001, 0.00125];
stepsToExecute = 20;
initialA = np.zeros((n, 1));
stepsExecuted = [];
for stepSize in stepSizes:
    stepsExecuted.append([stepSize, calculateSteps(initialA, X, y, m, n, stepSize, stepsToExecute)]);
print "1.B The performances obtained for GD is " + str(stepsExecuted);

## 1.C
stepSizes = [0.001, 0.01, 0.02];
stepsToExecute = 1;
initialA = np.zeros((n, 1));
stepsExecuted = [];
for stepSize in stepSizes:
    stepsExecuted.append([stepSize, calculateStepsStochastic(initialA, X, y, m, n, stepSize, stepsToExecute)]);
print "1.C The performance obtained for SGD is in file 'SGD.txt'";
f = open("SGD.txt", 'w')
f.write(str(stepsExecuted))  # python will convert \n to os.linesep
f.close()  # you can omit in most cases as the destructor will call it