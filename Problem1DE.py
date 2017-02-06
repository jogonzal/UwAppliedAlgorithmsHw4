import numpy as np
import random

def calculateObjectiveFunction(a, X, y, m, lambdaValue):
    totalObjectiveFunction = 0;
    for i in range(0, m):  # For all datapoints
        individualObjectiveFunction = np.dot(a.T, X[i]) - y[i];
        individualObjectiveFunction = 0.5 * individualObjectiveFunction * individualObjectiveFunction;
        totalObjectiveFunction += individualObjectiveFunction;
    if (lambdaValue > 0):
        moduloOfMatrix = np.linalg.norm(a);
        totalLambdaContribution = lambdaValue * moduloOfMatrix * moduloOfMatrix;
        return totalObjectiveFunction[0] + totalLambdaContribution;
    return totalObjectiveFunction[0];

def calculatePartialDerivative(currentA, X, y, m, j, lambdaValue):
    totalDerivative = 0;
    for i in range(0, m):  # For all datapoints
        individualError = np.dot(currentA.T, X[i]) - y[i];
        # TODO: Very unsure whether this goes here...
        if (lambdaValue > 0):
            currentAContribution = currentA[j];
            totalLambdaContribution = 2 * lambdaValue  * currentAContribution;
            individualError += totalLambdaContribution;
        totalDerivative += individualError * X[i][j];

    return totalDerivative;

def calculateNextEstimate(currentA, X, y, m, n, stepSize, lambdaValue):
    nextA = np.zeros((n, 1));
    for j in range(0, n):
        nextA[j] = currentA[j] - stepSize * calculatePartialDerivative(currentA, X, y, m, j, lambdaValue);
    return nextA;

def calculateSteps(initialA, X, y, m, n, stepSize, stepsToExecute, lambdaValue):
    stepsExecuted = [];
    stepsExecuted.append(calculateObjectiveFunction(initialA, X, y, m, lambdaValue));
    currentA = initialA;
    for s in range(0, stepsToExecute):
        currentA = calculateNextEstimate(currentA, X, y, m, n, stepSize, lambdaValue);
        newObjValue = calculateObjectiveFunction(currentA, X, y, m, lambdaValue);
        stepsExecuted.append(newObjValue);
    return [currentA, stepsExecuted];

train_m = 100
test_m = 1000
n = 100
X_train = np.random.normal(0,1, size=(train_m,n))
a_true = np.random.normal(0,1, size=(n,1))
y_train = X_train.dot(a_true) + 0.5*np.random.normal(0,1,size=(train_m,1))
X_test = np.random.normal(0,1, size=(test_m,n))
y_test = X_test.dot(a_true) + 0.5*np.random.normal(0,1,size=(test_m,1))

## 1.D

stepsToExecute = 20;
initialA = np.zeros((n, 1));
stepsExecuted = [];
stepSize = 0.0001;

# check performance against train data with 100
stepsExecuted = calculateSteps(initialA, X_train, y_train, train_m, n, stepSize, stepsToExecute, 0);
performanceAgainstTrainData = calculateObjectiveFunction(stepsExecuted[0], X_train, y_train, train_m, 0);
print "1.D Performance against training data with " + str(train_m) + " samples is " + str(performanceAgainstTrainData);
performanceAgainstRealData = calculateObjectiveFunction(stepsExecuted[0], X_test, y_test, test_m, 0);
print "1.D Performance against test data with " + str(train_m) + " samples is " + str(performanceAgainstRealData);

# check performance against train data with 20
train_m = 20;
stepsExecuted = calculateSteps(initialA, X_train, y_train, train_m, n, stepSize, stepsToExecute, 0);
performanceAgainstTrainData = calculateObjectiveFunction(stepsExecuted[0], X_train, y_train, train_m, 0);
print "1.D Performance against training data with " + str(train_m) + " samples is " + str(performanceAgainstTrainData);
performanceAgainstRealData = calculateObjectiveFunction(stepsExecuted[0], X_test, y_test, test_m, 0);
print "1.D Performance against test data with " + str(train_m) + " samples is " + str(performanceAgainstRealData);

## 1.E
lambdaValues = [100, 10, 1, 0.1, 0.01, 0.001];
train_m = 100; # Back to normal
lambdaValueResults = [];
for lambdaValue in lambdaValues:
    lambdaValueResults = calculateSteps(initialA, X_train, y_train, train_m, n, stepSize, stepsToExecute, lambdaValue);
    lambdaValuePerfOnTrain = calculateObjectiveFunction(lambdaValueResults[0], X_train, y_train, train_m, 0);
    lambdaValuePerfOnTest = calculateObjectiveFunction(lambdaValueResults[0], X_test, y_test, test_m, 0);
    print "1.E: Ran with lambda " + str(lambdaValue) + " and samples " + str(train_m) + " and scored " + str(lambdaValuePerfOnTrain) + " on train and " + str(lambdaValuePerfOnTest) + " on test ";
    lambdaValueResults.append([lambdaValue, lambdaValuePerfOnTrain, lambdaValuePerfOnTrain]);

