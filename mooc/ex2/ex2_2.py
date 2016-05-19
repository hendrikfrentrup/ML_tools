import numpy as np
import matplotlib.pyplot as pl

def loadData(data_file):
    # load data into matrices
    data = np.loadtxt(data_file, delimiter=',')
    X=data[:,0:2]
    y=data[:,2] # dtype=int)
    return X, y

def plotData(X,y):
    #plotData Plots the data points X and y (into a new figure)
    #   plotData(X,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # Find indeces where y is True/1 or False/0
    pos = y==1
    neg = y==0

    # Plot (into new figure?)
    pl.plot( X[pos,0],X[pos,1],'g+', X[neg,0],X[neg,1],'ro')
    # block=False to prevent blocking while plot persists
    pl.show(block=False)

def sigmoid(z):
    #sigmoid Compute sigmoid function
    #   J = sigmoid(z) computes the sigmoid of z.
    return 1. / (1. + np.exp(-np.array(z)))

def costFunction(theta, X, y):
    # costFunction Compute cost and gradient for logistic regression
    # J = costFunction(theta, X, y) computes the cost of using theta as the
    # parameter for logistic regression and the gradient of the cost
    # w.r.t. to the parameters.

    # Initialize array dimensions
    m = y.shape[0] # number of training examples

    J = 0.0
    grad = np.zeros(np.size(theta))

    h_theta=sigmoid(np.transpose(np.dot(X,theta))).flatten()
    grad = np.dot(h_theta-y, X) / m

    h_theta=np.where(h_theta<1e-15, 1e-15*np.ones(m), h_theta)
    h_theta=np.where(h_theta>1-1e-15, 1-1e-15*np.ones(m), h_theta)
    J = -sum(y*np.log(h_theta) + (1-y)*np.log(1-h_theta))/m

    return J# , grad


def plotDecisionBoundary(theta, X, y):

    # Find indeces where y is True/1 or False/0
    pos = y==1
    neg = y==0

    # Plot (into new figure?)
    pl.plot( X[pos,1],X[pos,2],'g+', X[neg,1],X[neg,2],'ro')

    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,2])+2])

    # Calculate the decision boundary line
    plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

    # Plot
    pl.plot(plot_x, plot_y, 'b-')
    pl.show(block=False)       

def plotDecisionCompare(theta, X, y):

    # Find indeces where y is True/1 or False/0
    pos = y==1
    neg = y==0

    # Plot (into new figure?)
    pl.plot( X[pos,1],X[pos,2],'g+', X[neg,1],X[neg,2],'ro')

    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,2])+2])

    # Calculate the decision boundary line
    plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

    # Plot decision boundary
    pl.plot(plot_x, plot_y, 'b--')
   
    predictions = np.where(sigmoid(np.dot(X,theta))>0.5, 1.0, 0.0)
    wrong = y!=predictions
    # Plot wrong predictions
    pl.plot( X[wrong,1],X[wrong,2],'rx', lw=3)
    pl.show(block=False)

def prediction(theta, x):
    return sigmoid(np.dot(x,theta))

def calcAccuracy(theta, X, y):
    return np.mean(np.where(sigmoid(np.dot(X,theta))>0.5, 1.0, 0.0)==y)

X, y = loadData(data_file = 'ex2data1.txt')
#plotData(X,y)

# Stack a columns of 1 as intercept term to X
[m, n] = np.shape(X)
# Optimisation note: It is faster to copy into matrix of ones than numpy's hstack function
#X = np.hstack( [np.ones([m, 1]), X] )
temp = np.copy(X)
X = np.ones([m,n+1])
X[:,1:] = temp
del temp

# Initialize fitting parameters
initial_theta = np.zeros( [n+1, 1] )

cost = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): %f\n' % cost)
#print('Gradient at initial theta (zeros)',grad)

from scipy.optimize import minimize
res = minimize(costFunction, initial_theta, (X,y), method="BFGS")
plotDecisionCompare(res.x, X, y)
print 'chance of getting in with 45 & 85 score: ', prediction(res.x, [1, 45, 85])
print 'average prediction accuracy: ', calcAccuracy(res.x, X, y)

def test_sigmoid():
    for t in [ 0.0, 10, 1e6, -1e6, [1,2,3], [[-1,-2,-3],[1,2,-50]], np.eye(5) ]:
        print t ,' : ', sigmoid(t)

def test_costFunction():
    pass
 
def run_tests():
    test_sigmoid
