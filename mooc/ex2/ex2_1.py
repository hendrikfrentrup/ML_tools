#temp for iterative development
#%load_ext autoreload
#%autoreload 2


import numpy as np

# load data into matrices
data_file = 'ex2data1.txt'
data = np.loadtxt(data_file, delimiter=',')
X=data[:,0:2]
y=data[:,2] # dtype=int)



# import import matplotlib.pyplot as pl
# fig1 = pl.figure(1)

# plot data with purpose built function
from plotData import plotData
#plotData(X,y) #,fig1)
# # Labels and Legend
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')

# # Specified in plot order
# legend('Admitted', 'Not admitted')



#  setup the data matrix appropriately, and add ones for the intercept term
[m, n] = np.shape(X)

# Stack a columns of 1 as intercept term to X
# Optimisation note: It is faster to copy into matrix of ones than numpy's hstack function
#X = np.hstack( [np.ones([m, 1]), X] )
temp = np.copy(X)
X = np.ones([m,n+1])
X[:,1:] = temp
del temp

# Initialize fitting parameters
initial_theta = np.zeros( [n+1, 1] )

from sigmoid import sigmoid

# Compute and display initial cost and gradient
from costFunction import costFunction
[cost, grad] = costFunction(initial_theta, X, y);

print('Cost at initial theta (zeros): %f\n' % cost)
print('Gradient at initial theta (zeros)',grad)



from scipy.optimize import fmin_bfgs #minimize #fmin_ncg
from costFunction import fun_costFunction, jac_costFunction

res = fmin_bfgs( f=fun_costFunction, x0=initial_theta,args=(X,y),maxiter=400,fprime=jac_costFunction)
     
#options = {'maxiter':400}
#res = fmin( costFunction, x0=initial_theta, args=(X,y))#,
                #maxiter=500, full_output=True)
#                jac=jac_costFunction,
#                method='BFGS')#, options)

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost
#[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

# Print theta to screen
#print('Cost at theta found by fminunc: %f\n' % cost);
#print('theta:', res.x)#)theta)
