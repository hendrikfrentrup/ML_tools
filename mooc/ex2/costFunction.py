def costFunction(theta, X, y):
    # costFunction Compute cost and gradient for logistic regression
    # J = costFunction(theta, X, y) computes the cost of using theta as the
    # parameter for logistic regression and the gradient of the cost
    # w.r.t. to the parameters.
    from numpy import zeros, log, dot, sum, size, transpose, shape
    from sigmoid import sigmoid
   
    # Initialize array dimensions
    m = shape(y)[0] # number of training examples

    J = 0.0
    grad = zeros(size(theta))

    J = sum( -y*log(sigmoid(transpose(dot(X,theta))[0]))
        - (1-y)*log(1-sigmoid(transpose(dot(X,theta))[0]))) / m

    grad = dot(sigmoid(transpose(dot(X,theta))[0])-y, X) / m
   
    return J , grad
   
def fun_costFunction(theta, X, y):
    # callable costFunction
    from numpy import zeros, ones, log, dot, sum, size, transpose, shape, where
    from sigmoid import sigmoid
   
    # Initialize array dimensions
    m = X.shape[0] # number of training examples

    J = 0.0
    h_theta=sigmoid(transpose(dot(X,theta))).flatten()
    h_theta=where(h_theta<1e-15, 1e-15*ones(m), h_theta)
    h_theta=where(h_theta>1-1e-15, 1-1e-15*ones(m), h_theta)
    J = -sum(y*log(h_theta) + (1-y)*log(1-h_theta))/m
   
    return J
   
def jac_costFunction(theta, X, y):
    # callable gradient of costFunction
    from numpy import zeros, log, dot, sum, size, transpose, shape
    from sigmoid import sigmoid
   
    # Initialize array dimensions
    m = shape(X)[0] # number of training examples

    grad = zeros(size(theta))

    grad = dot(sigmoid(transpose(dot(X,theta))[0])-y, X) / m
   
    return grad
