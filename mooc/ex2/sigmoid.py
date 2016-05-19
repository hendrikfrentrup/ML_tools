def sigmoid(z):
    #sigmoid Compute sigmoid function
    #   J = sigmoid(z) computes the sigmoid of z.
    from numpy import array, exp
    return 1. / (1. + exp(-array(z)))
