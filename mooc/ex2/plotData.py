def plotData(X,y):
    #plotData Plots the data points X and y (into a new figure)
    #   plotData(X,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    import matplotlib.pyplot as pl
    # Find indeces where y is True/1 or False/0
    pos = y==1
    neg = y==0

    # Plot (into new figure?)
    pl.plot( X[pos,0],X[pos,1],'g+',
             X[neg,0],X[neg,1],'ro')
    # block=False to prevent blocking while plot persists
    pl.show(block=False)
