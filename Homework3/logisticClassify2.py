import numpy as np
import mltools as ml
import matplotlib.pyplot as plt


# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


def sigmoid(r):
    return 1 / (1 + np.exp(-r))


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array
        """
        self.classes = [0, 1]  # (default to 0/1; replace during training)
        self.theta = np.array([])  # placeholder value before training

        if len(args) or len(kwargs):  # if we were given optional arguments,
            self.train(*args, **kwargs)  # just pass them through to "train"

    ## METHODS ################################################################

    """ Plot the (linear) decision boundary of the classifier, along with data """

    def plotBoundary(self, X, Y):
        if len(self.theta) != 3:
            raise ValueError('Data & model must be 2D')

        ax = X.min(0), X.max(0)
        ax = (ax[0][0], ax[1][0], ax[0][1], ax[1][1])

        x1b = np.array([ax[0], ax[1]])  # at X1 = points in x1b
        ## ------ ADDED LINE: -------
        x2b = - (self.theta[0] + self.theta[1] * x1b) / self.theta[
            2]  # TODO find x2 values as a function of x1's values
        ## --------------------------

        A = (Y == self.classes[0])  # and plot it:
        plt.plot(X[A, 0], X[A, 1], 'b.', X[~A, 0], X[~A, 1], 'r.', x1b, x2b, 'k-')
        plt.axis(ax)
        plt.draw()

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0
        return P

    """ Return the predictied class of each data point in X"""

    def predict(self, X):
        ## TODO: compute linear response r[i] = theta0 + theta1 X[i,1] + theta2 X[i,2] + ... for each i
        r = np.zeros(X.shape[0])
        Yhat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            r[i] = self.theta[0] + self.theta[1] * X[i, 0] + self.theta[2] * X[i, 1]

            ## TODO: if z[i] > 0, predict class 1:  Yhat[i] = self.classes[1]
            if r[i] > 0:
                Yhat[i] = self.classes[1]
            ## else predict class 0:  Yhat[i] = self.classes[0]
            else:
                Yhat[i] = self.classes[0]
        return Yhat

    """ Train the logistic regression using stochastic gradient descent """

    def train(self, X, Y, init_step=1.0, stop_tol=1e-4, stop_epochs=5000,
              plot=True, alpha=0):
       """
          Train the logistic regression using stochastic gradient descent
       """
       M,N = X.shape;                 # initialize the model if necessary:
       self.classes = np.unique(Y);   # Y may have two classes, any values
       XX = np.hstack((np.ones((M,1)),X))
       # XX is X, but with an extra column of ones
       YY = ml.toIndex(Y,self.classes);
       # YY is Y, but with canonical values 0 or 1
       if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);

       # define sigmoid function
       def sigmoid(r):
            return 1/(1+np.exp(-r))

       # init loop variables:
       epoch=0; done=False; Jnll=[np.inf]; J01=[np.inf];
       while not done:
           stepsize, epoch = init_step*2.0/(2.0+epoch), epoch+1;
           # update stepsize
           # Do an SGD pass through the entire data set:
           for i in np.random.permutation(M):
               # TODO: compute linear response r(x)
               ri = np.dot(self.theta,XX[i,:])
               # TODO: compute gradient of NLL loss
               gradi = (sigmoid(ri)-YY[i])*XX[i,:]+2*alpha*self.theta;
               self.theta -= stepsize * gradi;  # take a gradient step

           J01.append(self.err(X,Y))  # evaluate the current error rate

           ## TODO: compute surrogate loss (logistic negative log-likelihood)
           ##  Jsur = sum_i [ (log si) if yi==1 else (log(1-si)) ]
           Jsur = [];
           for i in np.random.permutation(M):
               Jsur.append(-np.log(sigmoid(np.dot(self.theta, XX[i,:]))) \
               if YY[i]==1 else -np.log(1-sigmoid(np.dot(self.theta, XX[i,:]))))

           # TODO evaluate the current NLL loss
           Jnll.append(np.mean(Jsur)+alpha*np.dot(self.theta,self.theta))

           # TODO check stopping criteria: exit if exceeded # of epochs (>stopEpochs)
           # or if Jnll not changing between epochs (<stopTol)
           done = (epoch > stop_epochs) or np.abs(Jnll[-2] - Jnll[-1]) < stop_tol

       # plot losses
       if plot==True:
           print('Converged after %d iterations.  Misc surrogate loss = %.3f; error rate = %.3f' % (epoch, jnll[-1], j01[-1]))

           plt.figure(1); plt.plot(Jnll,'b-',label='surrogate loss');
           plt.plot(J01,'r-',label='error rate ');plt.xlabel("epoch");
           plt.title("Losses");
           plt.legend(); plt.draw();

           # & predictor if 2D
           if N==2:
               plt.figure(2); self.plotBoundary(X,Y);
               plt.title("Boundary"); plt.draw();
               plt.pause(.01);  # let OS draw the plot

       return (self.theta, epoch, Jnll[-1], J01[-1])

################################################################################
################################################################################
################################################################################
