import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

data = np.genfromtxt("/Users/brookeryan/Developer/CS273A Homework/data/curve80.txt",
                     delimiter=None)  # load the text file
X = data[:, 0]  # column 1, scalar feature x values
X = np.atleast_2d(X).T  # code expects shape (M,N) so make sure it’s 2-dimensional
Y = data[:, 1]  # column 2, doesn’t matter for Y

Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)  # split data set 75/25

#
# 1) Print the shapes of these four objects.
#
print("Xtr shape = {s}".format(s=Xtr.shape))
print("Xte shape = {s}".format(s=Xte.shape))
print("Ytr shape = {s}".format(s=Ytr.shape))
print("Yte shape = {s}".format(s=Yte.shape))

#
# 2.) Use the provided linearRegress class to create a linear regression predictor of y given x.
#
lr = ml.linear.linearRegress(Xtr, Ytr)  # create and train model
xs = np.linspace(0, 10, 200)  # densely sample 200 evenly-spaced x-values from 0 to 10
print(xs)
xs = xs[:, np.newaxis]  # force "xs" to be an Mx1 matrix (M data points with 1 feature)
# print(xs)
ys = lr.predict(xs)  # make predictions at xs
print(ys)
#
# (a) Plot the training data points along with your prediction function in a single plot. (10 points)
#
# TODO: is this right? or does the plot portion require multiplication by the theta points like in linear regression formula?
plt.plot(xs, ys)
plt.scatter(Xtr, Ytr)
#
# (b) Print the linear regression coefficients (lr.theta) and verify that they match your plot. (5 points)
# I think for this one what we want to do is
#
print("Linear regression coefficients = theta_0 = {t0}\n theta_1 = {t1}".format(t0=lr.theta[0,0], t1=lr.theta[0,1]))

#
# (c) What is the mean squared error of the predictions on the training and test data? (10 points)
#
mean_square_error = lr.mse(Xtr, Ytr)
print("Mean square error = {m}".format(m=mean_square_error))

plt.show()

#
# Sub-problem 3:
#
