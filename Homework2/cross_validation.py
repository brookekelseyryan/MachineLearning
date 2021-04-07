nFolds = 5
for iFold in range(nFolds):
  Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold)  # use ith block as validation
  learner = ml.linear.linearRegress(...                     # TODO: train on Xti, Yti, the data for this fold
  J[iFold] = ...                                            # TODO: now compute the MSE on Xvi, Yvi and save it
# the overall estimated validation error is the average of the error on each fold
print np.mean(J)