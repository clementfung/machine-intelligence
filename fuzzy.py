import numpy as np
import pandas as pd
import sys
import os
os.environ['DYLD_LIBRARY_PATH'] = '/Applications/MATLAB_R2015b/bin/maci64/app/MATLAB_R2015b.app/bin/maci64/'
from mlabwrap import mlab

class FuzzyInferenceSystem:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, x, y):
        Y = np.array(y)
        Y.shape = (len(y), 1)

        train = np.hstack([np.array(x),Y])
        fismat = mlab.genfis1(train)
        self.anfis = mlab.anfis(train,fismat)

    def predict(self, x):
        y = mlab.evalfis(np.array(x), self.anfis)
        # reshape 
        y.shape = len(y)
        return y

if __name__ == '__main__':
    df = pd.read_csv('data/train_sample_features.csv')
    fis = FuzzyInferenceSystem()
    x_cols = ['SearchAndTitleMatch', 'SearchAndDescriptionMatch']
    fis.fit(df[x_cols], df['relevance'])
    Y_pred =  fis.predict(df[x_cols])
    import pdb; pdb.set_trace()
