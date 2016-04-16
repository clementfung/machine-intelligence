"""
Generate a small sample of training and test data.
"""
import pandas as pd
import random

import sys
sys.path.append('../')

import util

if __name__ == "__main__":
    df_train = pd.read_csv("train.csv")
    ## this is a small sample, just so we can get the algorithms working
    
    n = 1000
    #df = df_train.iloc[df_train.index].reset_index()
    df = df_train.iloc[random.sample(df_train.index,n)].reset_index()
    
    #df = df.join(df_attr, on = "product_uid", rsuffix='_attr')
    #df = df.join(df_desc, on = "product_uid", rsuffix='_desc')
    df_new = util.join_raw(df,
            desc_path="product_descriptions.csv", 
            attr_path="attributes.csv",
            )

    df_new.to_csv("train_sample.csv", index=False)

    print "SUCCESS"

