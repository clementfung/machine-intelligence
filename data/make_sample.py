"""
Generate a small sample of training and test data.
"""
import random

import pandas as pd

if __name__ == "__main__":
    df_train = pd.read_csv("train.csv")
    df_desc = pd.read_csv("product_descriptions.csv")
    df_attr = pd.read_csv("attributes.csv")
    og_cols = df_train.columns.values.tolist()
    description_cols = df_desc.columns.values.tolist()
    ## this is a small sample, just so we can get the algorithms working
    n = 100
    #df = df_train.iloc[random.sample(df_train.index,n)].reset_index()
    df = df_train
    df = df.join(df_attr, on = "product_uid", rsuffix='_attr')
    df = df.join(df_desc, on = "product_uid", rsuffix='_desc')
    df.to_csv("train_sample.csv")

    df_new = pd.DataFrame() 

    for g, df_g in df.groupby("product_uid"):
        print "Grouping attributes for ", g
        names = df_g["name"].tolist()
        values = df_g["value"].tolist()
        attributes = [(names[i], values[i]) for i in xrange(len(df_g))]
        df_row = df_g[og_cols + description_cols]
        df_row["attributes"] = str(attributes)
        df_new = pd.concat([df_new, df_row])
    df_new.to_csv("train_sample.csv")



    print "SUCCESS"

