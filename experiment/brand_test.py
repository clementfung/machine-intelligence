import pandas as pd

df = pd.read_csv('../data/train_sample.csv')
brands = []

for i in xrange(len(df)):
    row = eval(df['attributes'][i])

    brand_found = False
    for j in xrange(len(row)):    
        if (row[j][0] == "MFG Brand Name"):
            brand_found = True
            brand = row[j][1]
            brands.append(brand)

    if brand_found == False:
        print row

import pdb; pdb.set_trace()