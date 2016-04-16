import pandas as pd

import sys
sys.path.append('../')
import cleaner
import feature_eng


if __name__ == '__main__':
    df_prod = pd.read_csv("product_descriptions.csv")
    df_prod[feature_eng.DESCRIPTION_CLEANED] = df_prod.fillna('').apply(
            cleaner.clean_description, axis=1
            )
    df_prod.to_csv("clean_product_descriptions.csv", encoding='ISO-8859-1')
