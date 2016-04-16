import pandas as pd

import sys
sys.path.append('../')
import feature_eng
import cleaner

def add_to_index(m_index, df):
    titles = df['product_title'].tolist()
    prod_uids = df['product_uid'].tolist()
    for i in xrange(len(df)):
        title = titles[i]
        prod_uid = prod_uids[i]
        if not m_index.has_key(prod_uid):
            m_index[prod_uid] = title
    return m_index


if __name__ == '__main__':
    df_train = pd.read_csv('train.csv', encoding='ISO-8859-1')
    df_test = pd.read_csv('test.csv', encoding='ISO-8859-1')
    df_prod = pd.read_csv(
            'clean_product_descriptions.csv',
            encoding='ISO-8859-1'
            )
    prod_index = {}
    prod_index = add_to_index(prod_index, df_train)
    prod_index = add_to_index(prod_index, df_test)
    df_prod[feature_eng.TITLE_CLEANED] = df_prod['product_uid'].apply(
            lambda prod_uid: cleaner.full_clean_string(prod_index[prod_uid])
            )
    df_prod.to_csv("clean_product_descriptions.csv", encoding='ISO-8859-1')


