import gensim, logging
import pandas as pd

import sys
sys.path.append('../')
import feature_eng


class SentanceGenerator(object):
    def __init__(self, df, col):
        self.sentances = df[col].tolist()
    def __iter__(self):
        for sen in self.sentances:
            yield sen.split(' ')

if __name__ == '__main__':
    df_prod = pd.read_csv('../data/clean_product_descriptions.csv')
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(
            SentanceGenerator(df_prod, feature_eng.TITLE_CLEANED),
            min_count=1)
    model.save('word2vec_title_model')
    import pdb; pdb.set_trace()

