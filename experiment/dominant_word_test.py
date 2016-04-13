import pandas as pd
import nltk
from nltk.corpus import stopwords

df = pd.read_csv('../data/train_sample.csv')
dom_words = []

for i in xrange(len(df)):
    title = df['product_title'][i]
    tags = nltk.pos_tag(nltk.word_tokenize(title))

    import pdb; pdb.set_trace()
