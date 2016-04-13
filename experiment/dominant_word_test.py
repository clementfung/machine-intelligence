import pandas as pd
import nltk
from nltk.corpus import stopwords

df = pd.read_csv('../data/train_sample.csv')

for i in xrange(len(df)):
    
    title = df['product_title'][i]
    tags = nltk.pos_tag(nltk.word_tokenize(title))
    dom_words = []

    for j in xrange(len(tags)):        
        if tags[j][0] in stopwords.words('english') and j > 0:
            dom_words.append(tags[j-1][0])

    dom_words.append(tags[-1][0])
            
    import pdb; pdb.set_trace()
