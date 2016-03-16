import nltk
import sys
import pandas as pd
import math

# Test the search terms, aggregate and find the frequency table of types

df_train = pd.read_csv("../data/train_sample.csv")
type_counts = {}
type_counts['matches'] = 0

for i in xrange(len(df_train)):
    query = df_train['search_term'][i]
    content = df_train['product_title'][i]

    if (query == None or content == None):
        continue

    if (isinstance(content, float) and math.isnan(content)):
        continue

    query_tags = nltk.pos_tag(nltk.word_tokenize(query))
    content_tags = nltk.pos_tag(nltk.word_tokenize(content))
    content_nouns = []

    for j in xrange(len(content_tags)):
        
        token_type = content_tags[j][1]
        
        if token_type.find("NN") == 0:
            content_nouns.append(content_tags[j][0])

    for j in xrange(len(query_tags)):
        
        token_type = query_tags[j][1]
        
        if token_type not in type_counts:
            type_counts[token_type] = 0

        type_counts[token_type] += 1

        if (query_tags[j][0] in content_nouns):
            type_counts['matches'] += 1
            print query


print type_counts