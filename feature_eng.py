"""
Set of rule based features
"""
import abc
import os

import cleaner
import pandas as pd
import re
import nltk

from util import flatten_to_list
from sklearn.externals import joblib
import cPickle as pickle


from sklearn.metrics.pairwise  import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
#################
### Util Functions
#################
def tokenize_string(string):
    """
    Clean and generate tokens (1-gram) from the string
    """
    return cleaner.tokenize_and_clean_str(string, stem = False)

def stem_and_tokenize_string(string):
    """
    Clean and generate tokens (1-gram) from the string
    """
    return cleaner.tokenize_and_clean_str(string, stem = True)

def reduce_and_tokenize_string(string):
    """
    Clean, reduce to nouns and adjectives, 
    and generate tokens (1-gram) from the string
    """
    return cleaner.tokenize_and_clean_str(string, stem = False, reduce = True)

def string_compare(str_a, str_b):
    """
    Comparison of two strings. The method is TODO
    Current uses boolean matching
    """
    a = set(tokenize_string(str_a))
    b = set(tokenize_string(str_b))
    return len(a.intersection(b))

def noun_and_adjective_compare(str_a, str_b):
    """
    Only compare the nouns in the two strings
    """
    a = set(reduce_and_tokenize_string(str_a))
    b = set(reduce_and_tokenize_string(str_b))
    return len(a.intersection(b))

def numbers_in_string(string):
    A = re.findall(r"[-+]?\d*\.\d+|\d+",string)
    return [float(x) for x in A]


def get_cosine_similarity(row, df_corpus, vectorizer, X):
    row_num = df_corpus[df_corpus['product_uid'] == row['product_uid']].index
    return cosine_similarity(X[row_num],vectorizer.transform([row['search_term']])).tolist()[0][0]



#################
### Feature functions
#################
class FeatureGenerator:
    """
    All feature engineering classes
    can inherent from this. 
    Easy way to standardize formatting
    """
    __metaclass__ = abc.ABCMeta
    feature_description = ''

    def __init__(self, *args, **kwargs):
        pass

    def get_feature_name(self):
        return [self.__class__.__name__]

    def get_feature_description(self):
        return [self.feature_description] if type(self.feature_description) == str else self.feature_description


    @abc.abstractmethod
    def apply_rules(self, row):
        """
        Override this function. This extracts features from a 
        row of data
        """
        pass

    def set_new_features(self, row_vals):
        row_dict = dict()
        feat_names = self.get_feature_name()
        if not isinstance(row_vals, list) and not isinstance(row_vals, tuple):
            # force it to be a tuple
            row_vals = (row_vals, )
        assert len(row_vals) == len(feat_names)
        for i in xrange(len(row_vals)):
            row_dict[feat_names[i]] = row_vals[i]
        return pd.Series(row_dict)

class SklearnGenerator:
    """
    Uses something from sklearn. We can pickle it
    so that its faster in load time
    https://www.youtube.com/watch?v=yYey8ntlK_E
    """
    def __init__(self, pickle_path = '', *args, **kwargs):
        self.path = '%s/%s.pkl' % (pickle_path, self.__class__.__name__)
        self.science = dict()
    def is_serialized(self):
        return os.path.isfile(self.path)

    def get_serialized(self):
        print "Loading science data for ", self.__class__.__name__, " from ", self.path
        m_data = pickle.load(open(self.path, 'rb'))
        self.science = m_data

    def set_serialized(self):
        print "Saving science data for ", self.__class__.__name__, " to ", self.path
        with open(self.path, 'wb') as f:
            pickle.dump(self.science, f)
######
## Feature ENG
#####


## Word and character counts
class NumOfWordsInSearchTerm(FeatureGenerator):
    feature_description = "Number of words in the search term"
    
  
    def apply_rules(self, row):
        search_term = row['search_term']
        return self.set_new_features((len(search_term.split())))

class NumOfCharsInSearchTerm(FeatureGenerator):
    feature_description = "Number of characters in the search term"
  
    def apply_rules(self, row):
        search_term = row['search_term']
        return self.set_new_features((len(search_term)))

class NumOfWordsInTitle(FeatureGenerator):
    feature_description = "Number of words in the product title"
  
    def apply_rules(self, row):
        product_title = row['product_title']
        return self.set_new_features((len(product_title.split())))

class NumOfCharsInTitle(FeatureGenerator):
    feature_description = "Number of characters in the product title"
  
    def apply_rules(self, row):
        product_title = row['product_title']
        return self.set_new_features((len(product_title)))

class NumOfWordsInProdDescrip(FeatureGenerator):
    feature_description = "Number of words in the product description"
  
    def apply_rules(self, row):
        prod_descrip = str(row['product_description'])
        return self.set_new_features((len(prod_descrip.split())))

class NumOfCharsInProdDescrip(FeatureGenerator):
    feature_description = "Number of characters in the product description"
  
    def apply_rules(self, row):
        prod_descrip = str(row['product_description'])
        return self.set_new_features((len(prod_descrip)))

class NumOfCharsInBrand(FeatureGenerator):
    feature_description = "Number of characters in the product brand"
  
    def apply_rules(self, row):
        BRAND_KEY = 'brand'.lower()
        attributes = eval(row['attributes'])
        for attr in attributes:
            if attr[0].lower().find(BRAND_KEY) != -1:
                attr_tokens = attr[1]
                return self.set_new_features((len(attr_tokens)))
        return self.set_new_features((0))

## Search term matches
class SearchAndTitleMatch(FeatureGenerator):
    feature_description = 'Is the search term in the product title?'

    def apply_rules(self, row):
        search_term = row['search_term']
        prod_title  = row['product_title']
        return self.set_new_features((string_compare(search_term, prod_title)))

class SearchAndTitleNAdjMatch(FeatureGenerator):
    feature_description = 'Is the search term in the product title?'

    def apply_rules(self, row):
        search_term = row['search_term']
        prod_title  = row['product_title']
        return self.set_new_features((noun_and_adjective_compare(search_term, prod_title)))

class SearchAndDescriptionMatch(FeatureGenerator):
    feature_description = 'How does the search term match the product description?'

    def apply_rules(self, row):
        search_term = row['search_term']
        prod_de  = row['product_description']
        return self.set_new_features((string_compare(search_term, prod_de)))
    
class SearchAndDescriptionNAdjMatch(FeatureGenerator):
    feature_description = 'How does the search term match the product description, nouns and adjectives?'

    def apply_rules(self, row):
        search_term = row['search_term']
        prod_de  = row['product_description']
        return self.set_new_features((noun_and_adjective_compare(search_term, prod_de)))

class SearchAndProductBrandMatch(FeatureGenerator):
    feature_description = 'Does the search term have a product?'

    def apply_rules(self, row):
        BRAND_KEY = 'brand'.lower()
        attributes = eval(row['attributes'])
        for attr in attributes:
            if attr[0].lower().find(BRAND_KEY) != -1:
                attr_tokens = attr[1]
                return self.set_new_features((string_compare(attr_tokens, row['search_term'])))
        return self.set_new_features((0))

class SearchAndProductBulletsMatch(FeatureGenerator):
    feature_description = 'Is the search term in the products bullet points?'

    def apply_rules(self, row):
        BULLETS_KEY = 'bullet'.lower()
        attributes = eval(row['attributes'])
        matches_sum = 0
        for attr in attributes:
            if attr[0].lower().find(BULLETS_KEY) != -1:
                attr_tokens = attr[1]
                matches_sum += string_compare(attr_tokens, row['search_term'])
        return self.set_new_features((matches_sum))

class SearchAndProductSizeMatch(FeatureGenerator):
    feature_description = 'If the search term contains size measurements do they match the product attributes?'

    def apply_rules(self, row):
        search_term = row['search_term']
        measure_match = False
        search_term_nums = numbers_in_string(search_term)
        if len(search_term_nums)>0:
          SIZE_KEY = '(in.)'
          attributes = eval(row['attributes'])
          for attr in attributes:
              if attr[0].lower().find(SIZE_KEY) != -1:
                  attr_tokens = attr[1]
                  measure_match = (attr_tokens in search_term_nums)
        return self.set_new_features((measure_match))

class SearchAndProductWeightMatch(FeatureGenerator):
    feature_description = 'If the search term contains weight measurements do they match the product attributes?'

    def apply_rules(self, row):
        search_term = row['search_term']
        measure_match = False
        search_term_nums = numbers_in_string(search_term)
        if len(search_term_nums)>0:
          SIZE_KEY = '(lb.)'
          attributes = eval(row['attributes'])
          for attr in attributes:
              if attr[0].lower().find(SIZE_KEY) != -1:
                  attr_tokens = attr[1]
                  measure_match = (attr_tokens in search_term_nums)
        return self.set_new_features((measure_match))

class SearchAndProductSizeInRange(FeatureGenerator):
    feature_description = 'If the search term contains size measurements, are they in range of the product attributes?'

    def apply_rules(self, row):
        search_term = row['search_term']
        measure_in_range = False
        search_term_nums = numbers_in_string(search_term)
        if len(search_term_nums)>0:
          SIZE_KEY = '(in.)'
          attributes = eval(row['attributes'])
          for attr in attributes:
              nums_in_attr = numbers_in_string(attr[1])
              if attr[0].lower().find(SIZE_KEY) != -1 and len(nums_in_attr)>0:
                  attr_tokens = nums_in_attr[0]
                  attr_tokens = float(attr_tokens)
                  if attr_tokens !=0:
                    is_15percent_off = [abs(s-attr_tokens)/attr_tokens<=0.15 for s in search_term_nums]
                    measure_in_range = sum(is_15percent_off)>0
        return self.set_new_features((measure_in_range))

class SearchAndProductWeightInRange(FeatureGenerator):
    feature_description = 'If the search term contains weight measurements, are they in range of the product attributes?'

    def apply_rules(self, row):
        search_term = row['search_term']
        measure_in_range = False
        search_term_nums = numbers_in_string(search_term)
        if len(search_term_nums)>0:
          SIZE_KEY = '(lb.)'
          attributes = eval(row['attributes'])
          for attr in attributes:
              nums_in_attr = numbers_in_string(attr[1])
              if attr[0].lower().find(SIZE_KEY) != -1 and len(nums_in_attr)>0:
                  attr_tokens = nums_in_attr[0]
                  attr_tokens = float(attr_tokens)
                  if attr_tokens !=0:
                    is_15percent_off = [abs(l-attr_tokens)/attr_tokens<=0.15 for l in search_term_nums]
                    measure_in_range = sum(is_15percent_off)>0
        return self.set_new_features((measure_in_range))

class SearchAndProductLastWordMatch(FeatureGenerator):
    feature_description = "Matching last word in product title assuming that is the predomenent noun to the search term"

    def apply_rules(self, row):
        product_title = row['product_title']
        last_word = product_title.split()[-1]
        search_term = row['search_term']
        return self.set_new_features((string_compare(last_word, search_term)))

class SearchAndProductLastWordNAdjMatch(FeatureGenerator):
    feature_description = "Matching last word in product title assuming that is the predomenent noun to the search term"

    def apply_rules(self, row):
        product_title = row['product_title']
        last_word = product_title.split()[-1]
        search_term = row['search_term']
        return self.set_new_features((noun_and_adjective_compare(last_word, search_term)))

class SearchAndTitleDominantNadjMatch(FeatureGenerator):
    feature_description = "Matching search term to dominant word"

    def apply_rules(self, row):
        dominant_words = row['dominant_words']
        search_term = row['search_term']
        return self.set_new_features(string_compare(dominant_words, search_term))

## Ratios
class RatioOfDescripToSearch(FeatureGenerator):
    feature_description = "Number of words in description to number of words in search term"

    def apply_rules(self, row):
        num_words_search = len(row['search_term'].split())
        num_words_descrip = len(str(row['product_description']).split())
        return self.set_new_features((num_words_descrip/num_words_search))

class RatioOfTitleToSearch(FeatureGenerator):
    feature_descriptoin= "Number of words in product title to number of words in search term"

    def apply_rules(self, row):
        num_words_search = len(row['search_term'].split())
        num_words_title = len(row['product_title'].split())
        return self.set_new_features((num_words_title/num_words_search))


####
# Semantic Based features
# -- these are slow on first loadup
####
class SearchDescriptionCountVectorizer(FeatureGenerator, SklearnGenerator):
    featur_description = 'Cosine similarity between search term and product description. Uses a count vectorizer'
    def __init__(self, corpus_csv='data/product_descriptions.csv', *args, **kwargs):
        FeatureGenerator.__init__(self)
        SklearnGenerator.__init__(self, *args, **kwargs)
        self.corpus_csv = corpus_csv

        if self.is_serialized():
            self.get_serialized()
        else:
            #TODO
            df_prods = pd.read_csv(self.corpus_csv)
            vect = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))
            X_vect = vect.fit_transform(df_prods['product_description'])
            self.science['vect'] = vect
            self.science['X_vect'] = X_vect
            self.science['corpus'] = df_prods
            self.set_serialized()

    def apply_rules(self, row):
        return self.set_new_features(
                get_cosine_similarity(
                    row, 
                    self.science['corpus'], 
                    self.science['vect'], 
                    self.science['X_vect'],
                    )
                )

class SearchDescriptionTfidfVectorizer(FeatureGenerator, SklearnGenerator):
    featur_description = 'Cosine similarity between search term and product description. Uses a tfidf vectorizer'
    def __init__(self, corpus_csv='data/product_descriptions.csv', *args, **kwargs):
        FeatureGenerator.__init__(self)
        SklearnGenerator.__init__(self, *args, **kwargs)
        self.corpus_csv = corpus_csv

        if self.is_serialized():
            self.get_serialized()
        else:
            #TODO
            df_prods = pd.read_csv(self.corpus_csv)
            vect = TfidfVectorizer(min_df=1, stop_words=stopwords.words('english'))
            X_vect = vect.fit_transform(df_prods['product_description'])
            self.science['vect'] = vect
            self.science['X_vect'] = X_vect
            self.science['corpus'] = df_prods
            self.set_serialized()

    def apply_rules(self, row):
        return self.set_new_features(
                get_cosine_similarity(
                    row, 
                    self.science['corpus'], 
                    self.science['vect'], 
                    self.science['X_vect'],
                    )
                )

######
# Using all the feature functions at once
#####
class FeatureFactory:
    def __init__(self, *args, **kwargs):
        # instantiate all the feature classes
        self.feature_generators = map(lambda x: x(*args, **kwargs), self.feature_classes())

    def feature_classes(self):
        """
        Returns list of feature generator class
        The list will be anything that inherits
        from the base FeatureGenerator class
        """
        return [cls for cls in FeatureGenerator.__subclasses__()]

    def get_feature_names(self):
        """
        Return a list of the features names. Same one used in each column
        """
        return flatten_to_list(map(lambda x: x.get_feature_name(), self.feature_generators))

    def get_feature_descriptions(self):
        """
        Return a list of the features descriptions. 
        """
        return flatten_to_list(map(lambda x: (x.get_feature_name(), x.get_feature_description()),
					self.feature_generators
					))

    def get_feature_descriptions_map(self):
        return { pair[0]: pair[1]
                for pair in map(lambda x: (x.get_feature_name(), x.get_feature_description()),
                    self.feature_generators
					)
                }

    def apply_feature_eng(self, df, verbose=False):
        """
        Generates a new set of features
        to the data frame passed in
        """
        for feat in self.feature_generators:
            if verbose:
                print "Computing feature ", feat.get_feature_name()

            df[feat.get_feature_name()] = df.fillna('').apply(
                    feat.apply_rules, axis=1
                    )
        return df

    def preprocess_columns(self, df, verbose=False):
        """
        Create new derived columns for feature engineering
        """
        # Spellcheck AND hardcore cleaning
        df['search_term_cleaned'] = df.fillna('').apply(
                    cleaner.hardcore_spell_check, axis=1
                    )

        df['product_title_nadj'] = df.fillna('').apply(
                    cleaner.reduce_title, axis=1
                    )

        df['product_description_nadj'] = df.fillna('').apply(
                    cleaner.reduce_description, axis=1
                    )

        df['dominant_words'] = df.fillna('').apply(
                    cleaner.reduce_to_dominant_words, axis=1
                    )

        print "FINISHED PRE-PROCESSING"
        return df

if __name__ == '__main__':
    # This is how we can use this class.
    # Just create a factory object 
    # and let it do the rest of the heavy lifting
    ff = FeatureFactory()

    # show that it actually creates objects
    print ff.get_feature_names()
    print ff.get_feature_descriptions()
    df = pd.read_csv('data/train_sample.csv', encoding='ISO-8859-1')
    df = ff.preprocess_columns(df)
    df.to_csv('features_pp.out')

    df2 = ff.apply_feature_eng(df, verbose=True)
    df2.to_csv('features.out')
