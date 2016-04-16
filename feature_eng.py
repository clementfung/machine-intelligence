"""
Set of rule based features
"""
import abc
import os

import cleaner
import pandas as pd
import re
import nltk
import itertools

from util import flatten_to_list
from sklearn.externals import joblib
import cPickle as pickle


from sklearn.metrics.pairwise  import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import gensim
import numpy as np

# dataframe column constants
SEARCH = 'search_term'
SEARCH_CLEANED = 'search_term_cleaned'

TITLE = 'product_title'
TITLE_CLEANED = 'product_title_cleaned'
TITLE_NADJ = 'product_title_nadj'

DESCRIPTION = 'product_description'
DESCRIPTION_CLEANED = 'product_description_cleaned'
DESCRIPTION_NADJ = 'product_description_nadj'

DOMINANT_WORDS = 'dominant_words'

#################
### Util Functions
#################
def tokenize_string(string):
    """
    Clean and generate tokens (1-gram) from the string
    """
    return cleaner.tokenize_and_clean_str(string)

def map_product_uid_to_index(df_prods):
    prod_uid = df_prods['product_uid'].tolist()
    index    = df_prods.index.tolist()
    return {
            prod_uid[i] : index[i]
            for i in xrange(len(df_prods))
            }


def string_compare(str_a, str_b):
    """
    Comparison of two strings. The method is TODO
    Current uses boolean matching
    """
    a = set(str_a.split())
    b = set(str_b.split())
    return len(a.intersection(b))

def numbers_in_string(string, prefilter=None):
    if prefilter != None:
        string = ''.join(re.findall(r"[-+]?\d*\.\d+|\d+" + prefilter,
            string))
    A = re.findall(r"[-+]?\d*\.\d+|\d+",string)
    return [float(x) for x in A]


def get_cosine_similarity(row, corpus_index, vectorizer, X):
    row_num = corpus_index[int(row['product_uid'])]
    return cosine_similarity(X[row_num],vectorizer.transform([row[SEARCH_CLEANED]])).tolist()[0][0]

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


    def preprocess_from_df(self, df):
        """
        Override,
        in most cases will do nothing
        """
        return df
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
        search_term = row[SEARCH_CLEANED]
        return self.set_new_features((len(search_term.split())))

class NumOfCharsInSearchTerm(FeatureGenerator):
    feature_description = "Number of characters in the search term"
  
    def apply_rules(self, row):
        search_term = row[SEARCH_CLEANED]
        return self.set_new_features((len(search_term)))

class NumOfWordsInTitle(FeatureGenerator):
    feature_description = "Number of words in the product title"
  
    def apply_rules(self, row):
        product_title = row[TITLE]
        return self.set_new_features((len(product_title.split())))

class NumOfCharsInTitle(FeatureGenerator):
    feature_description = "Number of characters in the product title"
  
    def apply_rules(self, row):
        product_title = row[TITLE]
        return self.set_new_features((len(product_title)))

class NumOfWordsInProdDescrip(FeatureGenerator):
    feature_description = "Number of words in the product description"
  
    def apply_rules(self, row):
        prod_descrip = str(row[DESCRIPTION])
        return self.set_new_features((len(prod_descrip.split())))

class NumOfCharsInProdDescrip(FeatureGenerator):
    feature_description = "Number of characters in the product description"
  
    def apply_rules(self, row):
        prod_descrip = str(row[DESCRIPTION])
        return self.set_new_features((len(prod_descrip)))

class NumOfCharsInBrand(FeatureGenerator):
    feature_description = "Number of characters in the product brand"
  
    def apply_rules(self, row):
        brand = row['product_brand']
        if brand.lower().find('unbranded') != -1:
            return self.set_new_features(0)
        return self.set_new_features(len(brand))

## Search term matches
class SearchAndTitleMatch(FeatureGenerator):
    feature_description = 'Is the search term in the product title?'

    def apply_rules(self, row):
        search_term = row[SEARCH_CLEANED]
        prod_title  = row[TITLE_CLEANED]
        return self.set_new_features(string_compare(search_term, prod_title))

class SearchAndTitleNAdjMatch(FeatureGenerator):
    feature_description = 'Is the search term in the product title?'

    def apply_rules(self, row):
        search_term = row[SEARCH_CLEANED]
        prod_title  = row[DESCRIPTION_NADJ]
        return self.set_new_features(string_compare(search_term, prod_title))

class SearchAndDescriptionMatch(FeatureGenerator):
    feature_description = 'How does the search term match the product description?'

    def apply_rules(self, row):
        search_term = row[SEARCH_CLEANED]
        prod_de  = row[DESCRIPTION_CLEANED]
        return self.set_new_features(string_compare(search_term, prod_de))
    
class SearchAndDescriptionNAdjMatch(FeatureGenerator):
    feature_description = 'How does the search term match the product description, nouns and adjectives?'

    def apply_rules(self, row):
        search_term = row[SEARCH_CLEANED]
        prod_de  = row[DESCRIPTION_NADJ]
        return self.set_new_features(string_compare(search_term, prod_de))

class SearchAndProductBrandMatch(FeatureGenerator):
    feature_description = 'Does the search term have a product?'

    def apply_rules(self, row):
        brand = row['product_brand']
        return self.set_new_features((string_compare(brand, row[SEARCH_CLEANED])))

class SearchAndProductBulletsMatch(FeatureGenerator):
    feature_description = 'Is the search term in the products bullet points?'

    def apply_rules(self, row):
        BULLETS_KEY = 'bullet'.lower()
        attributes = eval(row['attributes'])
        matches_sum = 0
        for attr in attributes:
            if attr[0].lower().find(BULLETS_KEY) != -1:
                attr_tokens = attr[1]
                matches_sum += string_compare(attr_tokens, row[SEARCH_CLEANED])
        return self.set_new_features((matches_sum))

class SearchAndProductSizeMatch(FeatureGenerator):
    feature_description = 'If the search term contains size measurements do they match the product attributes?'

    def apply_rules(self, row):
        
        search_term = row[SEARCH_CLEANED]
        size_dimensions = row['size_dimensions']
        
        search_term_nums = numbers_in_string(search_term)
        measure_match = set(size_dimensions).intersection(search_term_nums)
        
        return self.set_new_features(len(measure_match))

class SearchAndProductWeightMatch(FeatureGenerator):
    feature_description = 'If the search term contains weight measurements do they match the product attributes?'

    def apply_rules(self, row):
        
        search_term = row[SEARCH_CLEANED]
        weight_dimensions = row['weight_dimensions']
        
        search_term_nums = numbers_in_string(search_term)
        measure_match = set(weight_dimensions).intersection(search_term_nums)
        
        return self.set_new_features(len(measure_match))

class SearchAndProductSizeInRange(FeatureGenerator):
    feature_description = 'If the search term contains size measurements, are they in range of the product attributes?'

    def apply_rules(self, row):
        search_term = row[SEARCH_CLEANED]
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
        search_term = row[SEARCH_CLEANED]
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
        product_title = row[TITLE_CLEANED].split()
        last_word = product_title[-1] if len(product_title) > 0 else ''
        search_term = row[SEARCH_CLEANED]
        return self.set_new_features((string_compare(last_word, search_term)))

class SearchAndProductLastWordNAdjMatch(FeatureGenerator):
    feature_description = "Matching last word in product title assuming that is the predomenent noun to the search term"

    def apply_rules(self, row):
        product_title = row[TITLE_NADJ].split()
        last_word = product_title[-1] if len(product_title) > 0 else ''
        search_term = row[SEARCH_CLEANED]
        return self.set_new_features(string_compare(last_word, search_term))

class SearchAndTitleDominantNadjMatch(FeatureGenerator):
    feature_description = "Matching search term to dominant word"

    def apply_rules(self, row):
        dominant_words = row[DOMINANT_WORDS]
        search_term = row[SEARCH_CLEANED]
        return self.set_new_features(string_compare(dominant_words, search_term))

## Ratios
class RatioOfDescripToSearch(FeatureGenerator):
    feature_description = "Number of words in description to number of words in search term"

    def apply_rules(self, row):
        num_words_search = len(row[SEARCH_CLEANED].split())
        num_words_descrip = len(str(row[DESCRIPTION_CLEANED]).split())
        if (num_words_search == 0):
            return self.set_new_features(0)
        return self.set_new_features((num_words_descrip/num_words_search))

class RatioOfTitleToSearch(FeatureGenerator):
    feature_descriptoin= "Number of words in product title to number of words in search term"

    def apply_rules(self, row):
        num_words_search = len(row[SEARCH_CLEANED].split())
        num_words_title = len(row[DESCRIPTION_CLEANED].split())
        if (num_words_search == 0):
            return self.set_new_features(0)    
        return self.set_new_features((num_words_title/num_words_search))


####
# Semantic Based features
# -- these are slow
####
class SearchDominantWord2VecSimilarity(FeatureGenerator):
    feature_description = 'Word2Vec similarity between search and dominanat'
    #TODO
    def __init__(
            self, 
            set_params=True,
            word2vec_model_path='word2vec/word2vec_title_model',
            *args, **kwargs):
        if set_params:
            self.model = gensim.models.Word2Vec.load(word2vec_model_path)

    def get_feature_name(self):
        base = self.__class__.__name__
        return [
                '%sMean' % base,
                '%sMin' % base,
                '%sMedian' % base,
                '%sMax' % base,
                '%sPhraseSimilarity' % base,
                ]

    def preprocess_from_df(self, df):
        if not SEARCH_CLEANED in df.columns.tolist():
            df[SEARCH_CLEANED] = df.apply(
                    cleaner.clean_search, axis=1
                    )
        if not TITLE_CLEANED in df.columns.tolist():
            df[DOMINANT_WORDS] = df.apply(
                    cleaner.reduce_to_dominant_words, axis=1
                    )


    def apply_rules(self, row):
        search_terms = row[SEARCH_CLEANED].strip().split(' ')
        dom_terms = cleaner.full_clean_string(row[DOMINANT_WORDS]) \
                        .strip().split(' ')
        similarities = []
        for word1, word2 in itertools.product(search_terms, dom_terms):
            try:
                score = self.model.similarity(word1, word2)
                if score < 1.0:
                    similarities.append(score)
            except KeyError:
                pass
                #similarities.append(0)
        mean = np.mean(similarities) if len(similarities) > 0 else 0.
        min_, med_, max_ = np.percentile(similarities, [0, 50, 100]) \
                if len(similarities) > 0 else (0., 0., 0.)
        search_filtered = [w for w in search_terms if w in self.model.vocab]
        dom_filtered    = [w for w in dom_terms if w in self.model.vocab]
        try:
            phrase_score = self.model.n_similarity(search_filtered, dom_filtered)

            if not isinstance(phrase_score, np.float64):
                phrase_score = 0.
        except TypeError:
            phrase_score = 0.
        return self.set_new_features(
                (mean, min_, med_, max_, phrase_score)
                )


class SearchTitleWord2VecSimilarity(FeatureGenerator):
    feature_description = 'Word2Vec similarity'
    #TODO
    def __init__(
            self, 
            set_params=True,
            word2vec_model_path='word2vec/word2vec_title_model',
            *args, **kwargs):
        if set_params:
            self.model = gensim.models.Word2Vec.load(word2vec_model_path)

    def get_feature_name(self):
        base = self.__class__.__name__
        return [
                '%sMean' % base,
                '%sMin' % base,
                '%sMedian' % base,
                '%sMax' % base,
                '%sPhraseSimilarity' % base,
                ]

    def preprocess_from_df(self, df):
        if not SEARCH_CLEANED in df.columns.tolist():
            df[SEARCH_CLEANED] = df.apply(
                    cleaner.clean_search, axis=1
                    )
        if not TITLE_CLEANED in df.columns.tolist():
            df[TITLE_CLEANED] = df.apply(
                    cleaner.clean_title, axis=1
                    )


    def apply_rules(self, row):
        search_terms = row[SEARCH_CLEANED].strip().split(' ')
        title_terms = row[TITLE_CLEANED].strip().split(' ')
        similarities = []
        for word1, word2 in itertools.product(search_terms, title_terms):
            try:
                score = self.model.similarity(word1, word2)
                if score < 1.0:
                    similarities.append(score)
            except KeyError:
                pass
                #similarities.append(0)
        mean = np.mean(similarities) if len(similarities) > 0 else 0.
        min_, med_, max_ = np.percentile(similarities, [0, 50, 100]) \
                if len(similarities) > 0 else (0., 0., 0.)
        search_filtered = [w for w in search_terms if w in self.model.vocab]
        title_filtered    = [w for w in title_terms if w in self.model.vocab]
        try:
            phrase_score = self.model.n_similarity(search_filtered, title_filtered) 
            if not isinstance(phrase_score, np.float64):
                phrase_score = 0.

        except TypeError:
            phrase_score = 0.
        return self.set_new_features(
                (mean, min_, med_, max_, phrase_score)
                )

class SearchTitleTfidfVectorizer(FeatureGenerator, SklearnGenerator):
    feature_description = 'Cosine similarity between search term and product title. Uses a count vectorizer'
    def __init__(self, set_params = True,corpus_csv='data/cleaned_product_descriptions.csv', *args, **kwargs):
        FeatureGenerator.__init__(self)
        SklearnGenerator.__init__(self, *args, **kwargs)
        self.corpus_csv = corpus_csv
        if set_params == True:
            if self.is_serialized():
                self.get_serialized()
            else:
                #TODO
                df_prods = pd.read_csv(self.corpus_csv)
                vect = TfidfVectorizer(min_df=1, stop_words=stopwords.words('english'))
                X_vect = vect.fit_transform(df_prods[TITLE_CLEANED])
                self.science['vect'] = vect
                self.science['X_vect'] = X_vect
                self.science['corpus'] = df_prods[['product_uid']]
                self.set_serialized()
            print 'Generating a corpus index'
            self.product_index = map_product_uid_to_index(self.science['corpus'])

    def preprocess_from_df(self, df):
        if not SEARCH_CLEANED in df.columns.tolist():
            df[SEARCH_CLEANED] = df.apply(
                    cleaner.clean_search, axis=1
                    )


    def apply_rules(self, row):
        return self.set_new_features(
                get_cosine_similarity(
                    row, 
                    self.product_index,
                    self.science['vect'], 
                    self.science['X_vect'],
                    )
                )


class SearchTitleCountVectorizer(FeatureGenerator, SklearnGenerator):
    feature_description = 'Cosine similarity between search term and product title. Uses a count vectorizer'
    def __init__(self, set_params = True,corpus_csv='data/cleaned_product_descriptions.csv', *args, **kwargs):
        FeatureGenerator.__init__(self)
        SklearnGenerator.__init__(self, *args, **kwargs)
        self.corpus_csv = corpus_csv
        if set_params == True:
            if self.is_serialized():
                self.get_serialized()
            else:
                #TODO
                df_prods = pd.read_csv(self.corpus_csv)
                vect = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))
                X_vect = vect.fit_transform(df_prods[TITLE_CLEANED])
                self.science['vect'] = vect
                self.science['X_vect'] = X_vect
                self.science['corpus'] = df_prods[['product_uid']]
                self.set_serialized()
            print 'Generating a corpus index'
            self.product_index = map_product_uid_to_index(self.science['corpus'])

    def preprocess_from_df(self, df):
        if not SEARCH_CLEANED in df.columns.tolist():
            df[SEARCH_CLEANED] = df.apply(
                    cleaner.clean_search, axis=1
                    )


    def apply_rules(self, row):
        return self.set_new_features(
                get_cosine_similarity(
                    row, 
                    self.product_index,
                    self.science['vect'], 
                    self.science['X_vect'],
                    )
                )

class SearchTitleCountTwoGramVectorizer(FeatureGenerator, SklearnGenerator):
    feature_description = 'Cosine similarity between search term and product title for 2 grams. Uses a count vectorizer'
    def __init__(self, set_params = True,corpus_csv='data/cleaned_product_descriptions.csv', *args, **kwargs):
        FeatureGenerator.__init__(self)
        SklearnGenerator.__init__(self, *args, **kwargs)
        self.corpus_csv = corpus_csv
        if set_params == True:
            if self.is_serialized():
                self.get_serialized()
            else:
                #TODO
                df_prods = pd.read_csv(self.corpus_csv)
                vect = CountVectorizer(min_df=1, stop_words=stopwords.words('english'), ngram_range=(2,2))
                X_vect = vect.fit_transform(df_prods[TITLE_CLEANED])
                self.science['vect'] = vect
                self.science['X_vect'] = X_vect
                self.science['corpus'] = df_prods[['product_uid']]
                self.set_serialized()
            print 'Generating a corpus index'
            self.product_index = map_product_uid_to_index(self.science['corpus'])

    def preprocess_from_df(self, df):
        if not SEARCH_CLEANED in df.columns.tolist():
            df[SEARCH_CLEANED] = df.apply(
                    cleaner.clean_search, axis=1
                    )


    def apply_rules(self, row):
        return self.set_new_features(
                get_cosine_similarity(
                    row, 
                    self.product_index,
                    self.science['vect'], 
                    self.science['X_vect'],
                    )
                )



class SearchDescriptionCountVectorizer(FeatureGenerator, SklearnGenerator):
    feature_description = 'Cosine similarity between search term and product description. Uses a count vectorizer'
    def __init__(self, set_params = True,corpus_csv='data/cleaned_product_descriptions.csv', *args, **kwargs):
        FeatureGenerator.__init__(self)
        SklearnGenerator.__init__(self, *args, **kwargs)
        self.corpus_csv = corpus_csv
        if set_params == True:
            if self.is_serialized():
                self.get_serialized()
            else:
                #TODO
                df_prods = pd.read_csv(self.corpus_csv)
                vect = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))
                X_vect = vect.fit_transform(df_prods[DESCRIPTION_CLEANED])
                self.science['vect'] = vect
                self.science['X_vect'] = X_vect
                self.science['corpus'] = df_prods[['product_uid']]
                self.set_serialized()
            print 'Generating a corpus index'
            self.product_index = map_product_uid_to_index(self.science['corpus'])

    def preprocess_from_df(self, df):
        if not SEARCH_CLEANED in df.columns.tolist():
            df[SEARCH_CLEANED] = df.apply(
                    cleaner.clean_search, axis=1
                    )


    def apply_rules(self, row):
        return self.set_new_features(
                get_cosine_similarity(
                    row, 
                    self.product_index,
                    self.science['vect'], 
                    self.science['X_vect'],
                    )
                )

class SearchDescriptionTfidfVectorizer(FeatureGenerator, SklearnGenerator):
    feature_description = 'Cosine similarity between search term and product description. Uses a tfidf vectorizer'
    def __init__(self, set_params=True,corpus_csv='data/cleaned_product_descriptions.csv', *args, **kwargs):
        FeatureGenerator.__init__(self)
        SklearnGenerator.__init__(self, *args, **kwargs)
        self.corpus_csv = corpus_csv
        if set_params: 
            if self.is_serialized():
                self.get_serialized()
            else:
                #TODO
                df_prods = pd.read_csv(self.corpus_csv)
                vect = TfidfVectorizer(min_df=1, stop_words=stopwords.words('english'))
                X_vect = vect.fit_transform(df_prods[DESCRIPTION_CLEANED])
                self.science['vect'] = vect
                self.science['X_vect'] = X_vect
                #self.science['corpus'] = df_prods
                self.science['corpus'] = df_prods[['product_uid']]
                self.set_serialized()
            print 'Generating a corpus index'
            self.product_index = map_product_uid_to_index(self.science['corpus'])

    def preprocess_from_df(self, df):
        if not SEARCH_CLEANED in df.columns.tolist():
            df[SEARCH_CLEANED] = df.apply(
                    cleaner.clean_search, axis=1
                    )


    def apply_rules(self, row):
        return self.set_new_features(
                get_cosine_similarity(
                    row, 
                    self.product_index,
                    self.science['vect'], 
                    self.science['X_vect'],
                    )
                )

######
# Using all the feature functions at once
#####
class FeatureFactory:
    def __init__(self, ignore_features = [], *args, **kwargs):
        # instantiate all the feature classes
        self.ignore_features = ignore_features
        self.feature_generators = map(lambda x: x(*args, **kwargs), self.feature_classes())

    def feature_classes(self):
        """
        Returns list of feature generator class
        The list will be anything that inherits
        from the base FeatureGenerator class
        """
        all_classes = [cls for cls in FeatureGenerator.__subclasses__()]
        keep_list = []
        for c in all_classes:
            if not c in self.ignore_features:
                keep_list.append(c)
        return keep_list 

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
            feat.preprocess_from_df(df)

            df[feat.get_feature_name()] = df.fillna('').apply(
                    feat.apply_rules, axis=1
                    )
        return df

    def preprocess_columns(self, df, verbose=False):
        """
        Create new derived columns for feature engineering
        """
        
        # Spellcheck, hardcore cleaning and porter stemming
        if verbose:
            print 'search clean'
        df[SEARCH_CLEANED] = df.fillna('').apply(
                    cleaner.clean_search, axis=1
                    )        

        if verbose:
            print 'title clean'
        df[TITLE_CLEANED] = df.fillna('').apply(
                    cleaner.clean_title, axis=1
                    )
        
        if verbose:
            print 'description clean'
        df[DESCRIPTION_CLEANED] = df.fillna('').apply(
                    cleaner.clean_description, axis=1
                    )
        
        if verbose:
            print 'title nadj'
        df[TITLE_NADJ] = df.fillna('').apply(
                    cleaner.reduce_title_nadj, axis=1
                    )
        
        if verbose:
            print 'description nadj'
        df[DESCRIPTION_NADJ] = df.fillna('').apply(
                    cleaner.reduce_description_nadj, axis=1
                    )

        if verbose: 
            print 'size dimensions'        
        df['size_dimensions'] = df.fillna('').apply(
                    cleaner.get_size, axis=1
                    )

        if verbose: 
            print 'weight dimensions'        
        df['weight_dimensions'] = df.fillna('').apply(
                    cleaner.get_weight, axis=1
                    )

        if verbose: 
            print 'brand'        
        df['product_brand'] = df.fillna('').apply(
                    cleaner.get_brand, axis=1
                    )

        if verbose:
            print 'Dominant Words'
        df[DOMINANT_WORDS] = df.fillna('').apply(
                    cleaner.reduce_to_dominant_words, axis=1
                    )

        print "FINISHED PRE-PROCESSING"
        return df

    def preprocess_columns_names(self):
        return [SEARCH_CLEANED, TITLE_CLEANED, DESCRIPTION_NADJ, TITLE_NADJ, DESCRIPTION_NADJ, 'product_brand', DOMINANT_WORDS]

if __name__ == '__main__':
    # This is how we can use this class.
    # Just create a factory object 
    # and let it do the rest of the heavy lifting
    ff = FeatureFactory(corpus_csv='data/product_descriptions.csv', pickle_path = 'pickles/')

    # show that it actually creates objects
    print ff.get_feature_names()
    print ff.get_feature_descriptions()

    #df = pd.read_csv('data/train_sample.csv', encoding='ISO-8859-1')
    df = pd.read_csv('data/test_joined.csv', encoding='ISO-8859-1')
    #df = pd.read_csv('data/train_joined.csv', encoding='ISO-8859-1')
    df = ff.preprocess_columns(df, verbose=True)
    df[ff.preprocess_columns_names()].to_csv('data/test_features_pp_full.csv', index=False, encoding='utf-8')
    df2 = ff.apply_feature_eng(df, verbose=True)
    # lets keep only computed features to reduce memory size
    cols = ff.get_feature_names() + ['id', 'product_uid']
    df2[cols].to_csv('data/test_features_full.csv', index=False, encoding='utf-8')
    print 'saving to csv...'
    #df2[cols].to_csv('data/train_features_v2.csv', index=False)
    #df2[cols].to_csv('data/train_sample_features.csv', index=False)
