"""
Set of rule based features
"""
import abc

import cleaner
import pandas as pd
import re


#################
### Util Functions
#################
def tokenize_string(string):
    """
    Clean and generate tokens (1-gram) from the string
    """
    string = str(string)
    return cleaner.tokenize_and_clean_str(string)

def reduce_and_tokenize_string(string):
    """
    Clean, reduce to nouns and adjectives, 
    and generate tokens (1-gram) from the string
    """
    string = str(string)
    return cleaner.tokenize_and_clean_str(string, True)

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

    def __init__(self):
        pass

    def get_feature_name(self):
        return self.__class__.__name__

    def get_feature_description(self):
        return self.feature_description


    @abc.abstractmethod
    def apply_rules(self, row):
        """
        Override this function. This extracts features from a 
        row of data
        """
        pass

######
## Feature ENG
#####


## Word and character counts
class NumOfWordsInSearchTerm(FeatureGenerator):
    feature_description = "Number of words in the search term"
  
    def apply_rules(self, row):
        search_term = row['search_term']
        return len(search_term.split())

class NumOfCharsInSearchTerm(FeatureGenerator):
    feature_description = "Number of characters in the search term"
  
    def apply_rules(self, row):
        search_term = row['search_term']
        return len(search_term)

class NumOfWordsInTitle(FeatureGenerator):
    feature_description = "Number of words in the product title"
  
    def apply_rules(self, row):
        product_title = row['product_title']
        return len(product_title.split())

class NumOfCharsInTitle(FeatureGenerator):
    feature_description = "Number of characters in the product title"
  
    def apply_rules(self, row):
        product_title = row['product_title']
        return len(product_title)

class NumOfWordsInProdDescrip(FeatureGenerator):
    feature_description = "Number of words in the product description"
  
    def apply_rules(self, row):
        prod_descrip = str(row['product_description'])
        return len(prod_descrip.split())

class NumOfCharsInProdDescrip(FeatureGenerator):
    feature_description = "Number of characters in the product description"
  
    def apply_rules(self, row):
        prod_descrip = str(row['product_description'])
        return len(prod_descrip)

class NumOfCharsInBrand(FeatureGenerator):
    feature_description = "Number of characters in the product brand"
  
    def apply_rules(self, row):
        BRAND_KEY = 'brand'.lower()
        attributes = eval(row['attributes'])
        for attr in attributes:
            if attr[0].lower().find(BRAND_KEY) != -1:
                attr_tokens = attr[1]
                return len(attr_tokens)
        return 0

## Search term matches
class SearchAndTitleMatch(FeatureGenerator):
    feature_description = 'Is the search term in the product title?'

    def apply_rules(self, row):
        search_term = row['search_term']
        prod_title  = row['product_title']
        return string_compare(search_term, prod_title)

class SearchAndDescriptionMatch(FeatureGenerator):
    feature_description = 'How does the search term match the product description?'

    def apply_rules(self, row):
        search_term = row['search_term']
        prod_de  = row['product_description']
        return string_compare(search_term, prod_de)
    
class SearchAndProductBrandMatch(FeatureGenerator):
    feature_description = 'Does the search term have a product?'

    def apply_rules(self, row):
        BRAND_KEY = 'brand'.lower()
        attributes = eval(row['attributes'])
        for attr in attributes:
            if attr[0].lower().find(BRAND_KEY) != -1:
                attr_tokens = attr[1]
                return string_compare(attr_tokens, row['search_term'])
        return 0

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
        return matches_sum

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
        return measure_match

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
        return measure_match

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
        return measure_in_range

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
        return measure_in_range

class SearchAndProductLastWordMatch(FeatureGenerator):
    feature_description = "Matching last word in product title assuming that is the predomenent noun to the search term"

    def apply_rules(self, row):
        product_title = row['product_title']
        last_word = product_title.split()[-1]
        search_term = row['search_term']
        return string_compare(last_word, search_term)

## Ratios
class RatioOfDescripToSearch(FeatureGenerator):
    feature_description = "Number of words in description to number of words in search term"

    def apply_rules(self, row):
        num_words_search = len(row['search_term'].split())
        num_words_descrip = len(str(row['product_description']).split())
        return num_words_descrip/num_words_search

class RatioOfTitleToSearch(FeatureGenerator):
    feature_descriptoin= "Number of words in product title to number of words in search term"

    def apply_rules(self, row):
        num_words_search = len(row['search_term'].split())
        num_words_title = len(row['product_title'].split())
        return num_words_title/num_words_search



######
# Using all the feature functions at once
#####
class FeatureFactory:
    def __init__(self):
        # istantiate all the feature classes
        self.feature_generators = map(lambda x: x(), self.feature_classes())

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
        return map(lambda x: x.get_feature_name(), self.feature_generators)

    def get_feature_descriptions(self):
        """
        Return a list of the features descriptions. 
        """
        return map(lambda x: (x.get_feature_name(), x.get_feature_description()),
					self.feature_generators
					)

    def get_feature_descriptions_map(self):
        return { pair[0]: pair[1]
                for pair in map(lambda x: (x.get_feature_name(), x.get_feature_description()),
                    self.feature_generators
					)
                }


    def apply_feature_eng(self,df):
        """
        Generates a new set of features
        to the data frame passed in
        """
        for feat in self.feature_generators:
            df[feat.get_feature_name()] = df.apply(
                    feat.apply_rules, axis=1
                    )
        return df

if __name__ == '__main__':
    # This is how we can use this class.
    # Just create a factory object 
    # and let it do the rest of the heavy lifting
    ff = FeatureFactory()

    # show that it actually creates objects
    print ff.get_feature_names()
    df = pd.read_csv('data/train_sample.csv')
    df2 = ff.apply_feature_eng(df)
    df2.to_csv('features.out')
