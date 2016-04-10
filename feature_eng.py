
"""
Set of rule based features
"""
import abc

import cleaner
import pandas as pd


#################
### Util Functions
#################
def tokenize_string(string):
    """
    Clean and generate tokens (1-gram) from the string
    """
    string = str(string)
    return cleaner.tokenize_and_clean_str(string)

def string_compare(str_a, str_b):
    """
    Comparison of two strings. The method is TODO
    Current uses boolean matching
    """
    a = set(tokenize_string(str_a))
    b = set(tokenize_string(str_b))
    return len(a.intersection(b))

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
    import pdb; pdb.set_trace()
