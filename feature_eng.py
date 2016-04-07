
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
    string = str(string)
    return string.lower()

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
    feature_name = ''
    feature_description = ''

    def __init__(self):
        pass

    def get_feature_name(self):
        return self.__class__.__name__

    def get_feature_description(self):
        return self.feature_description


    @abc.abstractmethod
    def apply_rules(self, row):
        pass

class SearchTermMatch(FeatureGenerator):
    feature_description = 'Is the search term in the product title?'

    def apply_rules(self, row):
        searh_term = tokenize_string(row['search_term'])
        prod_title = tokenize_string(row['product_title'])
        return prod_title.find(searh_term) != - 1






######
# Using all the feature functions at once
#####
class FeatureFactory:
    def __init__(self):
        self.feature_generators = map(lambda x: x(), self.feature_classes())

    def feature_classes(self):
        """
        Add new features to this list
        """
        return [
                SearchTermMatch,
                ]


    def get_feature_names(self):
        return map(lambda x: x.get_feature_name(), self.feature_generators)

    def apply_feature_eng(self,df):

        for feat in self.feature_generators:
            df[feat.get_feature_name()] = df.apply(
                    feat.apply_rules, axis=1
                    )
        return df

if __name__ == '__main__':
    st = SearchTermMatch()
    ff = FeatureFactory()

    # show that it actually creates objects
    print ff.get_feature_names()
    df = pd.read_csv('data/train_sample.csv')
    df2 = ff.apply_feature_eng(df)
    import pdb; pdb.set_trace()
