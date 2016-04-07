"""
Set of rule based features
"""
import abc

import cleaner


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
        return self.get_feature_name

    def get_feature_description(self):
        return self.feature_description


    @abc.abstractmethod
    def apply_rules(self, row):
        pass
class SearchTermMatch(FeatureGenerator):
    feature_name = 'feat_search_term'
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
        pass

    def feature_classes(self):
        """
        Add new features to this list
        """
        return [
                SearchTermMatch,
                ]


if __name__ == '__main__':
    st = SearchTermMatch()
    ff = FeatureFactory()

    # show that it actually creates objects
    map(lambda x: x().get_feature_name(), ff.feature_classes())
    df = pd.read_csv('data/train_sample.csv')
    import pdb; pdb.set_trace()
