import subprocess
import pandas as pd
import sys
import time
import abc

sys.path.append("../")
import cleaner
import feature_eng as fe

# For a tough to get around java-based reason, this MUST be run in the esalib directory.
def compare_terms(process, pipe, t1, t2):

    score = 0
    while (True):
        out = ""
        try:
            process.stdin.write(t1 + "\n" + t2 + "\n")
            time.sleep(0.05)
            out = pipe.read()
            # Do a redo, incase pipe didnt read it fast enough
            if (len(out.split("\n")) > 2):
                no_return = False
            score = float(out.split("\n")[-3])
            break
        except IndexError:
            print "Caught IndexError"
            print out
        except ValueError:
            print "Caught ValueError"
            print out

    return score

class ESAFeatureGenerator(fe.FeatureGenerator):
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

class ESAFeatureFactory(fe.FeatureFactory):

    def __init__(self):
        # istantiate all the feature classes
        self.feature_generators = map(lambda x: x(), self.feature_classes())

    def feature_classes(self):
        """
        Returns list of feature generator class
        The list will be anything that inherits
        from the base FeatureGenerator class
        """
        return [cls for cls in ESAFeatureGenerator.__subclasses__()]

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

        for feat in self.feature_generators:
            feat.teardown()

        return df

class ESASimilarityMatch(ESAFeatureGenerator):

    feature_description = "Do any nouns in the search term and description have high similarity?"
  
    def __init__(self):
        init_command = "./run_analyzer"        
        self.fw = open("tmpout", "wb")
        self.outpipe = open("tmpout", "r")
        self.process = subprocess.Popen(init_command, stdin=subprocess.PIPE, stdout=self.fw, stderr=self.fw)
        self.logfile = open("esa.log", "w")
        time.sleep(5)

    def teardown(self):
        self.logfile.close()

    def get_feature_name(self):
        return ['ESA_0', 'ESA_025', 'ESA_05', 'ESA_075', 'ESA_Max']

    def apply_rules(self, row):
        search_term = row['search_term_cleaned']
        dom_words  = row['dominant_words']
        relevance = row['relevance']

        search_reduced = search_term.split()
        dom_reduced = dom_words.split()

        scores = []

        for term in search_reduced:
            
            for t_term in dom_reduced:
                
                esa_score = compare_terms(self.process, self.outpipe, term, t_term)
                score_map = dict(search=search_term, dom_words=dom_words, relevance=relevance, t1=term, t2=t_term, esa_score=esa_score)
                print score_map
                self.logfile.write(str(score_map) + "\n")
                scores.append(score_map)

        esadf = pd.DataFrame(scores)
        esadf = esadf.fillna(0)
        
        # Remove all 0s and 1s
        esadf = esadf[esadf.esa_score != 0]
        esadf = esadf[esadf.esa_score < 0.999]

        quant = esadf.quantile(q=[0,0.25,0.5,0.75,1])

        return pd.Series(dict(ESA_0=quant['esa_score'][0], 
            ESA_025=quant['esa_score'][0.25], 
            ESA_05=quant['esa_score'][0.5], 
            ESA_075=quant['esa_score'][0.75], 
            ESA_Max=quant['esa_score'][1]))

if __name__ == '__main__':
    
    df = pd.read_csv('../features_pp.out')
    ff = ESAFeatureFactory()
    print ff.get_feature_names()

    df2 = ff.apply_feature_eng(df)
    df2.to_csv('esa_features.out')