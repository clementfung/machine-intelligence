import requests
import re
import time
from random import randint

import nltk
from nltk.corpus import stopwords

from unicodedata import  normalize
from nltk.tag.perceptron import PerceptronTagger
# Global variable to load once
print 'Loading global tagger... please wait a few seconds'
TAGGER = PerceptronTagger()

START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;'),
)

def spell_check(s):
    """
    Uses google to spell check
    https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos/notebook
    """
    q = '+'.join(s.split())
    time.sleep(  randint(0,2) ) #relax and don't let google be angry
    r = requests.get("https://www.google.ca/search?q="+q)
    content = r.text
    start=content.find(START_SPELL_CHECK) 
    if ( start > -1 ):
        start = start + len(START_SPELL_CHECK)
        end=content.find(END_SPELL_CHECK)
        search= content[start:end]
        search = re.sub(r'<[^>]+>', '', search)
        for code in HTML_Codes:
            search = search.replace(code[1], code[0])
        search = search[1:]
    else:
        search = s
    str(search)
    return search

###
# Basic word cleaning
###

def remove_stop_words(m_str):
    """
    Returns a string with common words removed
    """
    n_str = ''
    for c in m_str.split(' '):
        if c not in stopwords.words('english'):
            n_str += c + ' '
    return n_str

def downcase_str(m_str):
    """
    Downcase the string
    """
    return m_str.lower()

def stem_words(m_str):
    """
    Completes porter stemming
    """
    n_str = ''
    t =  nltk.stem.PorterStemmer()
    for c in m_str.split(' '):
        n_str += t.stem(c) + ' '
    return n_str

def reduce_to_nouns_and_adjectives(m_str, verbose=False):
    # Use global Tagger because its much faster
    tags = nltk.tag._pos_tag(nltk.word_tokenize(m_str), None, TAGGER)
    cleaned_string = ""
    for i in xrange(len(tags)):
        if is_noun_or_adjective(tags[i][1]):
            cleaned_string += (tags[i][0] + " ")
    if (len(cleaned_string) == 0 and verbose):
        print "WARNING:" + m_str + " reduced to nothing after NAdj"
         
    return cleaned_string

def is_noun_or_adjective(tag_str):
    return "NN" in tag_str or "JJ" in tag_str

def tokenize_and_clean_str(m_str, stem = True, reduce = False):
    """
    Puts together all the tokenizing / cleaning
    functions
    """
    try:
        m_str = m_str.decode('utf-8')
    except UnicodeEncodeError:
        m_str = m_str.encode('ascii', errors='ignore')
    except:
        import pdb; pdb.set_trace()

    m_str = hardcode_cleaning(m_str)
    cleaned_string = remove_stop_words(downcase_str(m_str))

    if (reduce):
        cleaned_string = reduce_to_nouns_and_adjectives(cleaned_string)
    
    if (stem):
        cleaned_string = stem_words(cleaned_string)

    return cleaned_string.strip().split(' ')

def hardcode_cleaning(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
    return s

if __name__ == '__main__':
    pass
