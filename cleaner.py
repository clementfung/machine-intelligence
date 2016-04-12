import requests
import re
import time
from random import randint

import nltk
from nltk.corpus import stopwords

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

def reduce_to_nouns_and_adjectives(m_str):
    # Use global Tagger because its much faster
    tags = nltk.tag._pos_tag(nltk.word_tokenize(m_str), None, TAGGER)
    cleaned_string = ""
    for i in xrange(len(tags)):
        if is_noun_or_adjective(tags[i][1]):
            cleaned_string += (tags[i][0] + " ")
    # TODO: Get some verbose mode going
    #print "REDUCTION:" 
    #print m_str
    #print cleaned_string
    #print "-----"
    if (len(cleaned_string) == 0):
        print "WARNING:" + m_str + " reduced to nothing after NAdj"
         
    return cleaned_string

def is_noun_or_adjective(tag_str):
    return "NN" in tag_str or "JJ" in tag_str

def tokenize_and_clean_str(m_str, reduce = False):
    """
    Puts together all the tokenizing / cleaning
    functions
    """
    m_str = m_str.decode('utf-8')
    cleaned_string = remove_stop_words(downcase_str(m_str))

    if (reduce):
        cleaned_string = reduce_to_nouns_and_adjectives(cleaned_string)
    
    return stem_words(cleaned_string)\
            .strip().split(' ')
