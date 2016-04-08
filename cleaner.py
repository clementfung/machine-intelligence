import requests
import re
import time
from random import randint
from PorterStemmer import PorterStemmer

import nltk
from nltk.corpus import stopwords

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
    #t = PorterStemmer()
    t =  nltk.stem.PorterStemmer()
    for c in m_str.split(' '):
        n_str += t.stem(c) + ' '
    return n_str

def tokenize_and_clean_str(m_str):
    """
    Puts together all the tokenizing / cleaning
    functions
    """
    return  stem_words(remove_stop_words(downcase_str(m_str)))\
            .strip().split(' ')
