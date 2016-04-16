import requests
import re
import time
import spellcheck

from random import randint

import nltk
from nltk.corpus import stopwords
from unicodedata import normalize
from nltk.tag.perceptron import PerceptronTagger

import feature_eng

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

def full_clean_string(m_str):
    """
    Perform hardcode_cleaning, downcase, stopword removal and stemming
    """
    try:
        m_str = m_str.decode('utf-8')
    except UnicodeEncodeError:
        m_str = m_str.encode('ascii', errors='ignore')
    except:
        import pdb; pdb.set_trace()

    m_str = hardcode_cleaning(m_str)
    cleaned_string = remove_stop_words(downcase_str(m_str))
    cleaned_string = stem_words(cleaned_string)
    return cleaned_string

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

def get_size(row):
    
    sizes = []
    attributes = eval(row['attributes'])
    SIZE_KEY = '(in.)'

    for attr in attributes:
        if attr[0].lower().find(SIZE_KEY) != -1:
            attr_tokens = attr[1]
            sizes += feature_eng.numbers_in_string(attr[1])
    # check the title and the description as well
    title = row['product_title']
    descript = row['product_description']
    size_filter = 'in.|inches|inch'

    title_nums = feature_eng.numbers_in_string(
            title, 
            prefilter=size_filter
            )
    descript_nums = feature_eng.numbers_in_string(
            descript,
            prefilter=size_filter,
            )
    sizes = sizes + title_nums + descript_nums
    return sizes

def get_weight(row):
    
    weights = []
    attributes = eval(row['attributes'])
    SIZE_KEY = '(lb.)'

    for attr in attributes:
        if attr[0].lower().find(SIZE_KEY) != -1:
            attr_tokens = attr[1]
            #weights.append(float(attr[1]))
            weights += feature_eng.numbers_in_string(attr[1])
    weight_filter = 'lb.|lb|pound|pounds'
    title_nums = feature_eng.numbers_in_string(
            row['product_title'], 
            prefilter=weight_filter
            )
    descript_nums = feature_eng.numbers_in_string(
            row['product_description'],
            prefilter=weight_filter,
            )
    weights = weights  + title_nums + descript_nums

    return weights

def get_brand(row):

    BRAND_KEY = 'brand'.lower()
    attributes = eval(row['attributes'])
    for attr in attributes:
        if attr[0].lower().find(BRAND_KEY) != -1:
            attr_tokens = attr[1]
            brand = full_clean_string(attr_tokens)
            return brand
    return 'Unbranded'

def clean_search(row):
    """
    Clean the search term
    """
    search_term = row['search_term']
    cleaned_term = search_term
    if (search_term in spellcheck.spellchecks):
        cleaned_term = spellcheck.spellchecks[search_term]
    cleaned_term = full_clean_string(cleaned_term)
    return cleaned_term

def clean_title(row):
    """
    Clean the title 
    """
    title = row['product_title']
    return full_clean_string(title)

def clean_description(row):
    """
    Clean the product_description 
    """
    product_description = row['product_description']
    return full_clean_string(product_description)

def reduce_title_nadj(row):
    """
    Reduce the title to noun or adjective
    """
    # Use global Tagger because its much faster
    title = row['product_title']
    tags = nltk.tag._pos_tag(nltk.word_tokenize(title), None, TAGGER)
    cleaned_string = ""
    for i in xrange(len(tags)):
        if is_noun_or_adjective(tags[i][1]):
            cleaned_string += (tags[i][0] + " ")
    return downcase_str(cleaned_string)

def reduce_description_nadj(row):
    """
    Reduce the product division to noun or adjective
    """
    # Use global Tagger because its much faster
    prod_des = row['product_description']
    tags = nltk.tag._pos_tag(nltk.word_tokenize(prod_des), None, TAGGER)
    cleaned_string = ""
    for i in xrange(len(tags)):
        if is_noun_or_adjective(tags[i][1]):
            cleaned_string += (tags[i][0] + " ")
    return downcase_str(cleaned_string)

def find_preceding_dominant_word(tags, index):

    i = index
    while i > 0:
        if is_noun_or_adjective(tags[i][1]):
            return tags[i][0]
        i = i-1
    return ""

def reduce_to_dominant_words(row):
    """
    Compare the dominant words in the product title with the search term
    """
    title = row['product_title']
    tags = nltk.tag._pos_tag(nltk.word_tokenize(title), None, TAGGER)
    dom_words_string = ""

    # add all NAdj words right preceding a stop word
    for j in xrange(len(tags)):        
        if tags[j][0] in stopwords.words('english') and j > 0:
            dom_words_string += (find_preceding_dominant_word(tags, j) + " ")

    # Also add the last word
    if len(tags) > 0:
        dom_words_string += (find_preceding_dominant_word(tags, len(tags)-1) + "")

    dom_words_string = hardcode_cleaning(dom_words_string)
    return dom_words_string

if __name__ == '__main__':
    pass
