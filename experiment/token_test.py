import nltk
import sys

sentence = sys.argv[1]
print nltk.pos_tag(nltk.word_tokenize(sentence))