import numpy as np
import pandas as pd
#import gensim
#from nltk.corpus import stopwords
import spacy
from spacy.tokenizer import Tokenizer

"""
    The script goal is to preprocess
    the data related to covid news
    located in /datasets/covid-news/news.csv
    to further feed a Natural Language Preprocessing
    model. The methods applied are stop word dropping,
    tokenization and lemmatization, and the target features
    are title, text, and description.
    @author Junior Vitor Ramisch <junior.ramisch@gmail.com> 

"""

DATA_PATH = "../../../../datasets/covid-news/"
N_ROWS=1

nlp_english = spacy.load("en_core_web_md")

def create_custom_tokenizer():
    # right away from https://stackoverflow.com/questions/55241927/spacy-intra-word-hyphens-how-to-treat-them-one-word
    infixes = [r"'s\b", r"(?<!\d)\.(?!\d)"] +  nlp_english.Defaults.prefixes
    infix_re = spacy.util.compile_infix_regex(infixes)
    nlp_english.tokenizer = Tokenizer(nlp_english.vocab, infix_finditer=infix_re.finditer)

def get_set(name):
    create_custom_tokenizer()

    document_set = pd.read_csv(filepath_or_buffer=DATA_PATH + name,
                               usecols=[ "text", "title", "description"],
                               nrows=N_ROWS, header=0, encoding="utf-8")
    return document_set

def get_tokens(string):
    
    doc=nlp_english(string)

    return [token for token in doc ]

def drop_stop_words(tokens):
    return [token for token in tokens if (not token.is_stop) and (not token.is_digit) and (not token.is_quote)]


def lemmatize():
    pass


