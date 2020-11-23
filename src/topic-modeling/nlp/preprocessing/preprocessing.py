import pandas as pd
import spacy
from spacy.util import compile_infix_regex
from spacy.tokenizer import Tokenizer
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups

"""
    The script goal is to preprocess
    the data related to covid news
    located in /datasets/covid-news/news.csv
    to further feed a Natural Language Preprocessing
    model. The methods applied are stop word dropping,
    tokenization and lemmatization, the target features
    are title, text, and description.
    @author Junior Vitor Ramisch <junior.ramisch@gmail.com> 

"""

DATA_PATH = "../../../../datasets/covid-news/"
N_ROWS=50

nlp_english = spacy.load("en_core_web_md")

lookups = Lookups()
lemmatizer = Lemmatizer(lookups)

def create_custom_tokenizer():

    inf = list(nlp_english.Defaults.infixes)
    inf = [x for x in inf if '-|–|—|--|---|——|~' not in x]
    infix_re = compile_infix_regex(tuple(inf))

    # right away from https://stackoverflow.com/questions/55241927/spacy-intra-word-hyphens-how-to-treat-them-one-word
    nlp_english.tokenizer = Tokenizer(nlp_english.vocab, prefix_search=nlp_english.tokenizer.prefix_search,
                                    suffix_search=nlp_english.tokenizer.suffix_search,
                                    infix_finditer=infix_re.finditer,
                                    token_match=nlp_english.tokenizer.token_match,
                                    rules=nlp_english.Defaults.tokenizer_exceptions)

def get_document_set(name):
    create_custom_tokenizer()

    document_set = pd.read_csv(filepath_or_buffer=DATA_PATH + name,
                               usecols=[ "text", "title", "description"],
                               nrows=N_ROWS, header=0, encoding="utf-8")
    return document_set

def tokenize(string):
    
    tokens=nlp_english(string)

    return [token for token in tokens ]

def drop_stop_words(tokens):
    return [token for token in tokens if (not token.is_stop) and (not token.is_digit) and (not token.is_quote) and (not token.is_punct)]


def lemmatize(tokens):
    return [token.lemma_ for token in tokens]


