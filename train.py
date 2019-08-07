import PyPDF2
from PyPDF2 import PdfFileReader

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from spacy.lang.es import Spanish
from spacy.lang.es.stop_words import STOP_WORDS

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import string
import warnings
warnings.filterwarnings("ignore")

def extract_pdf(paths):
    all_parties = []
    for path in paths:
        party_pdf = open(path, mode='rb')
        party = PyPDF2.PdfFileReader(party_pdf)
        pages = party.getNumPages()
        all_text = []
        for page in range(pages):
            info = party.getPage(page)
            text = info.extractText()
            text_clean = re.sub('\n', '', text)
            text_clean = re.sub("˜", "fi", text_clean)
            text_clean = re.sub("-", "", text_clean)
            all_text.append(text_clean)
        all_parties.append(str(all_text))

    return all_parties

def spacy_tokenizer(sentence):
    nlp=spacy.load('es')
    parser = Spanish()
    spacy_stopwords = spacy.lang.es.stop_words.STOP_WORDS
    STOPWORDS=list(spacy_stopwords)
    STOPWORDS.extend(('y','a','u','o','e'))
    tokens = parser(sentence)
    filtered_tokens = []
    for word in tokens:
        lemma = word.lemma_.lower().strip()
        lemma=re.sub("á", "a", lemma)
        lemma=re.sub("é", "e", lemma)
        lemma=re.sub("í", "i", lemma)
        lemma=re.sub("ó", "o", lemma)
        lemma=re.sub("ú", "u", lemma)
        lemma=re.sub("ñ", "n", lemma)
        if lemma not in STOPWORDS and re.search('^[a-zA-Z]+$', lemma):
            filtered_tokens.append(lemma)
    return filtered_tokens

party_names=['Podemos','PSOE','Ciudadanos', 'PP','Vox']
path_list=['data/podemos.pdf','data/psoe.pdf','data/ciudadanos.pdf','data/pp.pdf','data/vox.pdf']
parties=extract_pdf(path_list)
tfidf_vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,2) )
tfidf_matrix = tfidf_vectorizer.fit_transform(parties)

#tfidf_vectorizer_pkl_filename = 'tfidf_vectorizer.Wed.pkl'
tfidf_vectorizer_pkl = open('tfidf_vectorizer.Wed.pkl', 'wb')
pickle.dump(tfidf_vectorizer, tfidf_vectorizer_pkl)

#tfidf_matrix_pkl_filename = 'tfidf_matrix.Wed.pkl'
tfidf_matrix_pkl = open('tfidf_matrix.Wed.pkl', 'wb')
pickle.dump(tfidf_matrix, tfidf_matrix_pkl)

