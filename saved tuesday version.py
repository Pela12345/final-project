import PyPDF2
from PyPDF2 import PdfFileReader

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

import spacy
from spacy.lang.es import Spanish
from spacy.lang.es.stop_words import STOP_WORDS

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string
import warnings
warnings.filterwarnings("ignore")

question = [str(input("¿Que quieres de España?: "))]

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
            # text_clean=re.sub("á", "a", text_clean)
            # text_clean=re.sub("é", "e", text_clean)
            # text_clean=re.sub("í", "i", text_clean)
            # text_clean=re.sub("ó", "o", text_clean)
            # text_clean=re.sub("ú", "u", text_clean)
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


def tfdif_vect(parties, text):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
    tfidf_matrix = tfidf_vectorizer.fit_transform(parties)
    text_transformed=tfidf_vectorizer.transform(text)
    return cosine_similarity(tfidf_matrix, text_transformed)

def plot_result(result):
    df = pd.DataFrame(result, index=party_names)
    df.plot(kind='bar', colors = 'mrybg')
    plt.title('Recomendación de Voto')
    plt.xlabel('Partidos')
    plt.ylabel('Similaridad')
    return plt.show()



party_names=['Podemos','PSOE','Ciudadanos', 'PP','Vox']
path_list=['data/podemos.pdf','data/psoe.pdf','data/ciudadanos.pdf','data/pp.pdf','data/vox.pdf']
parties=extract_pdf(path_list)
similarities=tfdif_vect(parties, question)
plot_result(similarities)
