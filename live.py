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
import speech_recognition as sr
import pickle
import string
import warnings
warnings.filterwarnings("ignore")

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

def tfdif_vect(matrix, model, text):
    text_transformed=model.transform(text)
    return cosine_similarity(matrix, text_transformed)

def to_percent(x):
    x = (x / (x.sum())) * 100
    return x

def plot_result(result, question):
    color = ['purple', 'red', 'orange', 'blue', 'green']
    df = pd.DataFrame(result, index=party_names)
    plt.figure(figsize=(15, 11))
    plt.bar(party_names, df[0], color=color)
    #plt.title('Recomendación de Voto').set_fontsize(23)
    plt.xlabel('Partidos').set_fontsize(16)
    plt.ylabel('Porcentaje').set_fontsize(16)
    x = [i for i in np.where(result== np.amax(result))[0]]
    v = int(re.sub('[][]', '', str(x)))
    plt.text(-0.4, 104, question, style='italic',
             bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6}).set_fontsize(16)
    plt.text(1.2, 80, party_names[v], style='italic',
             bbox={'facecolor': color[v], 'alpha': 0.5, 'pad': 8}).set_fontsize(60)
    plt.ylim((0, 95))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=28)
    # plt.imshow(y)
    plt.show()

def reconocimiento_voz():                                   # graba audio
    r=sr.Recognizer()
    with sr.Microphone() as s:
        print('Escuchando...')
        r.adjust_for_ambient_noise(s)
        audio=r.listen(s,timeout=12)

    datos=''  # reconocimiento de voz
    try:      # Usa API key por defecto, para usar otra: `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        datos=r.recognize_google(audio, language='es-ES')
    except sr.UnknownValueError:
        print("Google Speech Recognition no ha podido reconocer el audio.")
    except sr.RequestError as e:
        print("No hay respuesta desde el servicio de Google Speech Recognition; {0}".format(e))
    return datos

party_names=['Podemos','PSOE','Ciudadanos', 'PP','Vox']
path_list=['data/podemos.pdf','data/psoe.pdf','data/ciudadanos.pdf','data/pp.pdf','data/vox.pdf']

tfidf_vectorizer_pkl = open('tfidf_vectorizer.Wed.pkl', 'rb')
tfidf_vectorizer = pickle.load(tfidf_vectorizer_pkl)

tfidf_matrix_pkl = open('tfidf_matrix.Wed.pkl', 'rb')
tfidf_matrix = pickle.load(tfidf_matrix_pkl)

while True:

    try:
        print("¿Que quieres de España?")
        data=reconocimiento_voz()
        #question = [str(input("¿Que quieres de España?: "))]
        question = [str(data)]
        #question = [str(input("¿Que quieres de España?: "))]
        similarities = tfdif_vect(tfidf_matrix, tfidf_vectorizer, question)
        percentages=to_percent(similarities)
        plot_result(percentages,question)


    except KeyboardInterrupt:
        print('\nexiting...')
        plt.close('all')
        break
while True:
    question = [str(input("¿Que quieres de España?: "))]
    similarities = tfdif_vect(tfidf_matrix, tfidf_vectorizer, question)
    percentages=to_percent(similarities)
    try:
        plot_result(percentages,question)
    except ValueError:
        print('No encuentro ningun partido apropiado, VOTA EN BLANCO')