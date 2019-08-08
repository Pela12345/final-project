import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import speech_recognition as sr
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

import pickle
import string
import warnings
warnings.filterwarnings("ignore")

def plot_result(result, question):
    color = ['purple', 'red', 'orange', 'blue', 'green']
    df = pd.DataFrame(result, index=party_names)
    plt.figure(figsize=(10, 12))
    plt.bar(party_names, df[0], color=color)
    plt.title('Recomendación de Voto').set_fontsize(23)
    plt.xlabel('Partidos').set_fontsize(16)
    plt.ylabel('Porcentaje').set_fontsize(16)
    x = [i for i in np.where(result== np.amax(result))[0]]
    v = int(re.sub('[][]', '', str(x)))
    plt.text(-0.3, 104, question, style='italic',
             bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 6}).set_fontsize(16)
    plt.text(1.2, 80, party_names[v], style='italic',
             bbox={'facecolor': color[v], 'alpha': 0.5, 'pad': 8}).set_fontsize(60)
    plt.ylim((0, 95))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=28)
    # plt.imshow(y)
    plt.show()

party_names = ['Podemos', 'PSOE', 'Ciudadanos', 'PP', 'Vox']
plot_result(([[10.98783229],
       [16.63368329],
       [46.811943  ],
       [15.33446994],
       [30.23207148]]), ['mas igualdad economica'])

'''
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

print(reconocimiento_voz())'''