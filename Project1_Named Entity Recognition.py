# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:06:34 2021

@author: ricky
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#pip install pdfminer.six
#pip install PyPDF2
import PyPDF2

mergeFile = PyPDF2.PdfFileMerger()
#reading PDF using mergerFile 
mergeFile.append(PyPDF2.PdfFileReader('27-30.pdf', 'rb'))
mergeFile.append(PyPDF2.PdfFileReader('888-97.pdf', 'rb'))
mergeFile.append(PyPDF2.PdfFileReader('888-896.pdf', 'rb'))
mergeFile.append(PyPDF2.PdfFileReader('603913.pdf', 'rb'))
# combining all pdf into 1 pdf file
mergeFile.write("NewMergedFile.pdf")

#extracting text using pdfminer library
from pdfminer.high_level import extract_text
text = extract_text('NewMergedFile.pdf')


# Pre-Processing of the obtained Text
wordnet=WordNetLemmatizer()                 # For Applying Lemmatization
sentences = nltk.sent_tokenize(text)        # Converting Paragraph to sentences
corpus = []                                 # Saving all sentences after applying the following stopwords etc.....
for i in range(len(sentences)):             # Reading through all sentences 
    punctuation = ['(',')',';',':','[',']',',','-']        # Custom list of stopwords to be removed from the texts 
    review = re.sub('punctuation', ' ', sentences[i])  
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


wordstr = ' '.join(str(e) for e in corpus)

tokens = nltk.word_tokenize(wordstr)         # 8341 in other
tokens = [w for w in tokens]

###############################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
text_data = np.array(tokens)
feature_matrix = tfidf.fit_transform(text_data)
feature_matrix.toarray()


# Show tf-idf feature matrix
tfidf.get_feature_names()

df1 = pd.DataFrame(feature_matrix.toarray(), columns=tfidf.get_feature_names())

# Preparing a Word Cloud on the RAW TEXT
from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=60).generate(text)
plt.figure(figsize=(16,12))

# plot wordcloud in matplotlib
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud on Keywords from All PDFs")
plt.show()
###############################################################################

# Now to extract Named Entities from the pre-prosessed text using MED-7 Model

#!pip install spacy 
import spacy
#pip install -U spacy

# MED-7 Model Installation
#!pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl

med7=spacy.load("en_core_med7_lg")    # Loading of MED-7 model Data


# create distinct colours for labels
col_dict = {}
seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
    col_dict[label] = colour

options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}

doc = med7(wordstr)

spacy.displacy.render(doc, style='ent', jupyter=True, options=options)

entities = [(ent.text, ent.label_) for ent in doc.ents]
entities

# Creating a Data frame and Adding the obtained Entities into it

import pandas as pd
doc = med7(wordstr)
entities = []
labels = []
pos_start = []
pos_end = []
for ent in doc.ents:
    entities.append(ent)
    labels.append(ent.label_)
    pos_start.append(ent.start_char)
    pos_end.append(ent.end_char)
df = pd.DataFrame({"entities":entities, "labels":labels, "pos_start":pos_start,"pos_end":pos_end}) 
df

df_csv = df.to_csv()
print(df_csv)

df
df.to_csv('results.csv', index=False)

df.to_csv(r"results.csv")
drug=df[df.labels=="DRUG"]

drug=pd.DataFrame(drug)
print(drug)





