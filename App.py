# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 17:07:32 2021

@author: ricky
"""

import pandas as pd
# from PIL import Image
from pdfminer.high_level import extract_text
import streamlit as st
import PyPDF2
import spacy



st.title(""" # Feature Extraction  Medical Journalssssss """)



mergeFile = PyPDF2.PdfFileMerger()

docx_file = st.file_uploader("Upload Document",type=['pdf','txt'])

if docx_file is not None:
                    
    text = extract_text(docx_file)
    
    print(text)
                  
    med7 = spacy.load("en_core_med7_lg")

    # create distinct colours for labels
    col_dict = {}
    seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
    for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
        col_dict[label] = colour
    
    options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}
    
    doc = med7(text)
    entities=[]
    en=[]
    labels=[]
    lb=[]
    pos_start=[]
    pos_end=[]
    for ent in doc.ents:
        entities.append(ent)
        labels.append(ent.label_)
        pos_start.append(ent.start_char)
        pos_end.append(ent.end_char)

    for entity in entities:
        en.append("'"+str(entity)+"'")
    for label in labels:
        lb.append("'"+str(label)+"'")
            
    data = {'entities':en, 'labels':lb, 'pos_start':pos_start, 'pos_end':pos_end}
    print(data)
    df=pd.DataFrame(data)
    
    st.write(df)
    
    df = df.to_csv()
    

