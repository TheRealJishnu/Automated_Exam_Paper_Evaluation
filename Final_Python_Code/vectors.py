# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:46:27 2023

@author: HP-PC
"""
from nltk.corpus import stopwords
import re
from striprtf.striprtf import rtf_to_text
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import os
import numpy as np
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
# import model_file
import data
# import data

tagged_data = data.tagged_data
tagged_train = data.tagged_train
doc = data.doc

'''
Q1: 1-105 --> 001-105 --> 1-20
Q2: 1-112 --> 106-217 --> 106-126
Q3: 1-106 --> 218-323 --> 218-238
Q4: 1-119 --> 324-442 --> 324-344
Q5: 1-108 --> 443-550 --> 443-463
'''

def Answers_of(q_no):
    if q_no == 1:
        return doc[0:105]
    elif q_no == 2:
        return doc[105:217]
    elif q_no == 3:
        return doc[217:323]
    elif q_no == 4:
        return doc[323:442]
    elif q_no == 5:
        return doc[442:550]
    else:
        raise ValueError("Invalid Question Number Selected, Should be 1-5")
        


def vecs(model, q):
    # model.build_vocab(tagged_data)
    # model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    # Get document vectors
    '''Not Used
    document_vectors = np.array([model.dv[str(i)] for i in range(len(tagged_data[442:550]))])
    '''
    # Get the document vectors for your unlabeled documents
    inferred_vectors = []
    # q = int(input("Enter Question No : "))
    for i, vector in enumerate(Answers_of(q)):     # Question Selection
        tokenized_doc = vector.split()
        doc_vec = model.infer_vector(tokenized_doc)
        inferred_vectors.append(doc_vec)
        #print(f"Vector for Document {i + 1}:")
        #print(doc_vec)  # Process the inferred vector as needed
        #print("\n")
    inferred_vectors = np.array(inferred_vectors)
    return inferred_vectors
