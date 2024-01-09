# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:46:27 2023

@author: HP-PC
"""
import numpy as np
import data
#from nltk.corpus import stopwords
#import re
#from striprtf.striprtf import rtf_to_text
#from gensim.models import Doc2Vec
#from gensim.models.doc2vec import TaggedDocument
#import os
#from scipy.cluster.hierarchy import linkage
#import matplotlib.pyplot as plt
#from sklearn.cluster import DBSCAN
#import model_file



tagged_data = data.tagged_data
doc = data.doc

def vecs(model):
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    # Get document vectors
    document_vectors = np.array([model.dv[str(i)] for i in range(len(tagged_data))])
    # Get the document vectors for your unlabeled documents
    inferred_vectors = []
    for i, vector in enumerate(doc):
        tokenized_doc = vector.split()
        doc_vec = model.infer_vector(tokenized_doc)
        inferred_vectors.append(doc_vec)
        #print(f"Vector for Document {i + 1}:")
        #print(doc_vec)  # Process the inferred vector as needed
        #print("\n")
    inferred_vectors = np.array(inferred_vectors)
    return inferred_vectors
