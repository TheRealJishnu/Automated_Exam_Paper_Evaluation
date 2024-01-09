# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:52:39 2023

@author: HP-PC
"""
import numpy as np
from sklearn.cluster import DBSCAN
#from nltk.corpus import stopwords
#import re
#from striprtf.striprtf import rtf_to_text
#from gensim.models import Doc2Vec
#from gensim.models.doc2vec import TaggedDocument
#import os
#from scipy.cluster.hierarchy import linkage
#import matplotlib.pyplot as plt
#import model_file
#import data
#import vectors
'''
tagged_data = data.tagged_data
doc = data.doc
'''

def clustering(vectors):
    # perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.0999, min_samples=3)
    dbscan.fit(vectors)
    
    cluster_labels = dbscan.labels_
    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    for cluster_id in range(n_clusters_):
        #print(f"Cluster {cluster_id}:")
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        #print(cluster_indices)
    outlier_count = 0
    for n in cluster_labels:
        if(n==-1):
            outlier_count = outlier_count+1 
    unique_labels = set(cluster_labels) - {-1}
    midpoints = []
    for label in unique_labels:
        cluster_points = vectors[cluster_labels == label]
        midpoint = np.mean(cluster_points, axis=0)
        midpoints.append(midpoint)
        
    overall_midpoint = np.mean(midpoints, axis=0)
    
    #print("Midpoint of Clusters:", overall_midpoint)
    
    #print("No. of Outliers:", outlier_count)
    return cluster_labels, outlier_count, n_clusters_