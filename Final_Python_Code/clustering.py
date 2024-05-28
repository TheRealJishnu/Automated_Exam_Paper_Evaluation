# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:52:39 2023

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
from sklearn.metrics import silhouette_score
# import model_file
# import data
# import vectors
'''
tagged_data = data.tagged_data
doc = data.doc
'''

def clustering(vectors, epsilon):
    # perform DBSCAN clustering
    # epsilon = 1.258
    # ff = open("Result//q5_Data.txt", "a")
    # ff.write("Q1\n")
    # for epsilon in np.arange(0.5, 2, 0.001):
    dbscan = DBSCAN(eps=epsilon, min_samples=3) # Changed eps
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
    #################
    n = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    # print(n)
    if n < 2:
        silhouette_avg = -2
    else:
        silhouette_avg = round(silhouette_score(vectors, cluster_labels), 4)  # SILHOUETTE
    
    #print("Midpoint of Clusters:", overall_midpoint)
    
    # ss = f"{round(epsilon, 3)} No. of Outliers: {outlier_count}, Clusters : {n_clusters_}, Silhoutte {silhouette_avg}"
    # ff.write(ss)
    # ff.write('\n')
        
    # ff.close()
    return cluster_labels, outlier_count