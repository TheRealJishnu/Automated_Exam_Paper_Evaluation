# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:43:00 2024

@author: there
"""

import nltk
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
import vectors
import clustering
from sklearn.metrics import silhouette_score
import helper

def Ideal_of(no, v_list):
    if(no == 1):
        return v_list[9]
    elif(no == 2):
        return v_list[1]
    elif(no == 3):
        return v_list[4]
    elif(no == 4):
        return v_list[4]
    elif(no == 5):
        return v_list[4]
    else:
        raise ValueError("Invalid Question Number Selected, Should be 1-5")
        exit(1)

def inp_tot_marks(q):
    while(True):
        tot_marks = input(f"Enter Full Marks for Question {q} : ")
        if tot_marks == "":
            print("You Cannot Skip Entering Full Marks")
        elif float(tot_marks) <= 0:
            print("Error: Full Marks Cannot be Less Than or Equal to 0, Enter Again")
        else:
            return float(tot_marks)
        

def Epsilon_Of(no:int) -> float:
    if(no == 1):
        return 1.032
    elif(no == 2):
        return 1.105
    elif(no == 3):
        return 1.23
    elif(no == 4):
        return 1.631
    elif(no == 5):
        return 1.905
    else:
        raise ValueError("Question Number Must be Between 1-5")

def evaluation_main(q):
    # DIFFERENT CONTAINERS
    # MANY ARE CURRENTLY NOT USED
    vector_list = []
    cluster_list = []
    tagged_data = data.tagged_data
    y_sil = []
    y_outlier = []
    y_clus = []
    # y=1
    
    # CURRENT VECTOR DIMENSION
    cur_vec = 100
    
    # CREATING MODEL AND CLUSTERING, PREVIOUSLY INSIDE FOR LOOP
    # model = Doc2Vec(vector_size=cur_vec, window=5, min_count=1, dm=0, epochs=10)
    m_path = r"C:\Users\there\Downloads\Segmentations\v10_customModel_Improved\book_model_100.bin"
    model = Doc2Vec.load(m_path)
    # q = int(input("Enter Question No : "))
    vector_list = (vectors.vecs(model, q))
    cluster_labels, outlier_n = clustering.clustering(vector_list, Epsilon_Of(q))
    cluster_list = (cluster_labels)
    y_outlier.append(outlier_n)
    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    y_clus.append(n_clusters_)
    
    # silhouette_avg = silhouette_score(vector_list, cluster_labels)
    # y_sil.append(silhouette_avg)
    # print(f"Dimension {cur_vec}: Silhouette Score {silhouette_avg} and No. of Outliers {outlier_n} and Clusters {n_clusters_}")
        
          
        
    # Separating Vectors based on Clusters
    # print(vector_list)
    v_list = vector_list
    clustered_vecs =[]
    labels = cluster_list
    n_clus = len(set(labels)) - (1 if -1 in cluster_labels else 0)
    # print(n_clus)       #########################################
    # print(labels)
    for i in range(n_clus):
        temp = []
        for j in range(105):
            if labels[j] == i:
                temp.append(v_list[j])
        clustered_vecs.append(temp)
        
        
        
        
    # Storing Vector Average in vec_aver list
    vec_aver = []
    for i in clustered_vecs:
        vec_aver.append(helper.Vector_Average(i, cur_vec))
    # print(vec_aver)
    
    # Calculating Distance
    dist_list = []
    # ideal = v_list[9].flatten()
    ideal = Ideal_of(q, v_list).flatten()
    vec_aver = [np.array(e).flatten() for e in vec_aver]
    for e in vec_aver:    
        dist_list.append(helper.Cosine_Similarity(ideal, e))
        # print(e.shape, ideal.shape)
    # print(dist_list)
    
    
    # EVALUATION
    '''
    ok so everything is in sorted ascending order, 
    dimension of dist_list and number of cluster is same, so we can mark them accordingly
    
    question? outlier?
    '''
    tot_marks = inp_tot_marks(q)
    # tot_marks = 3
    
    f = open("Result//EvaluatedAnswers.txt", "a")
    f.write(f"\t---------- Question {q} ----------\n")
    # f.close()
    mark_list = []
    
    # print(labels.shape[0])
    for i in range(labels.shape[0]):
        flg = False
        if labels[i] == -1:     # HANDLING OUTLIERS
            cur_marks = tot_marks * (helper.Cosine_Similarity(vec1=ideal, vec2=v_list[i]))
            # print(f"Q{i+1} : ",round(cur_marks, 2))
            flg = True
        else:
            cur_marks = tot_marks * dist_list[labels[i]]
        cur_marks = round(cur_marks, 2)
        mark_list.append(cur_marks)
        
    
        
        st = f"Marks for Q{q}S{i+1} : {cur_marks}\t"
        if flg:
            st += "Outlier\n"
        else:
            st += "\n"
        f.write(st)
        
        
    f.write("\n")
    f.close()
    
    
    
    # TESTING
    return v_list, mark_list