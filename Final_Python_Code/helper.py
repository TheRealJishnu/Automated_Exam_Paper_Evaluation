# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:59:04 2024

@author: there
"""

import numpy as np
from math import floor as fl

def Cosine_Similarity(vec1, vec2):
    """
    Parameters
    ----------
    vec1 :  TYPE.
                1D numpy array
            DESCRIPTION.
                vec1 is the first vector that will be used to calculate
        similarity.
        
    vec2 :  TYPE.
                1D numpy array
        DESCRIPTION.
                vec2 is the first vector that will be used to calculate
        similarity.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if type(vec1) is list:
        vec1 = np.array(vec1)
    if type(vec2) is list:
        vec2 = np.array(vec1)
    if(vec1.shape[0] != vec2.shape[0]):
        raise ValueError("Dimension size of the vectors must be same")
    else:
        dot_prod = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            # print(norm1, norm2)
            
            
            return 0
        return dot_prod/(norm1 * norm2)

def Minkowski_Distance_Between(vec1, vec2, p=2):
    if(vec1.shape[0] != vec2.shape[0]):
        raise ValueError("Dimension size of the vectors must be same")
    sum = 0
    for a,b in zip(vec1, vec2):
        sum += abs((a-b)**p)
    return sum**(1/p)
    
def Vector_Average(ls, vec_len):
    aver = [] # average vec
    n = len(ls) # n vectors in a cluster
    for i in range(vec_len):
        sum = 0
        for j in range(n):
            sum += ls[j][i]
        sum = sum/vec_len
        aver.append(sum)
    return aver


def normalize(num):
    integer = fl(num)
    num -= integer
    a = abs(0 - num)
    b = abs(0.5 - num)
    c = abs(1.0 - num)
    
    if a<b and a<c:
        return integer
    elif b<c:
        return round(integer + 0.5, 2)
    else:
        return integer + 1
# Testing Purpose
# a1 = np.array([1,1])
# a2 = np.array([-1, -1])
# print(cosine_similarity(a1, a2))