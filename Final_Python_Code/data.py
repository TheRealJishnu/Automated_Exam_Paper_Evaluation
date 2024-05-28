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

'''
Q1: 1-105 --> 001-105 --> 1-20
Q2: 1-112 --> 106-217 --> 106-126
Q3: 1-106 --> 218-323 --> 218-238
Q4: 1-119 --> 324-442 --> 324-344
Q5: 1-108 --> 443-550 --> 443-463
'''
tagged_train = []
tagged_data = []
doc = []
# directory = "Dataset/Q1"
q_list = ['1', '2', '3', '4', '5']
directory = r"D:\Internship\Automated_Exam_Paper_Evaluation\Dataset\Q" # Jishnu


for q in q_list:
    # lim=0
    direc = directory + q
    for filename in os.listdir(direc):
        # if lim >= 20:
        #     break
        if filename.endswith(".rtf"):
            with open(os.path.join(direc, filename), "rb") as file:
                rtf_text = file.read().decode("utf-8")
                # Convert .rtf to plain text
                text = rtf_to_text(rtf_text)
                text = re.sub(r'\[[0-9]*\]',' ',text)
                text = re.sub(r'\s+',' ',text)
                text = re.sub(r'\d+',' ',text)
                text = re.sub(r'\s+',' ',text)
                text = text.lower()
                tokens = nltk.word_tokenize(text)
                # print(tokens)
                words = [word for word in tokens if word.isalpha()]
                words = [word for word in words if word not in stopwords.words("english")]
                cleaned_text = " ".join(words)
                doc.append(cleaned_text)
                # Append a TaggedDocument to the list
                tagged_data.append(TaggedDocument(words=cleaned_text.split(), tags=[str(len(tagged_data))]))

 
# tagged_train = []
# tagged_train.extend(tagged_data[0:20])
# tagged_train.extend(tagged_data[105:125])
# tagged_train.extend(tagged_data[217:237])
# tagged_train.extend(tagged_data[323:343])
# tagged_train.extend(tagged_data[442:462])


