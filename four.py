import nltk
from nltk.corpus import stopwords
import re
from striprtf.striprtf import rtf_to_text
#from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import os
import numpy as np
#from scipy.cluster.hierarchy import linkage
#import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from input import models
tagged_data = []
doc = []
directory = "Dataset/Q1"

for filename in os.listdir(directory):
    if filename.endswith(".rtf"):
        with open(os.path.join(directory, filename), "rb") as file:
            rtf_text = file.read().decode("utf-8")
            # Convert .rtf to plain text
            text = rtf_to_text(rtf_text)
            text = re.sub(r'\[[0-9]*\]',' ',text)
            text = re.sub(r'\s+',' ',text)
            text = re.sub(r'\d+',' ',text)
            text = re.sub(r'\s+',' ',text)
            text = text.lower()
            tokens = nltk.word_tokenize(text)
            words = [word for word in tokens if word.isalpha()]
            words = [word for word in words if word not in stopwords.words("english")]
            cleaned_text = " ".join(words)
            doc.append(cleaned_text)
            # Append a TaggedDocument to the list
            tagged_data.append(TaggedDocument(words=cleaned_text.split(), tags=[str(len(tagged_data))]))

# Create the Doc2Vec model
model = models[0]
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
    print(f"Vector for Document {i + 1}:")
    print(doc_vec)  # Process the inferred vector as needed
    print("\n")
inferred_vectors = np.array(inferred_vectors)
########################################################################################################
# Perform Hierarchical clustering
'''
linked = linkage(inferred_vectors, method='ward', metric='euclidean')
# Plot clustered data points
plt.scatter(inferred_vectors[:, 0], inferred_vectors[:, 1], cmap='rainbow')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()'''
########################################################################################################
# perform DBSCAN clustering
dbscan = DBSCAN(eps=0.0999, min_samples=3)
dbscan.fit(inferred_vectors)

cluster_labels = dbscan.labels_
n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)


for cluster_id in range(n_clusters_):
    print(f"Cluster {cluster_id}:")
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    print(cluster_indices)
outlier_count = 0
for n in cluster_labels:
    if(n==-1):
        outlier_count = outlier_count+1 

unique_labels = set(cluster_labels) - {-1}
midpoints = []
for label in unique_labels:
    cluster_points = inferred_vectors[cluster_labels == label]
    midpoint = np.mean(cluster_points, axis=0)
    midpoints.append(midpoint)
    
overall_midpoint = np.mean(midpoints, axis=0)

print("Midpoint of Clusters:", overall_midpoint)

print("No. of Outliers:", outlier_count)

#########################################################################################################

from sklearn.metrics import silhouette_score

# Assuming doc_vectors and labels are from the DBSCAN clustering step
silhouette_avg = silhouette_score(inferred_vectors, cluster_labels)
print("Silhouette Score:", silhouette_avg)

#########################################################################################################






