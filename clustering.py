import numpy as np
from sklearn.cluster import DBSCAN

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
