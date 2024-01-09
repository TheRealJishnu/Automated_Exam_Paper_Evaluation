from gensim.models import Doc2Vec
import data
import vectors
import clustering
from sklearn.metrics import silhouette_score

vector_list = []
cluster_list = []
tagged_data = data.tagged_data

for i in range(2, 21):
    model = Doc2Vec(vector_size=i, window=5, min_count=1, dm=0, epochs=10)
    vector_list.append(vectors.vecs(model))
    cluster_labels, outlier_n, n_clusters = clustering.clustering(vector_list[i-2])
    cluster_list.append(cluster_labels)
    silhouette_avg = silhouette_score(vector_list[i-2], cluster_labels)
    print(f"Dimension {i}: Silhouette Score {silhouette_avg} and No. of Outliers {outlier_n}")
    print(f"No. of Clusers {n_clusters}")
