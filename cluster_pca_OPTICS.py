import cv2
import numpy as np
"""
emb_array = np.load("pca.res/emb_array_1200.npy")
w = np.load("pca.res/w_st_1200.npy")
my_file = open("pca.res/files.txt", "r")
"""
emb_array = np.load("pca.res/emb_array_wybrane_3600.npy")
w = np.load("pca.res/w_st_13804.npy")
my_file = open("pca.res/files.txt", "r")

file_content = my_file.read()
all_files = file_content.split("\n")

print(w / np.sum(w) * 100)
print(sum(w / np.sum(w) * 100))

img_count = 13804
threshold = 1
p = 50

#for a in range(emb_array.shape[0]):
#    emb_array[a] = emb_array[a] / w

#emb_array_copy = np.copy(emb_array[0:img_count, 3:50])
emb_array_copy = np.copy(emb_array[0:img_count, 0:50])



all_files_names = all_files[0:img_count]
print(emb_array_copy.shape)
"""
distance = np.zeros((img_count, img_count))


for a in range(distance.shape[0]):
  for b in range(a + 1, distance.shape[1]):
      x1 = emb_array_copy[a]
      x2 = emb_array_copy[b]
      distance[b,a] = distance[a,b] = np.linalg.norm(x1 - x2)
aa = 0
aa = 1

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(affinity="precomputed", linkage='average',
                                     distance_threshold=threshold, n_clusters=None).fit(distance)
print(clustering.labels_)
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# plot_dendrogram(clustering, truncate_mode="level", p=p, labels=all_files_names, leaf_rotation=45, leaf_font_size=10)
# plt.show()
"""

"""
print(all_files[0])
print(all_files[1199])
print(all_files[1200])
"""

"""
import matplotlib.pyplot as plt
xxx = emb_array_copy[:,0]
yyy = emb_array_copy[:,1]
plt.scatter(xxx, yyy)
plt.show()
"""

########################################################

from sklearn.cluster import OPTICS
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

db = OPTICS(min_samples=5, cluster_method = 'dbscan', eps=1.8).fit(emb_array_copy)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(emb_array_copy, labels))

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
"""
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = emb_array_copy[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = emb_array_copy[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
"""
result = np.where(labels == -1)
my_ids = result[0]
print(labels)
print(my_ids)
print(len(my_ids))


import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 8))
columns = 5
rows = int(len(my_ids) / 5) + 1


for aaa in range(len(my_ids)):
    #img_help = cv2.imread('d:\\dane\\credo\\png\\' + all_files_names[my_ids[aaa]])
    img_help = cv2.imread('d:\\dane\\credo\\nowe\\wybrane\\' + all_files_names[my_ids[aaa]])

    aaa1 = aaa + 1
    xxxxx = (aaa % 5) + 1
    fig.add_subplot(rows, columns, aaa1)
    plt.imshow(img_help)
    #cv2.imshow(str(aaa), img_help)
    #print(all_files_names[my_ids[0]])

plt.show()


#cv2.waitKey()
#plt.title("Estimated number of clusters: %d" % n_clusters_)
#plt.show()
