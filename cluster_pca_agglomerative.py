import cv2
import numpy as np

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

emb_array_copy = np.copy(emb_array[0:img_count, 0:62])



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
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(distance_threshold = 2.4, n_clusters=None).fit(emb_array_copy)

"""
count_labels = len(set(clustering.labels_))

print(count_labels)
xxx = np.zeros(count_labels)
for a in range(xxx.shape[0]):
    xxx[a] = 
print(clustering.labels_)
"""


from collections import Counter
c=Counter(clustering.labels_)
values_list = list(c.values())
keys_list = list(c.keys())
#print(c.values())
#print(c.keys())

def find_all_elements(my_list, id):
    ret_list = []
    for ind in range(len(my_list)):
        if my_list[ind] == id:
            ret_list.append(ind)
    return ret_list

#xxxx = [1,1,2,3,4,5,1]
#print(find_all_elements(xxxx, 1))

all_cluster_ids = []
all_ids = find_all_elements(values_list, 1)
for a in all_ids:
    all_cluster_ids.append(keys_list[a])
print(all_ids)
print(all_cluster_ids)

my_ids = []
for a in range(len(clustering.labels_)):
    if clustering.labels_[a] in all_cluster_ids:
        my_ids.append(a)
print(my_ids)


import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 8))
columns = 5
rows = int(len(my_ids) / 5) + 1

columns2 = 11
rows2 = int(len(my_ids) / columns2) + 1
ret_img = np.zeros((128 * rows2, 128 * columns2, 3))

xx = 0
yy = 0
for aaa in range(len(my_ids)):
    #img_help = cv2.imread('d:\\dane\\credo\\png\\' + all_files_names[my_ids[aaa]])
    img_help = cv2.imread('d:\\dane\\credo\\nowe\\wybrane\\' + all_files_names[my_ids[aaa]])
    print(all_files_names[my_ids[aaa]])
    aaa1 = aaa + 1

    ret_img[128 * yy : 128 * (yy + 1), 128 * xx : 128 * (xx + 1), :] = cv2.resize(img_help, (128, 128))
    xx = xx + 1

    if xx > columns2 - 1:
        xx = 0
        yy = yy + 1

    xxxxx = (aaa % 5) + 1



    fig.add_subplot(rows, columns, aaa1)
    plt.imshow(img_help)



    #cv2.imshow(str(aaa), img_help)
    #print(all_files_names[my_ids[0]])
ret_img = ret_img.astype(np.uint8)
cv2.imshow("ret_img",ret_img)
plt.show()

""""
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

db = DBSCAN(eps=1.8, min_samples=5).fit(emb_array_copy)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(emb_array_copy, labels))

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
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

result = np.where(labels == -1)
my_ids = result[0]
print(my_ids)

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
"""

#cv2.waitKey()
#plt.title("Estimated number of clusters: %d" % n_clusters_)
#plt.show()