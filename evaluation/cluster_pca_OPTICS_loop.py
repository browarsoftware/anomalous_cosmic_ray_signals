# Implementation of Eigenhits
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2022

import cv2
import numpy as np
#from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import timeit

from sklearn import metrics

emb_array = np.load("../pca.res_wybrane/emb_array_wybrane_13804.npy")
w = np.load("../pca.res_wybrane/w_st_13804.npy")
v = np.load("../pca.res_wybrane/v_st_13804.npy")
my_file = open("../pca.res_wybrane/files.txt", "r")

file_content = my_file.read()
all_files = file_content.split("\n")

eig_cumulative = np.copy(w) / np.sum(w)
for a in range(1,eig_cumulative.shape[0]):
    eig_cumulative[a] += eig_cumulative[a - 1]
end_coord = []
a = 0
while eig_cumulative[a] < 0.75:
    a = a + 1
end_coord.append(a)
while eig_cumulative[a] < 0.8:
    a = a + 1
end_coord.append(a)
while eig_cumulative[a] < 0.85:
    a = a + 1
end_coord.append(a)
while eig_cumulative[a] < 0.9:
    a = a + 1
end_coord.append(a)
while eig_cumulative[a] < 0.95:
    a = a + 1
end_coord.append(a)


for a in range(1, eig_cumulative.shape[0]):
    eig_cumulative[a] += eig_cumulative[a - 1]
start_coord = [0, 1, 2, 3, 4, 5]
#start_coord = [10, 20]

min_samples = [3, 4, 5, 6, 7, 8]
#eps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
eps = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


print(sum(w / np.sum(w) * 100))

#threshold = 1
#p = 50
all_files_names = all_files

output_file = "results_OPTICS/clusters"
output_file_results = "results_OPTICS/clusters.txt"
for my_start_coord in start_coord:
    for my_end_coord in end_coord:
        if my_start_coord < my_end_coord:
            for my_eps in eps:
                for my_min_samples in min_samples:
                    emb_array_copy = np.copy(emb_array[:, my_start_coord:my_end_coord])
                    ########################################################
                    start = timeit.default_timer()
                    #db = DBSCAN(eps=my_eps, min_samples=my_min_samples).fit(emb_array_copy)
                    db = OPTICS(min_samples=my_min_samples, cluster_method='dbscan', eps=my_eps).fit(emb_array_copy)

                    stop = timeit.default_timer()
                    elapsed_time = stop - start
                    print('Time: ', elapsed_time)

                    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                    #core_samples_mask[db.core_sample_indices_] = True
                    labels = db.labels_

                    # Number of clusters in labels, ignoring noise if present.
                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise_ = list(labels).count(-1)

                    result = np.where(labels == -1)
                    my_ids = result[0]
                    #print(my_ids)
                    output_file_ok = output_file + ",start=" + str(my_start_coord) + ",end=" + str(my_end_coord) \
                                     + ",eps=" + str(my_eps) + ",min_samples=" + str(my_min_samples) + ".txt"
                    print(output_file_ok)

                    with open(output_file_ok, 'w') as fp:
                        for element in my_ids:
                            fp.write(str(element) + "\n")
                    file_object = open(output_file_results, 'a')
                    with open(output_file_results, 'a') as fp:
                        fp.write(str(my_start_coord) + "," + str(my_end_coord) + "," + str(my_eps) + ","
                                      + str(my_min_samples) + "," + str(n_clusters_) + "," + str(len(my_ids)) +
                                 "," + str(elapsed_time) + "\n")
