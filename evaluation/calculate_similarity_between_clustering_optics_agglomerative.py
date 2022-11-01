# Implementation of eigenhits library
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2022

def JaccardSimilarity(A, B):
    div = len(set(A).union(B))
    if div > 0:
        return len(set(A).intersection(B)) / div
    else:
        return 0

def SorensenCoefficient(A, B):
    div = len(A) + len(B)
    if div > 0:
        return 2 * len(set(A).intersection(B)) / div
    else:
        return 0

def OverlapCoefficient(A, B):
    div = min(len(A),len(B))
    if div > 0:
        return len(set(A).intersection(B)) / div
    else:
        return 0

#porónać cluster start i cluster stop

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
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
#start_coord = [0, 1, 2, 3, 4, 5, 10, 20]
#start_coord = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#start_coord = [0, 1, 2, 3, 4]
end_coord = [62]
start_coord = [0]

#min_samples = [3, 4, 5, 6, 7, 8]
min_samples = [8]

#eps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
eps = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#eps = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#eps = [1.8]


print(sum(w / np.sum(w) * 100))

#threshold = 1
#p = 50
all_files_names = all_files

#output_file = "results_dbscan/clusters"
#output_file_results = "results_dbscan/clusters.txt"
output_file = "results_OPTICS/clusters"
output_file_results = "results_OPTICS/clusters.txt"
all_results1 = []

lbls1 = []

for my_start_coord in start_coord:
    for my_end_coord in end_coord:
        if my_start_coord < my_end_coord:
            for my_eps in eps:
                for my_min_samples in min_samples:
                    emb_array_copy = np.copy(emb_array[:, my_start_coord:my_end_coord])
                    ########################################################

                    output_file_ok = output_file + ",start=" + str(my_start_coord) + ",end=" + str(my_end_coord) \
                                     + ",eps=" + str(my_eps) + ",min_samples=" + str(my_min_samples) + ".txt"
                    help_results = []
                    with open(output_file_ok, 'r') as fp:
                        Lines = fp.readlines()
                        for line in Lines:
                            help_results.append(int(line))

                    all_results1.append(help_results)
                    #lbls.append("start=" + str(my_start_coord) + ",end=" + str(my_end_coord) \
                    #                 + ",eps=" + str(my_eps) + ",min_samples=" + str(my_min_samples))
                    lbls1.append(str(my_eps))


output_file2 = "results_AgglomerativeClustering/clusters"
output_file_results2 = "results_AgglomerativeClustering/clusters.txt"
all_results2 = []

lbls2 = []
distance_thresholds = [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
#distance_thresholds = [2.3]

for my_start_coord in start_coord:
    for my_end_coord in end_coord:
        if my_start_coord < my_end_coord:
            for distance_threshold in distance_thresholds:
                    emb_array_copy = np.copy(emb_array[:, my_start_coord:my_end_coord])
                    ########################################################

                    output_file_ok = output_file2 + ",start=" + str(my_start_coord) + ",end=" + str(my_end_coord) \
                                     + ",distance_threshold=" + str(distance_threshold) + ".txt"
                    help_results = []
                    with open(output_file_ok, 'r') as fp:
                        Lines = fp.readlines()
                        for line in Lines:
                            help_results.append(int(line))

                    all_results2.append(help_results)
                    #lbls.append("start=" + str(my_start_coord) + ",end=" + str(my_end_coord) \
                    #                 + ",eps=" + str(my_eps) + ",min_samples=" + str(my_min_samples))
                    lbls2.append(str(distance_threshold))

import matplotlib.pyplot as plt
import numpy as np

#plt.viridis()
#fig, ax = plt.subplots(1)
#plt.viridis()

p = np.zeros((len(all_results1), len(all_results2)))

#ax.pcolor(np.random.randn((10,10)))
#ax.pcolor(np.random.randn(10), np.random.randn(10))

for a in range(0, len(all_results1)):
    #for b in range(a + 1, len(all_results2)):
    for b in range(0, len(all_results2)):
        p[a,b] = OverlapCoefficient(all_results1[a], all_results2[b])
        #p[a, b] = SorensenCoefficient(all_results1[a], all_results2[b])
        #p[a, b] = JaccardSimilarity(all_results1[a], all_results2[b])
    #lbls.append(str(a))

#p = ax.pcolormesh(np.random.randn(10,10))
#p = ax.pcolormesh(p)
#fig.colorbar(p)
cm = plt.pcolormesh(p)
plt.colorbar(cm)



plt.yticks(np.arange(0.5, len(lbls1) + 0.5), lbls1,fontsize=8)
plt.xticks(np.arange(0.5, len(lbls2) + 0.5), lbls2, rotation=90, fontsize=8)

plt.xlabel("Agglomerative clustering distance threshold")
plt.ylabel("OPTICS eps")
plt.title("Overlap coefficient of Agglomerative and OPTICS clustering")
#plt.title("Sorensen coefficient of Agglomerative and OPTICS clustering")
#plt.title("Jaccard coefficient of Agglomerative and OPTICS clustering")

#for a in range(0, len(all_results)):
#    for b in range(a + 1, len(all_results)):
for a in range(len(all_results1)):
    for b in range(len(all_results2)):
        #if a > b:
            plt.text(b+0.5,(a+0.5),"{:.2f}".format((p[a,b])),
                ha='center',va='center',
                size=8,color='black')

plt.show()
