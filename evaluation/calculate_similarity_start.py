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
start_coord = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#start_coord = [0]
#start_coord = [0, 1, 2, 3, 4]
end_coord = [62]
#start_coord = [8]

#min_samples = [3, 4, 5, 6, 7, 8]
min_samples = [5]

#eps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
eps = [1.8]


print(sum(w / np.sum(w) * 100))

#threshold = 1
#p = 50
all_files_names = all_files

output_file = "results_dbscan/clusters"
output_file_results = "results_dbscan/clusters.txt"
all_results = []

lbls = []

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

                    all_results.append(help_results)
                    #lbls.append("start=" + str(my_start_coord) + ",end=" + str(my_end_coord) \
                    #                 + ",eps=" + str(my_eps) + ",min_samples=" + str(my_min_samples))
                    lbls.append(str(my_start_coord))

import matplotlib.pyplot as plt
import numpy as np

#plt.viridis()
#fig, ax = plt.subplots(1)
#plt.viridis()

p = np.zeros((len(all_results), len(all_results)))

#ax.pcolor(np.random.randn((10,10)))
#ax.pcolor(np.random.randn(10), np.random.randn(10))

for a in range(0, len(all_results)):
    for b in range(a + 1, len(all_results)):
        #p[a,b] = OverlapCoefficient(all_results[a], all_results[b])
        #p[a, b] = SorensenCoefficient(all_results[a], all_results[b])
        p[a, b] = JaccardSimilarity(all_results[a], all_results[b])
    #lbls.append(str(a))

#p = ax.pcolormesh(np.random.randn(10,10))
#p = ax.pcolormesh(p)
#fig.colorbar(p)
cm = plt.pcolormesh(p)
plt.colorbar(cm)


plt.xticks(np.arange(0.5, len(lbls) + 0.5), lbls,rotation=90,fontsize=8)
plt.yticks(np.arange(0.5, len(lbls) + 0.5), lbls, fontsize=8)

plt.xlabel("Number of PCA dimensions skipped")
plt.ylabel("Number of PCA dimensions skipped")
#plt.title("Overlap coefficient of Agglomerative and DBSCAN clustering")
#plt.title("Sorensen coefficient of Agglomerative and DBSCAN clustering")
plt.title("Jaccard coefficient of DBSCAN clusterings\n using different features sets")

#for a in range(0, len(all_results)):
#    for b in range(a + 1, len(all_results)):
for a in range(len(all_results)):
    for b in range(len(all_results)):
        if a > b:
            plt.text(a+0.5,(b+0.5),"{:.2f}".format((p[b,a])),
                ha='center',va='center',
                size=8,color='black')

plt.show()

"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1)

np.random.seed(10)

#ax.pcolor(np.random.randn((10,10)))
#ax.pcolor(np.random.randn(10), np.random.randn(10))
p = ax.pcolormesh(np.random.randn(10,10))
fig.colorbar(p)
plt.show()
#https://olgabotvinnik.com/blog/prettyplotlib-painlessly-create-beautiful-matplotlib/
#fig.savefig('pcolormesh_matplotlib_default.png')
"""
