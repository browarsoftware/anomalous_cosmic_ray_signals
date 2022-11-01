import cv2
import numpy as np
#from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import timeit
from collections import Counter

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
start_coord = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#start_coord = [10, 20]

min_samples = [3, 4, 5, 6, 7, 8]
#eps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
distance_thresholds = [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

print(sum(w / np.sum(w) * 100))

#threshold = 1
#p = 50
all_files_names = all_files

output_file = "results_AgglomerativeClustering/clusters"
output_file_results = "results_AgglomerativeClustering/clusters.txt"
for my_start_coord in start_coord:
    for my_end_coord in end_coord:
        if my_start_coord < my_end_coord:
            for distance_threshold in distance_thresholds:
                emb_array_copy = np.copy(emb_array[:, my_start_coord:my_end_coord])
                ########################################################
                start = timeit.default_timer()



                clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None).fit(emb_array_copy)
                stop = timeit.default_timer()
                elapsed_time = stop - start
                print('Time: ', elapsed_time)

                c = Counter(clustering.labels_)
                values_list = list(c.values())
                keys_list = list(c.keys())


                # print(c.values())
                # print(c.keys())

                def find_all_elements(my_list, id):
                    ret_list = []
                    for ind in range(len(my_list)):
                        if my_list[ind] == id:
                            ret_list.append(ind)
                    return ret_list


                # xxxx = [1,1,2,3,4,5,1]
                # print(find_all_elements(xxxx, 1))

                all_cluster_ids = []
                all_ids = find_all_elements(values_list, 1)
                for a in all_ids:
                    all_cluster_ids.append(keys_list[a])
                #print(all_ids)
                #print(all_cluster_ids)

                my_ids = []
                for a in range(len(clustering.labels_)):
                    if clustering.labels_[a] in all_cluster_ids:
                        my_ids.append(a)
                #print(my_ids)


                #print(my_ids)
                output_file_ok = output_file + ",start=" + str(my_start_coord) + ",end=" + str(my_end_coord) \
                                 + ",distance_threshold=" + str(distance_threshold) + ".txt"
                print(output_file_ok)

                with open(output_file_ok, 'w') as fp:
                    for element in my_ids:
                        fp.write(str(element) + "\n")
                file_object = open(output_file_results, 'a')
                with open(output_file_results, 'a') as fp:
                    fp.write(str(my_start_coord) + "," + str(my_end_coord) + "," + str(distance_threshold) + ","
                                  + str(len(my_ids)) + "," + str(elapsed_time) + "\n")
