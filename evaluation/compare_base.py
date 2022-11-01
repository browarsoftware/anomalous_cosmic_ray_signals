import math

import numpy as np
def planar_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    if math.isnan(angle):
        angle = 0
    return angle

def correct_vector(v1):
    v_help = np.copy(v1)
    max_val = np.max(v_help)
    min_val = np.min(v_help)
    if np.abs(min_val) > np.abs(max_val) and min_val < 0:
        v_help *= -1
    return v_help

how_many_images1 = 13804
how_many_images075 = 13113
how_many_images075 = 10353
how_many_images075 = 6902
how_many_images075 = 3451
how_many_images075 = 1380

path_to_results1 = "pca.evaluation/pca.res_wybrane_1.0//"
path_to_results075 = "pca.evaluation/pca.res_wybrane_0.95_0//"
path_to_results075 = "pca.evaluation/pca.res_wybrane_0.75_0//"
path_to_results075 = "pca.evaluation/pca.res_wybrane_0.5_0//"
path_to_results075 = "pca.evaluation/pca.res_wybrane_0.25_0//"
path_to_results075 = "pca.evaluation/pca.res_wybrane_0.1_0//"



v_correct1 = np.load("../" + path_to_results1 + "/v_st_" + str(how_many_images1) + ".npy")
#v_correct075 = np.load("../" + path_to_results1 + "/v_st_" + str(how_many_images1) + ".npy")
v_correct075 = np.load("../" + path_to_results075 + "/v_st_" + str(how_many_images075) + ".npy")

w_correct1 = np.load("../" + path_to_results1 + "/w_st_" + str(how_many_images1) + ".npy")
#w_correct075 = np.load("../" + path_to_results1 + "/w_st_" + str(how_many_images1) + ".npy")
w_correct075 = np.load("../" + path_to_results075 + "/w_st_" + str(how_many_images075) + ".npy")

w_correct1_scaled = w_correct1 / np.sum(w_correct1)
w_correct075_scaled = w_correct075 / np.sum(w_correct075)

#print(np.sum(w_correct1_scaled))
#print(np.sum(w_correct075_scaled))

#print(np.sum((w_correct1_scaled + w_correct075_scaled) / 2.0))

sum_angle_res100 = 0
sum_angle_res95 = 0
sum_angle_res90 = 0
sum_angle_res85 = 0
sum_angle_res80 = 0
sum_angle_res75 = 0
cumulatice_sum = 0

for a in range(v_correct1.shape[1]):
    v_help1 = correct_vector(v_correct1[:, a])
    v_help2 = correct_vector(v_correct075[:, a])
    angle_res = planar_angle(v_help1, v_help2)
    average_eigen = (w_correct1_scaled[a] + w_correct075_scaled[a]) / 2
    cumulatice_sum += average_eigen
    sum_angle_res100 += average_eigen * angle_res
    if cumulatice_sum < 0.95:
        sum_angle_res95 += average_eigen * angle_res
    if cumulatice_sum < 0.9:
        sum_angle_res90 += average_eigen * angle_res
    if cumulatice_sum < 0.85:
        sum_angle_res85 += average_eigen * angle_res
    if cumulatice_sum < 0.8:
        sum_angle_res80 += average_eigen * angle_res
    if cumulatice_sum < 0.75:
        sum_angle_res75 += average_eigen * angle_res

print(sum_angle_res100 / math.pi)
print(sum_angle_res95 / math.pi)
print(sum_angle_res90 / math.pi)
print(sum_angle_res85 / math.pi)
print(sum_angle_res80 / math.pi)
print(sum_angle_res75 / math.pi)