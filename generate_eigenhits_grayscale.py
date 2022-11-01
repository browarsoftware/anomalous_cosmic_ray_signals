# Implementation of Eigenhits
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2022

import cv2
import numpy as np
import os
from random import sample


#path = 'd:\\dane\\credo\\png_align_image\\'
path = 'd:\\dane\\credo\\nowe\\wybrane_align2\\'
path_to_results = "pca.res"
#path_to_results = "pca.res_wybrane_0.1//"
#path_to_results = "pca.res_wybrane_0.25//"
#path_to_results = "pca.res_wybrane_0.5//"
#path_to_results = "pca.res_wybrane_0.75//"
#path_to_results = "pca.res_wybrane_0.95//"

image_files_list = "image_files_list.txt"

files = []
#image_fraction = 0.95
#image_fraction = 0.75
#image_fraction = 0.5
#image_fraction = 0.25
#image_fraction = 0.1
image_fraction = 1
flip_mat = False

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

files = sample(files, int(image_fraction * len(files)))
with open(path_to_results + "//" + image_files_list, 'w') as fp:
    for ff in files:
        fp.write(ff + "\n")

how_many_images = len(files)


img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
old_shape = img.shape
img_flat = img.flatten('F')

T = np.zeros((img_flat.shape[0], len(files)))
for i in range(len(files)):
    if i % 1000 == 0:
        print("\tLoading " + str(i) + " of " + str(len(files)))
    img_help = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
    T[:,i] = img_help.flatten('F') / 255


print('Calculating mean face')
mean_face = T.mean(axis = 1)

for i in range(len(files)):
    T[:,i] -= mean_face


print('Calculating covariance')
if flip_mat:
    C = np.matmul(T.transpose(), T)
else:
    C = np.matmul(T, T.transpose())

C = C / how_many_images

print('Calculating eigenfaces')
from scipy.linalg import eigh
w, v = eigh(C)

if flip_mat:
    v_correct = np.matmul(T, v)
else:
    v_correct = v

sort_indices = w.argsort()[::-1]
w = w[sort_indices]  # puttin the evalues in that order
v_correct = v_correct[:, sort_indices]


norms = np.linalg.norm(v_correct, axis=0)# find the norm of each eigenvector
v_correct = v_correct / norms

#save results
np.save(path_to_results + "//T_st_" + str(how_many_images), T)
np.save(path_to_results + "//v_st_" + str(how_many_images), v_correct)
np.save(path_to_results + "//w_st_" + str(how_many_images), w)
np.save(path_to_results + "//mean_face_st_" + str(how_many_images), mean_face)
np.save(path_to_results + "//norms_st_" + str(how_many_images), norms)
np.save(path_to_results + "//old_shape_st_" + str(how_many_images), np.asarray(old_shape))