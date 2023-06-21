# Implementation of Eigenhits
# Author: Anonymized
# e-mail: Anonymized
# 2022

import cv2
import numpy as np



#how_many_images = 10353
how_many_images = 13804
#how_many_images = 6902
#how_many_images = 3451
#how_many_images = 1380
#how_many_images = 13113

#path_to_results = "pca.res_wybrane_0.75//"
#path_to_results = "pca.res_wybrane_1.0//"
#path_to_results = "pca.res_wybrane_0.5//"
#path_to_results = "pca.res_wybrane_0.25//"
#path_to_results = "pca.res_wybrane_0.1//"
#path_to_results = "pca.res_wybrane_0.95//"
path_to_results = "pca.res//"

image_files_list = "image_files_list.txt"
path = 'd:\\dane\\credo\\nowe\\wybrane_align2\\'

v_correct = np.load(path_to_results + "//v_st_" + str(how_many_images) + ".npy")
w = np.load(path_to_results + "//w_st_" + str(how_many_images) + ".npy")
mean_face = np.load(path_to_results + "//mean_face_st_" + str(how_many_images) + ".npy")
norms = np.load(path_to_results + "//norms_st_" + str(how_many_images) + ".npy")
old_shape = np.load(path_to_results + "//old_shape_st_" + str(how_many_images) + ".npy")
how_many_images = v_correct.shape[1]

#scale image
def scale(np_i):
    np1 = np.copy(np_i)
    np2 = (np1 - np.min(np1)) / np.ptp(np1)
    return np2


#scale and reshape image for visualization
def scale_and_reshape(np_i, mf, old_shape):
    np1 = np.copy(np_i)
    if mf is None:
        np2 = np1.reshape(old_shape, order='F')
    else:
        np2 = (np1 + mf).reshape(old_shape, order='F')
    np2 = scale(np2)
    return np2

#encode using eigenfaces (steganography)
def encode(data_to_code, carrier_img_i, v, mean_face, message_offset):
    carrier_img = np.copy(carrier_img_i)
    old_shape = carrier_img.shape
    img_flat = carrier_img.flatten('F')
    img_flat -= mean_face
    # generate eigenfaces from carier
    result = np.matmul(v.transpose(), img_flat)
    result_message = result
    ssss = len(result_message)
    result_message[50:len(result_message)] = 0
    # store message in features vector
    #result_message[message_offset:(message_offset + data_to_code.shape[0])] = data_to_code
    # reconstruct carrier image
    reconstruct_message = np.matmul(v, result_message)
    image_to_code2 = scale_and_reshape(reconstruct_message, mean_face, old_shape)
    return image_to_code2

#encode using eigenfaces (steganography)
def embedding(carrier_img_i, v, mean_face):
    carrier_img = np.copy(carrier_img_i)
    old_shape = carrier_img.shape
    img_flat = carrier_img.flatten('F')
    img_flat -= mean_face
    # generate eigenfaces from carier
    result = np.matmul(v.transpose(), img_flat)
    #result_message = result
    #ssss = len(result_message)
    #result_message[50:len(result_message)] = 0
    return result

#path = 'd:\\dane\\credo\\png_align_image\\'
#path = 'd:\\dane\\credo\\nowe\\wybrane_align\\'
#files = []
#how_many_images = 13804
#variance_explained = 0.95

all_embedding = []
all_files = []

with open(path_to_results + "//" + image_files_list, 'r') as fp:
    all_files = fp.readlines()

import os
# r=root, d=directories, f = files
for file in all_files:
    #files.append(os.path.join(r, file))
    full_path = os.path.join(path, str.strip(file))
    img_help = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    off = 0
    embed = embedding(img_help / 255, v_correct, mean_face)
    all_embedding.append(embed)
    #all_files.append(file)
    #img_help = cv2.imread(files[i + offset], cv2.IMREAD_GRAYSCALE)


emb_array = np.zeros((len(all_embedding), len(all_embedding[0])))
a = 0

with open(path_to_results + '//files.txt', 'w') as f:
    for emb in all_embedding:
        emb_array[a, :] = emb
        my_str = all_files[a].strip()
        my_str = os.path.basename(my_str)
        f.write(my_str + "\n")
        a = a + 1
np.save(path_to_results + "//emb_array_wybrane_" + str(how_many_images), emb_array)

