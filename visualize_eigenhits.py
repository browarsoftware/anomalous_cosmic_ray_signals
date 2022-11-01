import numpy as np
v_correct = np.load("./pca.res/v_st_13804.npy")
w = np.load("./pca.res/w_st_13804.npy")
mean_face = np.load("./pca.res/mean_face_st_13804.npy")
norms = np.load("./pca.res/norms_st_13804.npy")
old_shape = np.load("./pca.res/old_shape_st_13804.npy")

import cv2

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

mf = scale_and_reshape(mean_face, None, old_shape)
mf = cv2.resize(mf, (256, 256))
#cv2.imshow("mf",mf)

rys = np.zeros((3 * 256, 4 * 256))
rys[0:256, 0:256] = mf
xx = 1
yy = 0

for a in [0, 1, 2, 3, 4, 5, 11, 15, 21, 33, 62]:
    ef0 = scale_and_reshape(v_correct[:,a], mean_face, old_shape)
    ef0 = cv2.resize(ef0, (256, 256))

    rys[(256 * yy):(256 * (yy + 1)), (256 * xx):(256 * (xx + 1))] = ef0
    xx = xx + 1
    if xx % 4 == 0:
        xx = 0
        yy = yy + 1


    #cv2.imshow("ef" + str(a),ef0)

cv2.imshow("mf",rys)

cv2.waitKey()