'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission as sub
import helper
import matplotlib.pyplot as plt
import os.path

def findM2 ():

    data = np.load('../data/some_corresp.npz')
    # data = np.load('../data/some_corresp_noisy.npz')
    
    Ks = np.load('../data/intrinsics.npz')
    K1 = Ks['K1']
    K2 = Ks['K2']
    pts1 = data['pts1']
    pts2 = data['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(np.shape(im1))
    
    F = sub.eightpoint(data['pts1'], data['pts2'], M)
    # using RANSAC to find F
    # F,inliers = sub.ransacF(data['pts1'], data['pts2'], M)
    #print('F is : ',F)
    
    #p1,p2=helper.epipolarMatchGUI(im1, im2, F)
    
    
    E = sub.essentialMatrix(F, K1, K2)
    print('E is : ',E)
    
    M1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
    M2s = helper.camera2(E)
    row, col, num = np.shape(M2s)
    
    C1 = np.matmul(K1, M1)
    # minerr = np.inf
    # res = 0
    for i in range(num):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = sub.triangulate(C1, pts1, C2, pts2) 
        if (np.all(P[:,2] > 0)) :
            break
    #print("P is :",P)
    
    if(os.path.isfile('q3_3.npz')==False):
        np.savez('q3_3.npz', M2 = M2, C2 = C2, P = P)
    return M1,C1,M2,C2,F


if __name__ == '__main__':
    findM2()

    
