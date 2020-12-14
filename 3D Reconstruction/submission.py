"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import helper
import sympy
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import os.path
import util
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    
    pl = pts1*1.0/M
    pr = pts2*1.0/M
    
    xl=pl[:,0]#N*1
    yl=pl[:,1]#N*1
    
    xr=pr[:,0]#N*1
    yr=pr[:,1]#N*1
    
    U=np.vstack((xr*xl,xr*yl,xr,yr*xl,yr*yl,yr,xl,yl,np.ones(pts1.shape[0])))
    U=U.T
    u,s,vh=np.linalg.svd(U)
    F=vh[-1,:]
    F=F.reshape(3,3)
    #You must enforce the singularity condition of the F before unscaling
    
    #refine the solution by using local minimization
    F = util._singularize(F)
    F = util.refineF(F,pl,pr)
    
    #now we need to unnormalize F
    T=np.array([[1./M,0,0],[0,1./M,0],[0,0,1]])
    F=T.T @ F @ T
    
    #print("F1 is:",F)
    if(os.path.isfile('q2_1.npz')==False):
        np.savez('q2_1.npz',F = F, M = M)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1 
    
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    n, temp = pts1.shape
    P = np.zeros((n,3)) 
    P_homo = np.zeros((n,4))
    err=0
    for i in range(n):#For n no of correspondences
        x1=pts1[i,0]
        y1=pts1[i,1]
        x2=pts2[i,0]
        y2=pts2[i,1]
        
        A=np.vstack((((y1 * C1[2,:])- C1[1,:]),(C1[0,:]- (x1 * C1[2,:])), 
                      ((y2 * C2[2,:])-C2[1,:]),(C2[0,:]-(x2 *C2[2,:]))))
        #Now we find the SVD
        u, s, vh = np.linalg.svd(A)
        p=vh[-1, :]#We take the last column 1*4
        p=p/p[-1]#We divide to get the non homo
        P[i,:]=p[:3]# ( X,Y,Z )
        P_homo[i,:]=p#( X,Y,Z,1 )
        
    pts1_proj=C1 @ P_homo.T #This will be 3x4 @ 4 x N
    pts2_proj=C2 @ P_homo.T #This will be 3x4 @ 4 x N
    
    #We need to convert this to non homo
    pts1_proj=pts1_proj/pts1_proj[-1,:]#Shape is 3XN
    pts2_proj=pts2_proj/pts2_proj[-1,:]#Shape is 3XN   
    
    err=np.sum((pts1_proj[:2,:].T-pts1)**2)+ np.sum((pts2_proj[:2,:].T-pts2)**2)
          

    return P,err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    
    pt1 = np.array([[x1],[y1],[1]])
    data = np.load('../data/some_corresp.npz')
    eplipolar_line = F.dot(pt1)
    
    a = eplipolar_line [0]
    b = eplipolar_line [1]
    c = eplipolar_line [2]

    H,W,_= im1.shape
    
    range_width=30
    
    liney = np.arange(y1-range_width,y1+range_width)
    linex = (-(b*liney+c)/a)
    window = 5
    
    im1g = ndimage.gaussian_filter(im1, sigma=0.5)
    im2g = ndimage.gaussian_filter(im2, sigma=0.5)
    
    patch_im1 = im1g[y1-window:y1+window+1, x1-window:x1+window+1,:]
    minerr = np.inf
    ele = 0
    
    for i in range(range_width*2):
        x2 = int(linex[i])
        y2 = liney[i]
        
        if (x2>=window  and x2<= W-window-1 and y2>=window and y2<= H-window-1):
            patch_im2 = im2g[y2-window:y2+window+1, x2-window:x2+window+1,:]
            err = np.sum((patch_im1-patch_im2).flatten()**2)
            
            if (err<minerr):
                minerr = err
                ele = i
                               
    return linex[ele],liney[ele]

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=200, tol=2e-3):
    # Replace pass by your implementation
    print("In ransac")
    num_inlier=0
    for i in range(nIters):
        print("Iterations :",i)
        idx = np.random.choice(pts1.shape[0],8)
        pts1_ext = pts1[idx,:]
        pts2_ext = pts2[idx,:]
        F = eightpoint(pts1_ext, pts2_ext, M)
    
        inliers_max = []
        for j in range(pts1.shape[0]):
            p2 = np.append(pts2[j,:],1).reshape((1,3))
            p1 = np.append(pts1[j,:],1).reshape((3,1))
            err = abs(p2 @ F @ p1)
            #print("error :",err)
            if (err<tol):
                inliers_max.append(j)
        if(len(inliers_max) > num_inlier):
            num_inlier = len(inliers_max)
            inliers = inliers_max       
    
    inliers1 = pts1[inliers,:]
    inliers2 = pts2[inliers,:]

    F = eightpoint(inliers1, inliers2, M)
    # NOTE HERE! it will write a new q2_1.npz to replace the previous on

    return F,inliers
'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    r = r.flatten()
    ang = np.linalg.norm(r)
    if ang == 0: return np.eye(3)
    u = np.matrix(r/ang).T
    v = np.asarray([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = np.eye(3)*np.cos(ang) + (1 - np.cos(ang))*(u @ u.T) + v*np.sin(ang)
    
    return np.asarray(R)

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T)/2
    p = np.asarray([A[2, 1], A[0, 2], A[1, 0]])
    s = np.linalg.norm(p)
    c = (np.trace(R) - 1)/2
    I=np.eye(3)
    if s == 0 and c == 1: 
        return np.asarray([0, 0, 0])
    elif s == 0 and c == -1: 
        i = np.where((R + I).any(axis=0))[0]
        v = (R + I)[:, i[0]]
        u = v/np.linalg.norm(v)
        r = u * np.pi
        return -r if (np.linalg.norm(r) == np.pi) and ((r[0]==0 and r[1]==0 and r[2]<0) or (r[0]==0 and r[1]<0) or (r[0]<0)) else r
    else:
        u = p/s
        ang = np.arctan2(s, c)
        return u * ang
       
'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    
    M2 = np.hstack((rodrigues(x[-6:-3]), x[-3:].reshape(3, 1)))
    P = np.hstack((x[:-6].reshape(p1.shape[0], 3), np.ones((p1.shape[0], 1))))

    p1_hat = (K1 @ M1 @ P.T).T
    p2_hat = (K2 @ M2 @ P.T).T
    p1_hat = p1_hat/p1_hat[:, -1].reshape(p1.shape[0], 1)
    p2_hat = p2_hat/p2_hat[:, -1].reshape(p2.shape[0], 1)

    residuals = np.concatenate([(p1-p1_hat[:, :2]).reshape([-1]), (p2-p2_hat[:, :2]).reshape([-1])])
    return residuals
    
'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    
    R2, t2 = M2_init[:, :3], M2_init[:, 3]
    x0 = np.hstack((P_init.flatten(), invRodrigues(R2).flatten(), t2.flatten()))
    
    # Optimize least squares
    fun = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    res = optimize.least_squares(fun, x0)

    M2 = np.hstack((rodrigues(res.x[-6:-3]), res.x[-3:].reshape(3, 1)))
    P2 = res.x[:-6].reshape(p1.shape[0], 3)

    return M2, P2

    
