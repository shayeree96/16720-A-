import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    
    #AH=0
    N=x1.shape[0]
    u=x1[:,0].reshape(N,1)
    v=x1[:,1].reshape(N,1)
    
    x=x2[:,0].reshape(N,1)
    y=x2[:,1].reshape(N,1)
   
    #AH=0
    top=np.concatenate((x,y,np.ones((N,1)),np.zeros((N,3)),-np.multiply(x,u),-np.multiply(y,u),-u),axis=1)
    
    down=np.concatenate((np.zeros((N,3)),x,y,np.ones((N,1)),-np.multiply(x,v),-np.multiply(y,v),-1*v),axis=1)
    
    A=np.concatenate((top,down),axis=0)
    
    #To solve for AH=0 we perform singular value decomposition to find the eigen values and the eigen vectors
    
    u,s,vh = np.linalg.svd(A)
    h = vh[-1,:]#Last row will give the homography

    H2to1 = h.reshape(3,3)/vh[-1,-1]
    
    return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
    
    #print('Shape of x1 :',x1.shape)
    #print('Shape of x2 :',x2.shape)
    
    x1_mean=np.mean(x1,axis=0)
    x2_mean=np.mean(x2,axis=0)

	#Shift the origin of the points to the centroid
    #
    x1_shifted=x1-x1_mean#we shift the points to the origin
    x2_shifted=x2-x2_mean
    
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_max=np.amax(abs(x1_shifted),axis=0)
    x2_max=np.amax(abs(x2_shifted),axis=0)
       
    #x1_max=np.expand_dims(x1_max,axis=1)
    #x1_max=x1_max.T
    
    #x2_max=np.expand_dims(x2_max,axis=1)
    #x2_max=x2_max.T
    
    #print('Shape of x1_max:',x1_max.shape)
    #print('Shape of x2_max:',x2_max.shape)

    x1_norm=np.divide(x1_shifted,x1_max)
    x2_norm=np.divide(x2_shifted,x2_max)
    
    #print('Shape of x1_norm:',x1_norm.shape)
    #print('Shape of x2_norm:',x2_norm.shape)

	#Similarity transform 1
    T1=np.array(([1/x1_max[0], 0, -x1_mean[0]/x1_max[0]],[0, 1/x1_max[1], -x1_mean[1]/x1_max[1]],[0,0,1]))

	#Similarity transform 2
    T2=np.array(([1/x2_max[0], 0, -x2_mean[0]/x2_max[0]],[0, 1/x2_max[1], -x2_mean[1]/x2_max[1]],[0,0,1]))

	#Similarity transform 2

	#Compute homography
    H2to1 = computeH(x1_norm, x2_norm)#we compute the homography between the normalized points

	#Denormalization
    H2to1_Denorm=np.matmul(np.linalg.inv(T1),np.matmul(H2to1,T2))
    
    return H2to1_Denorm


def computeH_ransac(locs1, locs2, opts):
    
    '''
    We have use the compute H_norm function later on
    '''
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
    N=locs1.shape[0]
    #print("N is:",N)
    
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    

    match_locs1=np.hstack((locs1,np.ones((N,1))))#u,v,1
    match_locs2=np.hstack((locs2,np.ones((N,1))))#x,y,1

    max_inlier=-1
    for i in range(max_iters):
        idx=np.random.randint(N,size=4)
        p1=locs1[idx]
        p2=locs2[idx]
        
        H=computeH(p1,p2)
        #print("Homo :",H)
        #now that we have found the homography , we can multiply that to locs2 which is destination
        match=np.matmul(H,match_locs2.T)
        #Now we compare this with match_locs1
        #print("Shape of match:",match.shape)
        #print("before match division:",match)
        match=match.T#We reshape 3xN to Nx3
        #print("After Shape of match:",match.shape)
        div=np.expand_dims(match[:,-1],axis=1)
        
        #print('match after division:',match)
        #print('match_locs1',match_locs1)
        #We use the Euclidean distance metric to find the difference
        diff=((match/div)-match_locs1)#So we get N*3
        
        #We normalize the difference
        diff=np.linalg.norm(diff,axis=1)
        
        #print("Shape of difference:",diff.shape)
        
        inlier_calc=np.where(diff<inlier_tol,1,0)
        
        #print('inlier_calc :',inlier_calc)
        #print('tolerance level is:',inlier_tol)
        
        if(np.sum(inlier_calc)>max_inlier):
            max_inlier=np.sum(inlier_calc)
            #print('max values :',max_inlier)
            inliers=inlier_calc
    #print(inliers) 
    ind_use=np.where(inliers==1)
    #print(ind_use[0])
    #print('No of inliers:',ind_use[0].shape)
                
    bestH2to1=computeH(locs1[ind_use[0],:],locs2[ind_use[0],:])
        
    return bestH2to1, inliers

def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.

	#Create mask of same size as template
    mask=np.ones((template.shape[0],template.shape[1],3),dtype=np.uint8)
	#Warp mask by appropriate homography
    warp_mask= cv2.warpPerspective(mask,H2to1,(img.shape[1],img.shape[0]))
    cv2.imwrite('../data/Mask_1.jpg',warp_mask)

	#Warp template by appropriate homography
    warp_template= cv2.warpPerspective(template,H2to1,(img.shape[1],img.shape[0]))
    cv2.imwrite('../data/warp_template.jpg',warp_template)

    inverted_warp_mask=np.where(warp_mask>=1,0,1).astype(dtype=np.uint8)
    
	#Use mask to combine the warped template and the image
    cv_desk_cut=np.multiply(img,inverted_warp_mask)
    composite_img= cv_desk_cut+warp_template

    
    return composite_img


