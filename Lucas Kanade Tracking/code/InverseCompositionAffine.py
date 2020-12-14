import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """
    
    #We have to find M 
    #We precompute the Jacobian
    #We have to get the initializations
    p=np.zeros(6)
    dp=np.ones(6)#Since we have six parameters to be determined
    
    #We hvae to initialize M
    M = np.eye(3)
    
    row_1,col_1=It.shape
    row_2,col_2=It.shape
    
    imH0, imW0 = np.shape(It)
    imH1, imW1 = np.shape(It1)
    
    splinet=RectBivariateSpline(np.linspace(0,row_1,row_1),np.linspace(0,col_1,col_1),It) #For image 1
    splinet1=RectBivariateSpline(np.linspace(0,row_2,row_2),np.linspace(0,col_2,col_2),It1)#For image 2
    
    Iy,Ix=np.gradient(It)# This we find out the Affine subtraction
    spline_x=RectBivariateSpline(np.linspace(0,row_1,row_1),np.linspace(0,col_1,col_1),Ix)#For image 2
    spline_y=RectBivariateSpline(np.linspace(0,row_1,row_1),np.linspace(0,col_1,col_1),Iy)#For image 2
    
    #we have to get all the coordinates for the template image
    
    x,y=np.mgrid[0:col_1,0:row_1]
    #print("Shape od x and y:",x.shape,y.shape)
    x_coor=np.reshape(x,(1,-1))
    y_coor=np.reshape(y,(1,-1))
    
    #We make [x,y,1]
    coor=np.vstack((x_coor,y_coor,np.ones((1,row_1*col_1))))#
    
    grad_x=spline_x.ev(y,x).flatten()
    grad_y=spline_y.ev(y,x).flatten()
    
    T = splinet.ev(y, x).flatten()
   
    
    A1=np.multiply(grad_x,x_coor)
    A2=np.multiply(grad_x,y_coor)
    A3=np.reshape(grad_x,(1,-1))
    A4=np.multiply(grad_y,x_coor)
    A5=np.multiply(grad_y,y_coor)
    A6=np.reshape(grad_y,(1,-1))
    
    A = np.vstack((A1,A2,A3,A4,A5,A6)) #this is the Jaconian and the gradient of I
    A=A.T
    
    #print("Shape of A in my program:",A.shape)
    #print("Shape of b:",error.shape)
    
    H=A.T@A#We calculate the Hessian
    n=1
    
    while(np.square(dp).sum()>threshold and n<num_iters):        
        
        M=np.array([[1+p[0],p[1],p[2]],[p[3],1+p[4],p[5]],[0,0,1]])#Dimension i
        #print("Shape of M:",M.shape)
        #print("Shape of coor :",coor.shape)
        #We have to find the waroed image and hence we have to multiply m with x
        warp=M@coor#This is  3*N  
        
        #now we have the xp and yp coordinates
        warp_x=warp[0]
        warp_y=warp[1]
        
        #Now we have to find the gradient splines
        warp_final = splinet1.ev(warp_y,warp_x).flatten()
        
        #We now find the error image
        error=np.reshape(T-warp_final,(len(warp_x),1))#
        #b = np.reshape(Itp - It1p, (len(xp), 1))
        
        dp=np.linalg.inv(H) @ A.T @ error # 2x2 @ 2xn @ nx1 --> Results in 2x1
        #print("Shape of dp:",dp.shape)
        p = (p + dp.T).ravel()
        
        n+=1
        
        #print("Shape of dp:",dp.shape)
        dM = np.vstack((dp.reshape(2,3), [0, 0, 1]))
        
        M = M @ np.linalg.inv(dM)
    
    return M
            
