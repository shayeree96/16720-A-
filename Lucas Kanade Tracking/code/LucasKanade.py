import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect,threshold,num_iters, p0 = np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    
    #print("im here")
    #threshold=0.01#1.
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]#2.
    row,column=It.shape#3.
    row_rec=y2-y1
    col_rec=x2-x1
    
    # Put your implementation here
    #We have two images and first we are going to take the coordinates for the both  
    
    y=np.arange(0,row,1)#6.
    x=np.arange(0,column,1)#7.
    rr,cc=np.meshgrid(np.linspace(y1,y2,row_rec),np.linspace(x1,x2,col_rec))
    
    splinet=RectBivariateSpline(y,x,It)
    splinet1=RectBivariateSpline(y,x,It1)
    T=splinet.ev(rr,cc)
    
    Iy,Ix=np.gradient(It1)#5.
    spline_x=RectBivariateSpline(y,x,Ix)
    spline_y=RectBivariateSpline(y,x,Iy)
    
    J=np.array([[1,0],[0,1]])
    
    dp=[[1],[1]]
    n=1
    
    while(np.square(dp).sum()>threshold and n<num_iters):
        x1_dp=x1+p0[0]
        y1_dp=y1+p0[1]
        x2_dp=x2+p0[0]
        y2_dp=y2+p0[1]
        
        #Now we find the grid mesh
        rr_new ,cc_new= np.meshgrid(np.linspace(y1_dp, y2_dp,row_rec), np.linspace(x1_dp, x2_dp, col_rec))
        
        #cc_new,rr_new=np.meshgrid(np.linspace(x1_dp,x2_dp,col_rec),np.linspace(y1_dp,y2_dp,row_rec))
        
        warp=splinet1.ev(rr_new,cc_new)
        #print("Shape of warp:",warp.shape)
        
        #Now we calculate the error
        #print("Shape of T:",T.shape)
        e=T-warp #print shape #This should be of shape nx1
        
        e_Image=e.reshape(-1,1)
        #print("Shape of error :",e.shape)
        
        #Now we apply the formulas
        grad_x=spline_x.ev(rr_new,cc_new)
        grad_y=spline_y.ev(rr_new,cc_new)
        
        I_grad=np.hstack((grad_x.reshape(-1,1),grad_y.reshape(-1,1)))#Here we get the shape nx2
        
        #We calculate the Hessian
        mul=I_grad @ J# this is shape nx2
        
        H=mul.T @ mul # This is shape 2x2
        
        dp=np.linalg.inv(H) @ mul.T @ e_Image # 2x2 @ 2xn @ nx1 --> Results in 2x1
        
        p0[0]+=dp[0,0]
        p0[1]+=dp[1,0] 
        n+=1

    return p0