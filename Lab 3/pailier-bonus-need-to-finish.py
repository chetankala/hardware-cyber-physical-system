# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:02:35 2022

@author: wang
"""


import numpy as np
from phe import paillier
import time


#Note that we do not have zero padding here for simplicity. 
#This function performs Conv in 2D
def conv2d(X, H):
   # make sure both X and H are 2-D
   assert( X.ndim == 2)
   assert( H.ndim == 2)
    
   # get the horizontal and vertical size of X and H
   imageColumns = X.shape[1]
   imageRows = X.shape[0]
   kernelColumns = H.shape[1]
   kernelRows = H.shape[0]

   # calculate the horizontal and vertical size of Y 
   newRows = imageRows - kernelRows + 1
   newColumns = imageColumns - kernelColumns + 1

   # create an empty output array
   Y = np.zeros((newRows,newColumns))

   start_time = time.time()
   # go over output locations
   for m in range(newRows):
       for n in range(newColumns):

    # go over input locations
                Y[m,n] = np.sum(H*X[m:m + kernelRows, n:n + kernelColumns])
                         
        # make sure kernel is within bounds
        
        # calculate the convolution sum

   print("Plaintext execution time --- %s seconds ---" % (time.time() - start_time))
   return Y



#Note that we do not have zero padding and 180 degree flip here for simplicity. 
#we implement a 5x5 image conv with 3x3 kernel, the output will be 3x3. 
kernel = np.array([[1, 0, 1], [0, 1, 0], [0,0,1]])

P = np.array([[5,0,3,1,1], [2,3,13,0,6], [5,3,15,20,11], [8,3,18,4,8], [4,2,6,5,7], [4,2,6,5,7], [4,2,6,5,7]])


###You need to write sec_conv2d$$$$
#result_plaintext = sec_conv2d(P,kernel)


#print("Plaintext Result is \n {}".format(result_plaintext))


public_key, private_key = paillier.generate_paillier_keypair()

b = [public_key.encrypt(i) for i in P.flatten().tolist()]

b_encrypted_list = [b[i:i+5] for i in range(0, len(b), 5)]


X = P
H = kernel

imageColumns = X.shape[1]
imageRows = X.shape[0]
kernelColumns = H.shape[1]
kernelRows = H.shape[0]

#Y is the encrypted list, initialize to empty
Y = []




# write your own implementation of secure convolution here
# start_time = time.time()
# for m in range(newRows):
#        for n in range(newColumns):
        
#            for i in range(kernelRows):
#                for j in range(kernelColumns):
#                    Y = np.sum(kernel * b_encrypted_list ())      ----> make this step into Paillier compatible 
                        

# Output the encrypted list Y, and transform that to the correct output shape
# decrypt that with d = [private_key.decrypt(i) for i in Y]
#print("Plaintext Result is \n {}".format([d[i:i+newRows] for i in range(0, len(d), newRows)]))
#print("Plaintext execution time --- %s seconds ---" % (time.time() - start_time))



