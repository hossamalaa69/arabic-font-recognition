import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import os
import pickle
import time
import sys

# this function reads the dataset
def readDataSet():
    data_set = []
    Y = []
    for i in range(1, 10):
        for filename in os.listdir("ACdata_base/" + str(i)):
            img = cv2.imread(os.path.join("ACdata_base/" + str(i),filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                data_set.append(pre_processing(img))
                Y.append(i-1)
    return (data_set, Y)

# this function reads the test dataset
def readTestSet(folder="data"):
    test_set = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            test_set.append(img)
    return test_set

# this function gets the dominant color of image border 
def getBorderColor(img):
    border = np.asarray(img[0,:])
    border = np.concatenate((border, np.asarray(img[-1, :])))
    border = np.concatenate((border, np.asarray(img[:, 0])))
    border = np.concatenate((border, np.asarray(img[:, -1])))
    return np.bincount(border).argmax()


# this function binarize the image and crop it to the text only
def pre_processing(img):
    _, img_binarized = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    if(getBorderColor(img_binarized) != 0):
        img_binarized = cv2.bitwise_not(img_binarized) 
    
    kernel = np.ones((30,30), np.uint8)
    img_dilation = cv2.dilate(img_binarized, kernel, iterations=1)
    cnts = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
    (x,y,w,h) = cv2.boundingRect(cntsSorted[0])
    return img_binarized[y:y+h, x:x+w]    

# this function prints the image
def showImage(img):
    plt.imshow(img, cmap='gray')
    plt.show()    


# copyrights: this function is based on a public repository and we applied some modifications to it 
# github link: https://github.com/onyxe/Multilayer-descriptors-for-medical-image-classification/blob/master/lpq.py
# Local phase quantization descriper
# this function gets the LPQ features from the image into a vector of size 255
def lpq(img):

    # the window size that we will slide the image with
    winSize=3
    # alpha in STFT approaches
    STFTalpha=1/winSize
    
    # Compute descriptor responses only on part that have full neigborhood
    convmode='valid' 
    
    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    w0=np.ones_like(x)
    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
    w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]
    LPQdesc=LPQdesc/LPQdesc.sum()
    
    return LPQdesc


# this function gets the lpq features of one image
def getFeatures(img):
    return lpq(img)

# this function gets the features of array of images and returns
# features matrix, each row is an example i, each column is feature j
def getFeaturesList(images):
   features = []
   for i in range(len(images)):
        features.append(getFeatures(images[i]))
   return np.asarray(features)        