import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d
import torch

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


# Local phase quantization descriper
# this function gets the LPQ features from the image into a vector of size 255
def lpq(img, winSize=3, freqestim=1, mode='nh'):
    

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).
    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
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

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
        LPQdesc=LPQdesc/LPQdesc.sum()
    
    return LPQdesc


# this function gets the features of array of images and returns
# features matrix, each row is an example i, each column is feature j
def getFeaturesList(images):
   features = []
   for i in range(len(images)):
        features.append(lpq(images[i]))
   return np.asarray(features)    


# holds the whole dataset
data_set = []
# holds the whole dataset labels
Y = []
for i in range(1, 10):
    for filename in os.listdir("ACdata_base/" + str(i)):
        img = cv2.imread(os.path.join("ACdata_base/" + str(i),filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            data_set.append(pre_processing(img))
            Y.append(i-1)

# divide the dataset into 60% training, 40% test_validation
X_train, X_testValid, Y_train, Y_testValid = train_test_split(data_set, Y, test_size=0.4, random_state=13)
# divide the test_validation into 50% 50% validation and test
X_validation, X_test, Y_validation, Y_test = train_test_split(X_testValid, Y_testValid, test_size=0.5, random_state=38)            

# extract the feature for the training, validation, and test
train_features = getFeaturesList(X_train)
validation_features = getFeaturesList(X_validation)
test_features = getFeaturesList(X_test)


# H1 is hidden dimension; D_out is output dimension.
N, D_in, H1, D_out = train_features.shape[0], train_features.shape[1], 128, 9

# holds the input of NN
x = torch.tensor(train_features).double()
# labels of NN output to be compared with
y = torch.tensor(Y_train)

# holds the validation input of NN into tensor
x_validation = torch.tensor(validation_features).double()
# holds the validation output of NN into tensor
Y_validation = torch.tensor(Y_validation)

# holds the test input of NN into tensor
x_test = torch.tensor(test_features).double()
# holds the test output of NN into tensor
y_test = torch.tensor(Y_test)

# set the device to be cpu or cuda of applicable
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the network architecture
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),    
    torch.nn.Linear(H1, D_out),
    torch.nn.LogSoftmax(dim=1)
)

# loss function of the network
loss_fn = torch.nn.NLLLoss()

# learning rate of the network
learning_rate = 2*1e-5
model.to(device)
model.double()
# adam's opimizer for the training
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# holds the losses of the training to be printed
training_losses = []
validation_losses = []

# reset the model parameters
for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()

# training loop
for t in range(200000):
    x = x.to(device)
    y = y.to(device)
    
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    
    # every 500 iteration show the loss of training
    if t % 500 == 499:
        print(t+1, loss.item(),"  [Training]")
        training_losses.append(loss.item())
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # every 500 iteration show the loss of validation without update
    if t % 500 == 499:
        with torch.no_grad():
            x_valid = x_validation.to(device)
            y_valid = Y_validation.to(device)
            y_pred2 = model(x_valid)
            loss = loss_fn(y_pred2, y_valid)
            print(t+1, loss.item(),"  [Validation]" )
            validation_losses.append(loss.item())

# plot both training the validtion losses
plt.plot(training_losses, '-r')
plt.plot(validation_losses, '-g')
plt.show()         

# calculate the accuracy of the model on training, validation, test
with torch.no_grad():
    y_pred = model(x)
    y_pred = torch.argmax(torch.exp(y_pred), dim=1)
    acc = (torch.sum(y_pred==y).item() / len(y)) * 100
    print("Training Accuracy = ", acc, "%")    

    x_valid = x_validation.to(device)
    y_valid = Y_validation.to(device)
    y_pred = model(x_valid)
    y_pred = torch.argmax(torch.exp(y_pred), dim=1)
    acc = (torch.sum(y_pred==y_valid).item() / len(y_valid)) * 100
    print("Validation Accuracy = ", acc, "%")

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    y_pred = model(x_test)
    y_pred = torch.argmax(torch.exp(y_pred), dim=1)
    acc = (torch.sum(y_pred==y_test).item() / len(y_test)) * 100
    print("Test Accuracy = ", acc, "%")

# saving the model parameters
torch.save(model, "model_test.pt")    