from sklearn.model_selection import train_test_split
import torch
from Utils import *

# reading the whole dataset image and apply pre-processing
# the data_set now contains array of images that are binarized
# and cropped and pre-processed 
data_set, Y = readDataSet()

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