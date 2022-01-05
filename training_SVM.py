from Utils import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# reading the whole dataset image and apply pre-processing
# the data_set now contains array of images that are binarized
# and cropped and pre-processed 
data_set, Y = readDataSet()
# split the dataset into 80% 20% training test
X_train, X_testValid, Y_train, Y_testValid = train_test_split(data_set, Y, test_size=0.2, random_state=60)            
# get the features of the training and test data
# each matrix now contains M rows of examples and
# 255 columns of LPQ features
features_train = getFeaturesList(X_train)
features_test = getFeaturesList(X_testValid)
# fitting SVM model to the training features
X = np.array(features_train)
y = np.array(Y_train)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)
# predict the test data and print accuracy
y_pred = (clf.predict(features_test))
acc = np.mean(y_pred == Y_testValid) * 100
print(acc)