from Utils import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Defining main function
def main():
    
    # reading data folder that contains all test images
    data = readTestSet()
    # loading the model of the classifier
    clf2 = pickle.load(open('SVM_model.pkl', 'rb'))  

    # open output files (results, time)
    f_results = open("results.txt", "w")
    f_time = open("time.txt", "w")
    
    # iterate over all images in order
    for img in data:
        # open timer
        start_time = time.time()
        # get image features
        feature = getFeatures(img).reshape(1, -1)
        # predict image class
        y_class = clf2.predict(feature) + 1
        # stop timer
        end_time = time.time()
        
        # print results of current image
        f_results.write(str(y_class[0]) + "\n")
        f_time.write(str(round((end_time-start_time), 2)) + "\n")

    # closing the files
    f_results.close()
    f_time.close()


if __name__=="__main__":
    main()