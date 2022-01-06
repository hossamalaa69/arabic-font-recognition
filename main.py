from Utils import *

# Defining main function
def main():
    
    # reading data folder that contains all test images
    data = readTestSet()
    # loading the model of the classifier
    model = pickle.load(open('RF_model.pkl', 'rb'))  

    # open output files (results, time)
    f_results = open("results.txt", "w")
    f_time = open("time.txt", "w")
    
    # iterate over all images in order
    for img in data:
        # start the timer
        start_time = time.time()
        # 1-apply binarization and text cropping 
        img = pre_processing(img)
        # 2-get image features
        feature = getFeatures(img).reshape(1, -1)
        # 3-predict image class
        y_class = model.predict(feature) + 1
        # stop the timer
        end_time = time.time()

        # print results of current image
        f_results.write(str(y_class[0]) + "\n")
        f_time.write(str(round((end_time-start_time), 2)) + "\n")

    # closing the files
    f_results.close()
    f_time.close()

if __name__=="__main__":
    main()