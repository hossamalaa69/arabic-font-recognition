from Utils import *

# Defining main function
def main():
    # reading data folder that contains all test images
    data = readTestSet(folder=sys.argv[1])
    # loading the model of the classifier
    model = pickle.load(open('RF_model.pkl', 'rb'))  

    # open output files (results, time)
    f_results = open(os.path.join(sys.argv[2],"results.txt"), "w")
    f_time = open(os.path.join(sys.argv[2],"times.txt"), "w")
    
    # iterate over all images in order
    for i in range(len(data)):
        try:
            # start the timer
            start_time = time.time()
            # 1-apply binarization and text cropping 
            img = pre_processing(data[i])
            # 2-get image features
            feature = getFeatures(img).reshape(1, -1)
            # 3-predict image class
            y_class = model.predict(feature) + 1
            # stop the timer
            end_time = time.time()
            delta_time = str(round((end_time-start_time), 2))
            if delta_time == "0.00":
                delta_time = "0.01"
            # print results of current image
            f_results.write(str(y_class[0]))
            f_time.write(delta_time)
        
        except:
            f_results.write("-1")
            f_time.write("0.01") 
            print("An exception occured at i =", i)       

        if i+1 != len(data):
            f_results.write("\n")
            f_time.write("\n")

    # closing the files
    f_results.close()
    f_time.close()

if __name__=="__main__":
    main()