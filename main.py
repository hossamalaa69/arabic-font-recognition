import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from skimage.morphology import binary_erosion
from sklearn.mixture import GaussianMixture



def getBorderColor(img):
    border = np.asarray(img[0,:])
    border = np.concatenate((border, np.asarray(img[-1, :])))
    border = np.concatenate((border, np.asarray(img[:, 0])))
    border = np.concatenate((border, np.asarray(img[:, -1])))
    return np.bincount(border).argmax()

def pre_processing(img):
    _, img_binarized = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    if(getBorderColor(img_binarized) != 0):
        img_binarized = cv2.bitwise_not(img_binarized) 
    return img_binarized

# pre processing of the image (baseline estimation)
def baseline_estimatator(img):
    horz_proj = np.sum(img, 1)
    lb = np.argmax(horz_proj)
    avg_row_density = np.average(horz_proj)
    for inx, row_proj in enumerate(horz_proj):
        if row_proj >= avg_row_density:
            lu = inx
            break
    return lb,lu

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(pre_processing(img))
    return images

def divide_data():
    #follow the ration 60% 20% 20%
    TRAINING_SIZE = 114
    VALIDATION_SIZE = 38
    TEST_SIZE = 38

    TRAINING_OFFSET = TRAINING_SIZE
    VALIDATION_OFFSET = TRAINING_OFFSET + VALIDATION_SIZE

    training_set = []
    validation_set = []
    test_set = []
    

    for i in range (1, 10):
        class_i_images = load_images_from_folder("ACdata_base/" + str(i))
        training_set.append(class_i_images[0:TRAINING_OFFSET])
        validation_set.append(class_i_images[TRAINING_OFFSET:VALIDATION_OFFSET])
        test_set.append(class_i_images[VALIDATION_OFFSET:])

    return training_set, validation_set, test_set
    
def showImage(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def leftUp(img, height):
    k1 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    k2 = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ])
    
    return np.sum(binary_erosion(255 - img, k2) * binary_erosion(img, k1)) / height


def rightUp(img, height):
    k1 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    k2 = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])    
    
    return np.sum(binary_erosion(255 - img, k2) * binary_erosion(img, k1)) / height

def rightDown(img, height):
    k1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    k2 = np.array([
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 0]
    ]) 
    return np.sum(binary_erosion(255 - img, k2) * binary_erosion(img, k1)) / height

def leftDown(img, height):
    k1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    k2 = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1]
    ]) 
    return np.sum(binary_erosion(255 - img, k2) * binary_erosion(img, k1)) / height

def vertical(img, height):
    k1 = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])
    k2 = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]) 
    return np.sum(binary_erosion(255 - img, k2) * binary_erosion(img, k1)) / height

def horizontal(img, height):
    k1 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]
    ])
    k2 = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])    
    return np.sum(binary_erosion(255 - img, k2) * binary_erosion(img, k1)) / height

def get_center_of_mass(window):
    # calculate moments of binary image
    M = cv2.moments(window)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / (M["m00"] + 1e-5))
    cY = int(M["m01"] / (M["m00"] + 1e-5))
    return cX,cY

def getImageFeatures(img, WIDTH, HEIGHT, STRID, N, baseline):
    lb, lu = baseline
    img_height = img.shape[0]
    img_width = img.shape[1]
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    f7 = []
    f8 = []
    f9 = []
    f10 = []
    f11 = []
    f12 = []
    f13 = []
    f14 = []
    f15 = []
    f16 = []
    f17 = []
    f18 = []
    f19 = []
    f20 = []
    f21 = []
    f22 = []
    f23 = []
    f24 = []
    f25 = []
    f26 = []
    f27 = []
    f28 = []
    centers = []
    k = -1
    index = 0
    for c in range(0, img_height, HEIGHT):
        if c >= lb:
            k = c
            break
        index += 1
    for r in range(img_width - 1, WIDTH - 1, -STRID):
        window = img[:, r-WIDTH:r]
        centers.append(get_center_of_mass(window))
        f5.append(np.sum(img[0:lb-1, r-WIDTH:r] == 255) / (lb * WIDTH))
        f6.append(np.sum(img[lb:, r-WIDTH:r] == 255) / ((img_height - lb) * WIDTH))
        foreground = []
        cells = []
        for c in range(0, img_height, HEIGHT):
            window_2 = img[c:c+HEIGHT-1, r-WIDTH:r]
            foreground.append(np.sum(window_2 == 255) / (WIDTH * HEIGHT))
            cells.append(window_2)
        f1.append(np.sum(foreground))
        summtion = 0
        for i in range(2,N):
            bi = int(np.sum(cells[i] == 255) == 0)
            b_i = int(np.sum(cells[i - 1] == 255) == 0)
            summtion += np.abs(bi - b_i )
        f2.append(summtion)
        cells = []
        for c in range(0, k, HEIGHT):
            cells.append(img[c:c+HEIGHT-1, r-WIDTH:r])
        summtion = 0
        for i in range(2, len(cells)):
            bi = int(np.sum(cells[i] == 255) == 0)
            b_i = int(np.sum(cells[i - 1] == 255) == 0)
            summtion += np.abs(bi - b_i)
        f7.append(summtion)
        center_y = get_center_of_mass(window)[1]
        if center_y > lb:
            f8.append(3)
        elif center_y < lb and center_y > lu:
            f8.append(2)
        else:
            f8.append(1)
        f9.append(leftUp(window, img_height))
        f10.append(rightUp(window, img_height))
        f11.append(rightDown(window, img_height))
        f12.append(leftDown(window, img_height))
        f13.append(vertical(window, img_height))
        f14.append(horizontal(window, img_height))
        window = img[lu:lb, r-WIDTH:r]
        f15.append(leftUp(window, np.abs(lb-lu)))
        f16.append(rightUp(window, np.abs(lb-lu)))
        f17.append(rightDown(window, np.abs(lb-lu)))
        f18.append(leftDown(window, np.abs(lb-lu)))
        f19.append(vertical(window, np.abs(lb-lu)))
        f20.append(horizontal(window, np.abs(lb-lu)))
        window = img[lu:lb, r-WIDTH:r]
        f21.append(np.sum(window[0, :]) / img_height)
        f22.append(np.sum(window[1, :]) / img_height)
        f23.append(np.sum(window[2, :]) / img_height)
        f24.append(np.sum(window[3, :]) / img_height)
        f25.append(np.sum(window[4, :]) / img_height)
        f26.append(np.sum(window[5, :]) / img_height)
        f27.append(np.sum(window[6, :]) / img_height)
        f28.append(np.sum(window[7, :]) / img_height)


    f3.append(np.sqrt(np.power(centers[0][0], 2) + np.power(centers[0][1], 2)))
    for i in range(1,len(centers)):
        f3.append(np.sqrt(np.power(centers[i][0] - centers[i - 1][0],
                        2) + np.power(centers[i][1] - centers[i - 1][1], 2)))
    for i in range(0, len(centers)):
        f4.append(np.sqrt(np.power(centers[i][1] - lb, 2) / img_height))


def gaussianMixtureModel(n, random_state, X, X_test):
    gm = GaussianMixture(n_components=n, random_state=random_state).fit(X)
    return gm.predict(X_test)
# feature extraction parameters
# number of cells according to paper guidlines
N = 20
# width of kernal according to paper guidlines
WIDTH = 8 
# a paramter of my selection
STRID = 4

training_set, validation_set, test_set = divide_data()
image = load_images_from_folder("ACdata_base/1")[0]
getImageFeatures(image, WIDTH, int(image.shape[0] / N), STRID, N, baseline_estimatator(image))
