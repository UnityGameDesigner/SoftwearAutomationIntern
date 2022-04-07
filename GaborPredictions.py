
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tqdm
from sklearn import svm, metrics

from matplotlib import style


DATADIR = "C:\GHP_FinalProject\Train"
DATADIR1 = "C:\GHP_FinalProject\Test"

CATEGORIES = ["CokeCan", "Banana"]
CATEGORIES1 = ["CokeCan", "Banana"]

IMG_SIZE = 170

training_data = []
test_data = []

orig_num = 0

total_perGabor = []
total_perGabor1 = []

dataperGabor1 = []

num_filters = 18

def create_training_data():
    for category in CATEGORIES:  #two catagories are exterior and interior

        path = os.path.join(DATADIR,category)  # create a path to the exterior and interior folders in the train foler
        class_num = CATEGORIES.index(category)  # set an index for exterior and interior, 0 is exterior and 1 is interior

        for img in os.listdir(path):  #iterate over each image
            try:

                new_array = cv2.imread(os.path.join(path,img))# read the image, cv2. imread converts this image to an array
                
                new_array = cv2.resize(new_array, (IMG_SIZE, IMG_SIZE))  #resize the image, this is not necessary but is nice if images are different sizes
                #new_array = new_array[340:540, 340:540]
                for i in range(num_filters):

                    total = 0
                    g_kernel = cv2.getGaborKernel((20,20), 1, i*np.pi/num_filters, 7, 2, 0, ktype=cv2.CV_32F)
                    filtered_img = cv2.filter2D(new_array, cv2.CV_8UC3, g_kernel)
                    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
                  
                    for x in range(IMG_SIZE):
                        for y in range(IMG_SIZE):
                            total = total + filtered_img[x,y]
                    total_perGabor.append(total)
                            
                training_data.append([total_perGabor, class_num])
                
                
            except Exception as e:  # in the interest in keeping the output clean...
                pass


def create_test_data():
    for category in CATEGORIES1:  # do dogs and cats

        path1 = os.path.join(DATADIR1,category)  # create path to dogs and cats
        class_num1 = CATEGORIES1.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        print("creating_test_data")
        for img in os.listdir(path1):  # iterate over each image per dogs and cats
            try:

                
                new_array = cv2.imread(os.path.join(path1,img))# read the image, cv2. imread converts this image to an array
                
                #new_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                new_array = cv2.resize(new_array, (IMG_SIZE, IMG_SIZE))  #resize the image, this is not necessary but is nice if images are different sizes
                #new_array = new_array[340:540, 340:540]
                #cv2.imshow(new_array)

                for i in range(num_filters):

                    total = 0
                    g_kernel = cv2.getGaborKernel((20,20), 1, i*np.pi/num_filters, 7, 2, 0, ktype=cv2.CV_32F)
                    filtered_img = cv2.filter2D(new_array, cv2.CV_8UC3, g_kernel)
                    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

                    #ret,filtered_img = cv2.threshold(filtered_img,70,255,cv2.THRESH_BINARY)
                    
                    for x in range(IMG_SIZE):
                        for y in range(IMG_SIZE):
                            total = total + filtered_img[x,y]
                    test_data.append(total)
                       

            except Exception as e:  # in the interest in keeping the output clean...
                pass




create_test_data()
X1 = []

X3 = []

for features in test_data:
    X3.append(features)


X1 = list(zip(*[iter(X3)]*num_filters))


X1 = np.array(X1)

n_samples = len(X1)

orig_num = int(orig_num/30)
X1 = X1.reshape((n_samples, -1))


create_training_data()
X = []
y = []
num_img = 0

for features, label in training_data:
    X.append(features)
    y.append(label)
    num_img = num_img +1



X = np.array(X)


finallist = []
LastElement = X[-1]
for i in LastElement:
    finallist.append(i)
    

finallist = list(zip(*[iter(finallist)]*num_filters))

finallist = np.array(finallist)


n_samples = len(finallist)
finallist = finallist.reshape((n_samples, -1))



clf = svm.SVC(kernel = 'linear', C = 1.0)
print("start classifying values")
clf.fit(finallist,y)
print("classified values")


predictions = clf.predict(X1)
print(predictions)



for i in range(len(predictions)):
    if( predictions[i] == 1 ):
        print("The", int(i)+1, "image is a banana")
    if( predictions[i] == 0 ):
        print("The", int(i) + 1, "image is a coke can")

