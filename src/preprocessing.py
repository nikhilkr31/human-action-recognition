import os
import cv2 as cv
import glob

#create resized directory in the current project folder
parent_dir = os.getcwd()
directory_train = 'resized/train_set'
directory_test = 'resized/test_set'
path_train = os.path.join(parent_dir, directory_train)
os.mkdir(path_train)
path_test = os.path.join(parent_dir, directory_test)
os.mkdir(path_test)
print('Created "resized" folder')

#Declaring the dimensions
# 224x224x3 is standard size for VGG16 training data
width = 224
height = 224
dim = (width, height)

#Resize and store function

def resize_store(img, filename, dim, path):
    '''This function will take an image, resize it and output into '/resized/' folder in the same directory'''
    
    #Read the images in a list
    train_img = [cv.imread(file) for file in glob.glob(os.getcwd() + "/data/train/*.jpg")]
    



    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    cv.imwrite(path + filename ,resized)

#Resizing the train set images
