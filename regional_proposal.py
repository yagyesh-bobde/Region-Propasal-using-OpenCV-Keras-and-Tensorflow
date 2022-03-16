# import the packages
import numpy as np
import cv2
from tensorflow.keras.applications import Xception  # this is the model we'll be using for object detection
from tensorflow.keras.applications.resnet50 import preprocess_input # for preprocessing the input
from tensorflow.keras.applications import imagenet_utils 
from tensorflow.keras.preprocessing.image import img_to_array 
from imutils.object_detection import non_max_suppression



# instanciate the selective search segmentation algorithm of opencv
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image) # set the base image as the input image

search.switchToSelectiveSearchFast() # you can also use this for more accuracy -> search.switchToSelectiveSearchQuality()

rects = ss.process()
